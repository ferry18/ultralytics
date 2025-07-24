"""
LWMP-YOLO specific modules for lightweight multi-scale object detection.
Based on the paper "LWMP-YOLO: A Lightweight Multi-scale Small Target Detection Algorithm for UAVs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv, autopad


class StdPool(nn.Module):
    """Standard deviation pooling layer."""
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std


class MCAGate(nn.Module):
    """Multi-dimensional Collaborative Attention Gate."""
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    """Multi-dimensional Collaborative Attention Layer."""
    def __init__(self, inp, no_spatial=True):
        """
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightweightMSFFM(nn.Module):
    """Lightweight Multi-Scale Feature Fusion Module."""
    def __init__(self, inp):
        super(LightweightMSFFM, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(inp, inp // 8, kernel_size=3, padding=1, groups=inp // 8)
        self.conv5 = nn.Conv2d(inp, inp // 8, kernel_size=5, padding=2, groups=inp // 8)
        self.conv7 = nn.Conv2d(inp, inp // 8, kernel_size=7, padding=3, groups=inp // 8)
        self.fusion = nn.Conv2d(inp // 2, inp, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inp)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x1, x3, x5, x7], dim=1)
        out = self.relu(self.fusion(out))
        out = self.se(out)
        return out + x  # Add residual connection


class MiniResidualBlock(nn.Module):
    """Mini Residual Block for gradient propagation."""
    def __init__(self, channels):
        super(MiniResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class MAFR(nn.Module):
    """Multi-scale Adaptive Feature Refinement module."""
    def __init__(self, inp, no_spatial=True):
        super(MAFR, self).__init__()
        self.mca = MCALayer(inp, no_spatial)
        self.lmsffm = LightweightMSFFM(inp)
        self.mini_res = MiniResidualBlock(inp)

    def forward(self, x):
        x_mca = self.mca(x)
        x_lmsffm = self.lmsffm(x_mca)
        x_out = self.mini_res(x_lmsffm)
        return x_out


class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish activation function."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class LCBlock(nn.Module):
    """Lightweight Convolution Block - building block for LCNet backbone."""
    def __init__(self, inp, oup, stride=1, se_ratio=0, act='swish'):
        super(LCBlock, self).__init__()
        
        self.use_res_connect = stride == 1 and inp == oup
        self.use_se = se_ratio > 0
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        
        # SE block
        if self.use_se:
            self.se = SELayer(inp, reduction=int(1/se_ratio))
        
        # Pointwise convolution
        self.pwconv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        
        # Activation
        if act == 'swish':
            self.act = h_swish()
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        identity = x
        
        # Depthwise
        out = self.dwconv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # SE
        if self.use_se:
            out = self.se(out)
        
        # Pointwise
        out = self.pwconv(out)
        out = self.bn2(out)
        
        # Residual
        if self.use_res_connect:
            out += identity
            
        return out


def make_divisible(v, divisor, min_value=None):
    """Ensure that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LCBackbone(nn.Module):
    """PP-LCNet based lightweight backbone for YOLO."""
    def __init__(self, scale=0.75, in_channels=3, act='swish'):
        super(LCBackbone, self).__init__()
        
        self.scale = scale
        
        # Define PP-LCNet architecture
        # Format: [kernel_size, exp_size, out_channels, use_se, act, stride]
        self.net_config = [
            # Stage 0
            [3, 16, 16, False, 'relu', 1],
            # Stage 1
            [3, 32, 24, False, 'relu', 2],
            [3, 36, 24, False, 'relu', 1],
            # Stage 2  
            [5, 72, 40, True, 'swish', 2],
            [5, 72, 40, True, 'swish', 1],
            # Stage 3
            [5, 120, 80, True, 'swish', 2],
            [5, 120, 80, True, 'swish', 1],
            [3, 200, 80, True, 'swish', 1],
            [3, 184, 80, True, 'swish', 1],
            [3, 184, 80, True, 'swish', 1],
            [3, 480, 112, True, 'swish', 1],
            [3, 672, 112, True, 'swish', 1],
            # Stage 4
            [5, 672, 160, True, 'swish', 2],
            [5, 960, 160, True, 'swish', 1],
            [5, 960, 160, True, 'swish', 1],
            [5, 960, 160, True, 'swish', 1],
            [5, 960, 160, True, 'swish', 1],
        ]
        
        # First conv
        self.conv1 = nn.Conv2d(in_channels, make_divisible(16 * scale, 8), 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(make_divisible(16 * scale, 8))
        self.act1 = h_swish()
        
        # Build stages
        self.stages = nn.ModuleList()
        self.stage_out_channels = []
        
        input_channel = make_divisible(16 * scale, 8)
        
        for i, (k, exp, c, se, act, s) in enumerate(self.net_config):
            output_channel = make_divisible(c * scale, 8)
            exp_channel = make_divisible(exp * scale, 8)
            
            # Build the block
            if i == 0:
                # First block doesn't use expansion
                block = LCBlock(input_channel, output_channel, s, se_ratio=0.25 if se else 0, act=act)
            else:
                # Regular block with expansion
                block = nn.Sequential(
                    # Expand
                    nn.Conv2d(input_channel, exp_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(exp_channel),
                    h_swish() if act == 'swish' else nn.ReLU(inplace=True),
                    # Depthwise
                    nn.Conv2d(exp_channel, exp_channel, k, s, k//2, groups=exp_channel, bias=False),
                    nn.BatchNorm2d(exp_channel),
                    h_swish() if act == 'swish' else nn.ReLU(inplace=True),
                    # SE
                    SELayer(exp_channel, reduction=4) if se else nn.Identity(),
                    # Project
                    nn.Conv2d(exp_channel, output_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(output_channel),
                )
            
            self.stages.append(block)
            input_channel = output_channel
            
            # Save output channels for each stage
            if i in [2, 4, 11, 16]:  # End of each stage
                self.stage_out_channels.append(output_channel)
        
        # Last conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, make_divisible(1280 * scale, 8), 1, 1, 0, bias=False),
            nn.BatchNorm2d(make_divisible(1280 * scale, 8)),
            h_swish(),
            nn.Conv2d(make_divisible(1280 * scale, 8), 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            h_swish()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # First conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Stages
        features = []
        for i, block in enumerate(self.stages):
            x = block(x)
            if i in [2, 4, 11, 16]:  # End of each stage
                features.append(x)
        
        # Last conv
        x = self.last_conv(x)
        features.append(x)
        
        # Return features for FPN
        # Return P3, P4, P5 (indices 1, 2, 4)
        return features[1], features[2], features[4]


class lcnet_075(nn.Module):
    """LCNet model with 0.75x width multiplier wrapped for YOLO integration."""
    def __init__(self, c1=3, pretrained=True):
        super().__init__()
        self.backbone = LCBackbone(scale=0.75, in_channels=c1)
        # Get output channels for P3, P4, P5
        self.out_channels = []
        with torch.no_grad():
            x = torch.zeros(1, c1, 256, 256)
            outs = self.backbone(x)
            self.out_channels = [o.shape[1] for o in outs]
    
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        return p5  # Return only P5 for the next layer, others will be accessed via indexing