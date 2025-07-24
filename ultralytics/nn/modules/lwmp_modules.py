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


class DepSepConv(nn.Module):
    """Depthwise Separable Convolution for PP-LCNet."""
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                h_swish(),
                # SE
                SELayer(inp, reduction=4),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                h_swish(),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                h_swish(),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                h_swish()
            )

    def forward(self, x):
        return self.conv(x)


class LCBackbone(nn.Module):
    """PP-LCNet based lightweight backbone for YOLO."""
    def __init__(self, scale=0.75, in_channels=3):
        super(LCBackbone, self).__init__()
        
        self.scale = scale
        
        # PP-LCNet configuration from the paper
        # [kernel_size, channels, stride, use_SE]
        self.cfgs = [
            [3,  32, 1, 0],
            [3,  64, 2, 0],
            [3,  64, 1, 0],
            [3, 128, 2, 0],
            [3, 128, 1, 0],
            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]
        
        # Build first layer
        input_channel = make_divisible(16 * scale, 8)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            h_swish()
        )
        
        # Build PP-LCNet blocks
        self.stages = nn.ModuleList()
        self.stage_out_indices = []  # Store indices where we'll extract features
        
        for i, (k, c, s, use_se) in enumerate(self.cfgs):
            output_channel = make_divisible(c * scale, 8)
            self.stages.append(DepSepConv(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
            
            # Mark feature extraction points (after downsampling)
            if i in [2, 4, 12]:  # After 64x1, 128x1, 512x1
                self.stage_out_indices.append(i)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # First conv
        x = self.first_conv(x)
        
        # Extract features at different scales
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.stage_out_indices:
                features.append(x)
        
        # Return P3, P4, P5 features for YOLO
        return features[0], features[1], features[2]


class lcnet_075(nn.Module):
    """LCNet model with 0.75x width multiplier wrapped for YOLO integration."""
    def __init__(self, c1=3, c2=None, pretrained=True):
        super().__init__()
        self.backbone = LCBackbone(scale=0.75, in_channels=c1)
        # Store feature maps for multi-scale output
        self.features = []
        
        # Calculate output channels
        with torch.no_grad():
            x = torch.zeros(1, c1, 256, 256)
            p3, p4, p5 = self.backbone(x)
            self.out_channels = [p3.shape[1], p4.shape[1], p5.shape[1]]
            # YOLO expects c2 to be the output channel of the last feature
            self.c = p5.shape[1]
    
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        # Store features for later access in head
        self.features = [p3, p4, p5]
        # Return P5 for the next layer (SPPF)
        return p5