"""
PP-LCNet implementation for YOLO.
Based on the paper "PP-LCNet: A Lightweight CPU Convolutional Neural Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv


def make_divisible(v, divisor=8, min_value=None):
    """Ensure that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class HardSwish(nn.Module):
    """Hard Swish activation function."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for PP-LCNet."""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, make_divisible(channel // reduction), 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(make_divisible(channel // reduction), channel, 1, 1, 0)
        self.hardsigmoid = HardSigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hardsigmoid(x)
        return identity * x


class DepSepConv(nn.Module):
    """Depthwise Separable Convolution block for PP-LCNet."""
    def __init__(self, inp, oup, kernel_size, stride, use_se=False):
        super().__init__()
        
        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.act1 = HardSwish()
        
        # SE module
        self.use_se = use_se
        if use_se:
            self.se = SEModule(inp)
        
        # Pointwise convolution
        self.pwconv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.act2 = HardSwish()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        if self.use_se:
            x = self.se(x)
            
        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x


class PPLCNet(nn.Module):
    """PP-LCNet backbone implementation."""
    def __init__(self, scale=1.0, in_channels=3, out_indices=(2, 4, 12)):
        super().__init__()
        self.scale = scale
        self.out_indices = out_indices
        
        # PP-LCNet configuration
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
        input_channel = make_divisible(16 * scale)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish()
        )
        
        # Build PP-LCNet blocks
        self.stages = nn.ModuleList()
        
        for i, (k, c, s, use_se) in enumerate(self.cfgs):
            output_channel = make_divisible(c * scale)
            self.stages.append(DepSepConv(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        
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
        x = self.first_conv(x)
        
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                features.append(x)
        
        return features


class PPLCNet_x0_75(nn.Module):
    """PP-LCNet x0.75 for YOLO integration."""
    def __init__(self, c1=3, c2=None):
        super().__init__()
        self.backbone = PPLCNet(scale=0.75, in_channels=c1)
        
    def forward(self, x):
        features = self.backbone(x)
        # Return P3, P4, P5 features
        return features