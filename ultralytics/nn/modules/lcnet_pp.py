"""
PP-LCNet implementation for LWMP-YOLO that properly integrates with YOLO architecture.
This implementation follows the author's design where P2 and P3 are used for detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, make_divisible(channel // reduction, 8), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(make_divisible(channel // reduction, 8), channel, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se=False):
        super().__init__()
        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2

        # Depthwise
        self.conv1 = nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.act1 = HardSwish()
        
        # SE
        self.use_se = use_se
        if use_se:
            self.se = SELayer(inp)
        
        # Pointwise
        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.act2 = HardSwish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        if self.use_se:
            x = self.se(x)
            
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x


class PPLCNet(nn.Module):
    """PP-LCNet backbone that outputs features for YOLO."""
    def __init__(self, scale=0.75, in_channels=3):
        super().__init__()
        
        # PP-LCNet configuration
        # [kernel_size, channels, stride, use_SE]
        cfgs = [
            [3,  32, 1, 0],
            [3,  64, 2, 0],  # P2/4
            [3,  64, 1, 0],
            [3, 128, 2, 0],  # P3/8
            [3, 128, 1, 0],
            [5, 256, 2, 0],  # P4/16
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 512, 2, 1],  # P5/32
            [5, 512, 1, 1],
        ]
        
        # Build first layer
        input_channel = make_divisible(16 * scale, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        for k, c, s, use_se in cfgs:
            output_channel = make_divisible(c * scale, 8)
            self.stages.append(DepSepConv(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
            
        # Store which stages output P2, P3, P4, P5
        self.p2_idx = 2   # After 64x1
        self.p3_idx = 4   # After 128x1
        self.p4_idx = 10  # After 256x1 (5th)
        self.p5_idx = 12  # After 512x1
        
    def forward(self, x):
        x = self.stem(x)
        
        # Store features at different scales
        p2, p3, p4, p5 = None, None, None, None
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.p2_idx:
                p2 = x
            elif i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
            elif i == self.p5_idx:
                p5 = x
                
        # Return P2, P3, P4 for the head (following author's design)
        # But also store P5 as the main output
        return p2, p3, p4, p5