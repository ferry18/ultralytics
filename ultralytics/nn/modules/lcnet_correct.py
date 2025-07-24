"""
Correct PP-LCNet implementation for LWMP-YOLO.
This implementation outputs P2, P3, P4 features instead of P3, P4, P5
to better detect small objects as per the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_divisible(v, divisor=8, min_value=None):
    """Ensure channel number is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """Hard Swish activation function."""
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for PP-LCNet."""
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
    """Depthwise Separable Convolution for PP-LCNet."""
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


class lcnet_075(nn.Module):
    """
    PP-LCNet x0.75 backbone for LWMP-YOLO.
    Modified to output P2, P3, P4 features for small object detection.
    """
    def __init__(self, pretrained=True, c1=3):
        super().__init__()
        scale = 0.75
        
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
            [5, 512, 2, 1],  # P5/32 (not used for output)
            [5, 512, 1, 1],
        ]
        
        # Build first layer
        input_channel = make_divisible(16 * scale, 8)
        self.first_conv = nn.Sequential(
            nn.Conv2d(c1, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish()
        )
        
        # Build stages
        stages = []
        for k, c, s, use_se in cfgs:
            output_channel = make_divisible(c * scale, 8)
            stages.append(DepSepConv(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        
        self.stages = nn.ModuleList(stages)
        
        # Important: We need to output features at specific indices
        # Index 2 (after 64x1) -> P2/4
        # Index 4 (after 128x1) -> P3/8  
        # Index 10 (after 256x1 block 5) -> P4/16
        self.out_indices = [2, 4, 10]
        
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
    
    def forward(self, x):
        x = self.first_conv(x)
        
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                features.append(x)
        
        # Store intermediate features for multi-scale detection
        if len(features) >= 3:
            self.p2 = features[0]  # P2/4 - 48 channels
            self.p3 = features[1]  # P3/8 - 96 channels  
            self.p4 = features[2]  # P4/16 - 192 channels
        
        # Return the last feature (P5/32) for SPPF
        # The head will access p2, p3, p4 through special indexing
        return x  # P5/32 - 384 channels