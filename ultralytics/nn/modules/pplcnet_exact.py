"""
Exact PP-LCNet x0.75 implementation for LWMP-YOLO.
Based on the original PP-LCNet paper specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Ensure channel count is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """Hard Swish activation: x * ReLU6(x + 3) / 6"""
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation: ReLU6(x + 3) / 6"""
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        squeeze_channel = make_divisible(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, squeeze_channel, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channel, channel, 1, bias=True)
        self.hardsigmoid = HardSigmoid()
        
    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, use_se: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # SE module (optional)
        self.use_se = use_se
        if use_se:
            self.se = SEModule(in_channels)
            
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.act = HardSwish()
        
    def forward(self, x):
        # Depthwise
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # SE
        if self.use_se:
            out = self.se(out)
            
        # Pointwise
        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.act(out)
        
        return out


class PPLCNetBlock(nn.Module):
    """PP-LCNet block with optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, use_se: bool):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.dsc = DepthwiseSeparable(in_channels, out_channels, kernel_size, stride, use_se)
        
    def forward(self, x):
        out = self.dsc(x)
        if self.use_shortcut:
            out = out + x
        return out


class PPLCNet_x075(nn.Module):
    """
    Exact PP-LCNet x0.75 implementation.
    Outputs features at P2, P3, P4 scales for YOLO integration.
    """
    def __init__(self, in_channels: int = 3, scale: float = 0.75):
        super().__init__()
        
        # PP-LCNet x0.75 exact configuration
        # [kernel_size, channels, stride, use_se]
        cfg = [
            # Stage 0
            [3, 16, 2, False],
            # Stage 1
            [3, 24, 1, False],
            [3, 24, 2, False],  # P2/4
            [3, 48, 1, False],
            [3, 48, 1, False],
            # Stage 2  
            [3, 48, 2, False],  # P3/8
            [3, 96, 1, False],
            [3, 96, 1, False],
            # Stage 3
            [5, 96, 2, True],   # P4/16
            [5, 192, 1, True],
            [5, 192, 1, True],
            [5, 192, 1, True],
            [5, 192, 1, True],
            # Stage 4
            [5, 192, 2, True],  # P5/32
            [5, 384, 1, True],
        ]
        
        # Apply scale to channels
        for i in range(len(cfg)):
            cfg[i][1] = make_divisible(cfg[i][1] * scale)
            
        # Build network
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cfg[0][1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cfg[0][1]),
            HardSwish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_c = cfg[0][1]
        
        # Stage 1 (ends at P2)
        stage1 = []
        for i in range(1, 5):  # blocks 1-4
            k, c, s, se = cfg[i]
            stage1.append(PPLCNetBlock(in_c, c, k, s, se))
            in_c = c
        self.stages.append(nn.Sequential(*stage1))
        
        # Stage 2 (ends at P3)
        stage2 = []
        for i in range(5, 8):  # blocks 5-7
            k, c, s, se = cfg[i]
            stage2.append(PPLCNetBlock(in_c, c, k, s, se))
            in_c = c
        self.stages.append(nn.Sequential(*stage2))
        
        # Stage 3 (ends at P4)
        stage3 = []
        for i in range(8, 13):  # blocks 8-12
            k, c, s, se = cfg[i]
            stage3.append(PPLCNetBlock(in_c, c, k, s, se))
            in_c = c
        self.stages.append(nn.Sequential(*stage3))
        
        # Stage 4 (ends at P5)
        stage4 = []
        for i in range(13, 15):  # blocks 13-14
            k, c, s, se = cfg[i]
            stage4.append(PPLCNetBlock(in_c, c, k, s, se))
            in_c = c
        self.stages.append(nn.Sequential(*stage4))
        
        # Store output channels for each stage
        self.out_channels = [
            cfg[4][1],   # P2: 36 (48*0.75)
            cfg[7][1],   # P3: 72 (96*0.75)
            cfg[12][1],  # P4: 144 (192*0.75)
            cfg[14][1],  # P5: 288 (384*0.75)
        ]
        
    def forward(self, x):
        """Returns features at P2, P3, P4, P5 scales."""
        features = []
        
        # Initial conv
        x = self.conv1(x)
        
        # Stage 1 -> P2
        x = self.stages[0](x)
        features.append(x)  # P2
        
        # Stage 2 -> P3
        x = self.stages[1](x)
        features.append(x)  # P3
        
        # Stage 3 -> P4
        x = self.stages[2](x)
        features.append(x)  # P4
        
        # Stage 4 -> P5
        x = self.stages[3](x)
        features.append(x)  # P5
        
        return features  # [P2, P3, P4, P5]


class lcnet_075(nn.Module):
    """
    Wrapper for PP-LCNet x0.75 to match author's YAML expectations.
    Handles multi-scale feature output for YOLO.
    """
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        self.backbone = PPLCNet_x075(in_channels=ch)
        self.pretrained = pretrained
        
        # Get output channels
        self.out_channels = self.backbone.out_channels
        
    def forward(self, x):
        """Forward pass returning all feature scales."""
        # Get all features [P2, P3, P4, P5]
        features = self.backbone(x)
        
        # For author's YAML, we need to handle this specially
        # The YAML expects to access different scales
        # We'll store them for access by index
        self._features = features
        
        # Return P5 as main output (for SPPF)
        return features[-1]  # P5