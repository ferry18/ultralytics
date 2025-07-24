"""
Correct LWMP-YOLO modules implementation matching paper specifications.
Target: 1.23M parameters, 2.71MB model size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for extreme lightweight design."""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        
        # Depthwise convolution
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        
        # Pointwise convolution
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


class SEModuleLightweight(nn.Module):
    """Lightweight SE module for PP-LCNet."""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use smaller intermediate channels
        mid_channel = max(channel // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, mid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LightweightMAFR(nn.Module):
    """Lightweight Multi-scale Adaptive Feature Refinement."""
    def __init__(self, channels):
        super().__init__()
        # Reduce internal channels for lightweight design
        mid_channels = channels // 2
        
        # Multi-dimensional collaborative attention (simplified)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Lightweight multi-scale feature fusion
        # Use depthwise separable convolutions
        self.scale1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.scale3 = DepthwiseSeparableConv(channels, channels // 4, 3, act=False)
        self.scale5 = DepthwiseSeparableConv(channels, channels // 4, 5, act=False)
        self.scale7 = DepthwiseSeparableConv(channels, channels // 4, 7, act=False)
        
        self.fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(channels)
        
        # Mini residual block (simplified)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        
        # Multi-scale features
        s1 = self.scale1(x)
        s3 = self.scale3(x)
        s5 = self.scale5(x)
        s7 = self.scale7(x)
        
        # Concatenate and fuse
        ms = torch.cat([s1, s3, s5, s7], dim=1)
        ms = self.fusion(ms)
        ms = self.fusion_bn(ms)
        
        # Residual connection
        out = self.residual(ms) + x
        
        return out


class PPLCNetBlock(nn.Module):
    """PP-LCNet basic block with depthwise separable convolution."""
    def __init__(self, inp, oup, kernel_size, stride, use_se=False):
        super().__init__()
        self.use_se = use_se
        
        # Depthwise convolution
        self.dw = nn.Conv2d(inp, inp, kernel_size, stride, 
                           kernel_size//2, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        
        # SE module (if used)
        if use_se:
            self.se = SEModuleLightweight(inp, reduction=4)
        
        # Pointwise convolution
        self.pw = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        
        # Use ReLU6 for lightweight
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        identity = x
        
        # Depthwise
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # SE
        if self.use_se:
            x = self.se(x)
        
        # Pointwise
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        
        # Skip connection if same shape
        if identity.shape == x.shape:
            x = x + identity
            
        return x


class PPLCNetBackbone(nn.Module):
    """Lightweight PP-LCNet backbone for LWMP-YOLO."""
    def __init__(self, scale=0.5, in_channels=3):
        super().__init__()
        
        # PP-LCNet configuration with smaller scale
        # [kernel_size, channels, stride, use_SE]
        cfgs = [
            [3,  16, 1, 0],
            [3,  24, 2, 0],  # P2/4
            [3,  24, 1, 0],
            [3,  48, 2, 0],  # P3/8
            [3,  48, 1, 0],
            [5,  96, 2, 0],  # P4/16
            [5,  96, 1, 0],
            [5,  96, 1, 0],
            [5,  96, 1, 0],
            [5, 192, 2, 1],  # P5/32 with SE
            [5, 192, 1, 1],
        ]
        
        # First conv
        input_channel = int(16 * scale)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        self.stage_indices = []
        
        for i, (k, c, s, use_se) in enumerate(cfgs):
            output_channel = int(c * scale)
            self.blocks.append(PPLCNetBlock(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
            
            # Mark P3, P4, P5 output indices
            if i in [4, 8, 10]:  # After 48x1, 96x1, 192x1
                self.stage_indices.append(i)
        
    def forward(self, x):
        x = self.first_conv(x)
        
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_indices:
                features.append(x)
        
        return features  # [P3, P4, P5]


# Export functions
def build_lwmp_backbone(c1=3, scale=0.5):
    """Build LWMP backbone."""
    return PPLCNetBackbone(scale=scale, in_channels=c1)


def build_lwmp_mafr(channels):
    """Build LWMP MAFR module."""
    return LightweightMAFR(channels)