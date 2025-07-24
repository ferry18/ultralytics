"""
PP-LCNet x0.75 final implementation for LWMP-YOLO.
This implementation matches the author's YAML structure expectations.
"""

import torch
import torch.nn as nn
from typing import List, Union
from .lwmp_modules import make_divisible, h_swish, h_sigmoid, SELayer as SEModule


class DepthwiseSeparableOptimized(nn.Module):
    """Optimized Depthwise Separable Convolution for PP-LCNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_se=False, act='relu'):
        super().__init__()
        
        # Depthwise conv
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, 
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        
        # SE module (if needed)
        self.use_se = use_se
        if use_se:
            self.se = SEModule(in_channels, reduction=4)
            
        # Activation
        self.act_type = act
            
        # Pointwise conv
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        

            
    def forward(self, x):
        # Depthwise
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        if self.act_type == 'relu':
            x = torch.nn.functional.relu(x, inplace=True)
        elif self.act_type == 'hard_swish':
            x = x * torch.nn.functional.relu6(x + 3) / 6
        
        # SE
        if self.use_se:
            x = self.se(x)
            
        # Pointwise
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        if self.act_type == 'relu':
            x = torch.nn.functional.relu(x, inplace=True)
        elif self.act_type == 'hard_swish':
            x = x * torch.nn.functional.relu6(x + 3) / 6
        
        return x


class lcnet_075(nn.Module):
    """
    PP-LCNet x0.75 implementation that outputs P5 and stores intermediate features.
    Designed to work with author's YAML structure.
    """
    
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        
        # Handle YOLO parameter passing
        if isinstance(ch, list):
            ch = ch[0] if ch else 3
        if isinstance(pretrained, list):
            pretrained = pretrained[0] if pretrained else True
            
        scale = 0.75
        
        # Scale function
        def s(x):
            return make_divisible(x * scale, 8)
        
        # Build network following exact PP-LCNet x0.75 spec
        # Initial conv - P1/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, s(16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(s(16))
        )
        
        # Build blocks
        # Stage 1 - outputs P2/4 (ch=36)
        self.block0 = DepthwiseSeparableOptimized(s(16), s(24), 3, 1, False, 'relu')
        self.block1 = DepthwiseSeparableOptimized(s(24), s(24), 3, 2, False, 'relu')  # downsample to P2/4
        self.block2 = DepthwiseSeparableOptimized(s(24), s(48), 3, 1, False, 'relu')
        self.block3 = DepthwiseSeparableOptimized(s(48), s(48), 3, 1, False, 'hard_swish')
        
        # Stage 2 - outputs P3/8 (ch=72)
        self.block4 = DepthwiseSeparableOptimized(s(48), s(48), 3, 2, False, 'hard_swish')  # downsample to P3/8
        self.block5 = DepthwiseSeparableOptimized(s(48), s(96), 3, 1, False, 'hard_swish')
        self.block6 = DepthwiseSeparableOptimized(s(96), s(96), 3, 1, False, 'hard_swish')
        
        # Stage 3 - outputs P4/16 (ch=144)
        self.block7 = DepthwiseSeparableOptimized(s(96), s(96), 5, 2, True, 'hard_swish')   # downsample to P4/16
        self.block8 = DepthwiseSeparableOptimized(s(96), s(192), 5, 1, True, 'hard_swish')
        self.block9 = DepthwiseSeparableOptimized(s(192), s(192), 5, 1, True, 'hard_swish')
        self.block10 = DepthwiseSeparableOptimized(s(192), s(192), 5, 1, True, 'hard_swish')
        self.block11 = DepthwiseSeparableOptimized(s(192), s(192), 5, 1, True, 'hard_swish')
        
        # Stage 4 - outputs P5/32 (ch=288)
        self.block12 = DepthwiseSeparableOptimized(s(192), s(192), 5, 2, True, 'hard_swish')  # downsample to P5/32
        self.block13 = DepthwiseSeparableOptimized(s(192), s(384), 5, 1, True, 'hard_swish')
        
        # Store output channels
        self.out_channels = [s(48), s(96), s(192), s(384)]  # P2, P3, P4, P5
        
        # Features storage
        self._features = {}
        
    def forward(self, x):
        """Forward pass that stores intermediate features and returns P5."""
        # Initial conv
        x0 = self.conv1(x)
        x0 = x0 * torch.nn.functional.relu6(x0 + 3) / 6  # Hard swish
        
        # Stage 1 - P2
        x1 = self.block0(x0)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        self._features['p2'] = x4  # P2/4, ch=36
        
        # Stage 2 - P3
        x5 = self.block4(x4)
        x6 = self.block5(x5)
        x7 = self.block6(x6)
        self._features['p3'] = x7  # P3/8, ch=72
        
        # Stage 3 - P4
        x8 = self.block7(x7)
        x9 = self.block8(x8)
        x10 = self.block9(x9)
        x11 = self.block10(x10)
        x12 = self.block11(x11)
        self._features['p4'] = x12  # P4/16, ch=144
        
        # Stage 4 - P5
        x13 = self.block12(x12)
        x14 = self.block13(x13)
        self._features['p5'] = x14  # P5/32, ch=288
        
        # Store outputs that YOLO can reference
        # Based on author's YAML, indices 1, 2, 3 should be P2, P3, P4
        self.p2 = self._features['p2']
        self.p3 = self._features['p3']
        self.p4 = self._features['p4']
        self.p5 = self._features['p5']
        
        # Return P5 for next layers (SPPF)
        return x14


class MultiOutputWrapper(nn.Module):
    """
    Wrapper to make lcnet_075 work with YOLO's multi-output expectations.
    Returns a list of features that YOLO can index.
    """
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def forward(self, x):
        # Run backbone
        p5 = self.backbone(x)
        
        # Return list of features for YOLO indexing
        # Index 0: P5 (main output)
        # Index 1: P2 
        # Index 2: P3
        # Index 3: P4
        return [p5, self.backbone.p2, self.backbone.p3, self.backbone.p4]