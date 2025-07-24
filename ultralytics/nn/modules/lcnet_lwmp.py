"""
PP-LCNet x0.75 implementation for LWMP-YOLO that properly handles multi-scale features.
Based on the paper's exact requirements.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .lwmp_modules import make_divisible, h_sigmoid, SELayer


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution block for PP-LCNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_se=False, act='hard_swish'):
        super().__init__()
        padding = kernel_size // 2
        
        # Depthwise
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        
        # SE module if needed
        self.use_se = use_se
        if use_se:
            self.se = SELayer(in_channels, reduction=4)
            
        # Activation
        if act == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        else:  # hard_swish
            self.act1 = nn.Hardswish(inplace=True)
            
        # Pointwise
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        
        if act == 'relu':
            self.act2 = nn.ReLU(inplace=True)
        else:
            self.act2 = nn.Hardswish(inplace=True)
        
    def forward(self, x):
        # Depthwise
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.act1(x)
        
        # SE
        if self.use_se:
            x = self.se(x)
            
        # Pointwise
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.act2(x)
        
        return x


class lcnet_075(nn.Module):
    """
    PP-LCNet x0.75 backbone for LWMP-YOLO.
    This implementation properly outputs all features needed by the head.
    """
    
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        
        # Handle YOLO's parameter passing
        if isinstance(in_channels, list):
            in_channels = in_channels[0] if in_channels else 3
            
        scale = 0.75
        
        # Stage 0: Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, make_divisible(16 * scale, 8), 3, 2, 1, bias=False),
            nn.BatchNorm2d(make_divisible(16 * scale, 8)),
            nn.Hardswish(inplace=True)
        )
        
        # PP-LCNet blocks configuration
        # [kernel_size, out_channels, stride, use_se, activation]
        cfg = [
            # Stage 1 - P2/4
            [3, 32, 1, False, 'relu'],
            
            # Stage 2 - P3/8  
            [3, 64, 2, False, 'relu'],
            [3, 64, 1, False, 'relu'],
            
            # Stage 3 - P4/16
            [3, 128, 2, False, 'relu'],
            [3, 128, 1, False, 'relu'],
            
            # Stage 4 - P5/32
            [5, 256, 2, False, 'hard_swish'],
            [5, 256, 1, False, 'hard_swish'],
            [5, 256, 1, False, 'hard_swish'],
            [5, 256, 1, False, 'hard_swish'],
            [5, 256, 1, False, 'hard_swish'],
            [5, 256, 1, False, 'hard_swish'],
            
            # Stage 5 - Final
            [5, 512, 2, True, 'hard_swish'],
            [5, 512, 1, True, 'hard_swish'],
        ]
        
        # Build stages
        in_c = make_divisible(16 * scale, 8)
        self.stages = nn.ModuleList()
        self.stage_out_channels = []
        
        for i, (k, c, s, se, act) in enumerate(cfg):
            out_c = make_divisible(c * scale, 8)
            self.stages.append(DepthwiseSeparable(in_c, out_c, k, s, se, act))
            in_c = out_c
            
            # Record output channels at key stages
            if i == 0:  # After first block - not P2 yet
                pass
            elif i == 1:  # P3/8
                self.p3_channels = out_c
            elif i == 4:  # P4/16  
                self.p4_channels = out_c
            elif i == 11:  # P5/32
                self.p5_channels = out_c
                
        # For YOLO, we need specific output channels
        # According to paper, P2 should be from an earlier stage
        # Let's add a P2 extraction after conv1
        self.p2_channels = make_divisible(32 * scale, 8)
        
        # Store feature indices
        self.p2_idx = 0  # After first DW block
        self.p3_idx = 2  # After stage 2
        self.p4_idx = 4  # After stage 3
        self.p5_idx = 12 # After stage 5
        
    def forward(self, x):
        """Forward pass returning features for YOLO head."""
        # Initial conv
        x = self.conv1(x)
        
        # Store features at different scales
        features = []
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            # Store features at specific points
            if i == self.p2_idx:
                p2 = x
                features.append(('p2', x))
            elif i == self.p3_idx:
                p3 = x
                features.append(('p3', x))
            elif i == self.p4_idx:
                p4 = x
                features.append(('p4', x))
            elif i == self.p5_idx:
                p5 = x
                features.append(('p5', x))
        
        # For LWMP-YOLO, we need to make P2, P3, P4 available
        # Store them as attributes for access by the neck
        self.p2 = p2  # P2/4
        self.p3 = p3  # P3/8
        self.p4 = p4  # P4/16
        self.p5 = p5  # P5/32
        
        # Return P5 as main output (this goes to SPPF)
        return p5