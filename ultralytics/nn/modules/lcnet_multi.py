"""
PP-LCNet x0.75 implementation with multi-output support for YOLO.
This module returns multiple feature maps that YOLO can access via indices.
"""

import torch
import torch.nn as nn
from typing import List
from .lwmp_modules import DepthwiseSeparable, SEModule, HardSwish, make_divisible


class PPLCNetMultiOutput(nn.Module):
    """
    PP-LCNet x0.75 backbone that returns multiple feature maps.
    Exact implementation following the paper specifications.
    """
    
    def __init__(self, in_channels: int = 3, scale: float = 0.75):
        super().__init__()
        
        # PP-LCNet configuration for x0.75
        # [kernel_size, channels, stride, use_se]
        cfg = [
            # Stage 0
            [3, 16, 2, False],
            # Stage 1 - outputs P2
            [3, 24, 1, False],
            [3, 24, 2, False],  # P2/4
            [3, 48, 1, False],
            [3, 48, 1, False],
            # Stage 2 - outputs P3  
            [3, 48, 2, False],  # P3/8
            [3, 96, 1, False],
            [3, 96, 1, False],
            # Stage 3 - outputs P4
            [5, 96, 2, True],   # P4/16
            [5, 192, 1, True],
            [5, 192, 1, True],
            [5, 192, 1, True],
            [5, 192, 1, True],
            # Stage 4 - outputs P5
            [5, 192, 2, True],  # P5/32
            [5, 384, 1, True],
        ]
        
        # Scale channels
        def scale_channels(c):
            return make_divisible(c * scale, 8)
        
        # Build first conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, scale_channels(16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(scale_channels(16)),
            HardSwish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_c = scale_channels(16)
        
        # Stage indices for feature extraction
        self.stage_out_indices = [4, 7, 12, 14]  # P2, P3, P4, P5
        
        # Build all blocks
        blocks = []
        for i, (k, c, s, se) in enumerate(cfg):
            out_c = scale_channels(c)
            blocks.append(DepthwiseSeparable(
                in_c, out_c, k, s, 
                use_se=se,
                activation='hard_swish' if i > 2 else 'relu'  # Use ReLU for first 3 blocks
            ))
            in_c = out_c
            
        # Group into stages for feature extraction
        self.stage1 = nn.Sequential(*blocks[0:5])   # P2 output
        self.stage2 = nn.Sequential(*blocks[5:8])   # P3 output  
        self.stage3 = nn.Sequential(*blocks[8:13])  # P4 output
        self.stage4 = nn.Sequential(*blocks[13:15]) # P5 output
        
        # Output channels for each scale
        self.out_channels = [
            scale_channels(48),   # P2
            scale_channels(96),   # P3
            scale_channels(192),  # P4
            scale_channels(384),  # P5
        ]
        
    def forward(self, x):
        """
        Forward pass returning all feature scales.
        Returns list of [P2, P3, P4, P5] features.
        """
        # Initial conv
        x = self.conv1(x)
        
        # Stage 1 - P2
        p2 = self.stage1(x)
        
        # Stage 2 - P3
        p3 = self.stage2(p2)
        
        # Stage 3 - P4
        p4 = self.stage3(p3)
        
        # Stage 4 - P5
        p5 = self.stage4(p4)
        
        return [p2, p3, p4, p5]


class lcnet_075(nn.Module):
    """
    YOLO-compatible wrapper for PP-LCNet x0.75.
    This version properly handles multi-output for YOLO's indexing system.
    """
    
    def __init__(self, ch=3, *args, **kwargs):
        """
        Initialize lcnet_075.
        
        Args:
            ch: Input channels (default: 3)
            *args, **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        
        # Handle YOLO's parameter passing
        if isinstance(ch, list):
            ch = ch[0] if ch else 3
            
        # Create backbone
        self.backbone = PPLCNetMultiOutput(in_channels=ch, scale=0.75)
        
        # Channel info for YOLO
        self.out_channels = self.backbone.out_channels
        
        # Output indices that will be saved by YOLO
        self.out_indices = [0, 1, 2, 3]  # Save all outputs
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps [P2, P3, P4, P5]
        """
        return self.backbone(x)