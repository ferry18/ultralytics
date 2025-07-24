"""
PP-LCNet x0.75 implementation using split outputs for YOLO.
This approach creates separate output branches that YOLO can reference.
"""

import torch
import torch.nn as nn
from .lwmp_modules import DepthwiseSeparable, SEModule, HardSwish, make_divisible


class PPLCNetBackbone(nn.Module):
    """PP-LCNet x0.75 backbone with accessible intermediate features."""
    
    def __init__(self, in_channels: int = 3, scale: float = 0.75):
        super().__init__()
        
        # Scale function
        def s(c):
            return make_divisible(c * scale, 8)
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, s(16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(s(16)),
            HardSwish()
        )
        
        # Stage 1 - P2 output (ch=36)
        self.stage1_1 = DepthwiseSeparable(s(16), s(24), 3, 1, use_se=False, activation='relu')
        self.stage1_2 = DepthwiseSeparable(s(24), s(24), 3, 2, use_se=False, activation='relu')  # downsample
        self.stage1_3 = DepthwiseSeparable(s(24), s(48), 3, 1, use_se=False, activation='relu')
        self.stage1_4 = DepthwiseSeparable(s(48), s(48), 3, 1, use_se=False, activation='hard_swish')
        
        # Stage 2 - P3 output (ch=72)
        self.stage2_1 = DepthwiseSeparable(s(48), s(48), 3, 2, use_se=False, activation='hard_swish')  # downsample
        self.stage2_2 = DepthwiseSeparable(s(48), s(96), 3, 1, use_se=False, activation='hard_swish')
        self.stage2_3 = DepthwiseSeparable(s(96), s(96), 3, 1, use_se=False, activation='hard_swish')
        
        # Stage 3 - P4 output (ch=144)
        self.stage3_1 = DepthwiseSeparable(s(96), s(96), 5, 2, use_se=True, activation='hard_swish')  # downsample
        self.stage3_2 = DepthwiseSeparable(s(96), s(192), 5, 1, use_se=True, activation='hard_swish')
        self.stage3_3 = DepthwiseSeparable(s(192), s(192), 5, 1, use_se=True, activation='hard_swish')
        self.stage3_4 = DepthwiseSeparable(s(192), s(192), 5, 1, use_se=True, activation='hard_swish')
        self.stage3_5 = DepthwiseSeparable(s(192), s(192), 5, 1, use_se=True, activation='hard_swish')
        
        # Stage 4 - P5 output (ch=288)
        self.stage4_1 = DepthwiseSeparable(s(192), s(192), 5, 2, use_se=True, activation='hard_swish')  # downsample
        self.stage4_2 = DepthwiseSeparable(s(192), s(384), 5, 1, use_se=True, activation='hard_swish')
        
        # Store features
        self.features = {}
        
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Stage 1
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1_3(x)
        x = self.stage1_4(x)
        self.features['p2'] = x  # P2/4
        
        # Stage 2
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage2_3(x)
        self.features['p3'] = x  # P3/8
        
        # Stage 3
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3_4(x)
        x = self.stage3_5(x)
        self.features['p4'] = x  # P4/16
        
        # Stage 4
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        self.features['p5'] = x  # P5/32
        
        return x  # Return P5 for next layers


class PPLCNetSplitOutput(nn.Module):
    """
    Split output module that makes PP-LCNet features accessible to YOLO.
    This is a workaround to handle multi-scale features in YOLO's architecture.
    """
    
    def __init__(self, backbone, scale='p5'):
        super().__init__()
        self.backbone = backbone
        self.scale = scale
        
    def forward(self, x):
        # Backbone has already processed x, just return the stored feature
        if hasattr(self.backbone, 'features') and self.scale in self.backbone.features:
            return self.backbone.features[self.scale]
        else:
            # Fallback - this shouldn't happen in normal operation
            return x


class lcnet_075(nn.Module):
    """Main PP-LCNet x0.75 module for YOLO."""
    
    def __init__(self, ch=3, *args, **kwargs):
        super().__init__()
        if isinstance(ch, list):
            ch = ch[0] if ch else 3
            
        self.backbone = PPLCNetBackbone(in_channels=ch, scale=0.75)
        
    def forward(self, x):
        return self.backbone(x)


class lcnet_p2(nn.Module):
    """Extract P2 feature from PP-LCNet."""
    
    def __init__(self, backbone_idx):
        super().__init__()
        self.backbone_idx = backbone_idx
        
    def forward(self, x, model_outputs):
        # Access the backbone from model outputs
        backbone = model_outputs[self.backbone_idx]
        if hasattr(backbone, 'features') and 'p2' in backbone.features:
            return backbone.features['p2']
        return x


class lcnet_p3(nn.Module):
    """Extract P3 feature from PP-LCNet."""
    
    def __init__(self, backbone_idx):
        super().__init__()
        self.backbone_idx = backbone_idx
        
    def forward(self, x, model_outputs):
        # Access the backbone from model outputs
        backbone = model_outputs[self.backbone_idx]
        if hasattr(backbone, 'features') and 'p3' in backbone.features:
            return backbone.features['p3']
        return x


class lcnet_p4(nn.Module):
    """Extract P4 feature from PP-LCNet."""
    
    def __init__(self, backbone_idx):
        super().__init__()
        self.backbone_idx = backbone_idx
        
    def forward(self, x, model_outputs):
        # Access the backbone from model outputs
        backbone = model_outputs[self.backbone_idx]
        if hasattr(backbone, 'features') and 'p4' in backbone.features:
            return backbone.features['p4']
        return x