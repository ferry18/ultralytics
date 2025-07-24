"""
lcnet_075 wrapper for LWMP-YOLO to match author's YAML structure.
"""

import torch
import torch.nn as nn
from .lcnet_pp import PPLCNet


class lcnet_075(nn.Module):
    """
    PP-LCNet x0.75 wrapper for YOLO integration.
    This module stores intermediate features (P2, P3, P4) that can be accessed
    by the YOLO head through special indexing in the YAML.
    """
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        self.backbone = PPLCNet(scale=0.75, in_channels=3)
        self.pretrained = pretrained
        
        # These will store the intermediate features
        self._features = {}
        
    def forward(self, x):
        # Get all features from backbone
        p2, p3, p4, p5 = self.backbone(x)
        
        # Store features for later access
        # The YOLO framework will look for these when building the head
        self._features = {
            'p2': p2,  # P2/4 - 48 channels (64 * 0.75)
            'p3': p3,  # P3/8 - 96 channels (128 * 0.75)
            'p4': p4,  # P4/16 - 192 channels (256 * 0.75)
            'p5': p5   # P5/32 - 384 channels (512 * 0.75)
        }
        
        # Return P5 as the main output (for SPPF)
        return p5