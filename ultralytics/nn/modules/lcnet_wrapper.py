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
        
        # Store features as attributes so YOLO can access them
        # The author's YAML expects to access these via indices
        self.p2 = p2  # P2/4 - 48 channels  
        self.p3 = p3  # P3/8 - 96 channels
        self.p4 = p4  # P4/16 - 192 channels
        self.p5 = p5  # P5/32 - 384 channels
        
        # Return P5 as the main output
        # The YAML expects this for SPPF
        return p5