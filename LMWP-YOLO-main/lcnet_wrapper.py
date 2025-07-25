"""
Custom LCNet wrapper for YAML compatibility.
"""

import torch.nn as nn
from ultralytics.nn.modules.lcnet import LCNet

class LCNetBackbone(nn.Module):
    """LCNet backbone wrapper that returns features as a list."""
    
    def __init__(self, ch=3):
        """Initialize LCNet backbone.
        
        Args:
            ch: Input channels (default 3 for RGB)
        """
        super().__init__()
        self.model = LCNet(in_ch=ch, out_ch=1024, scale=0.75)
        
    def forward(self, x):
        """Forward pass returning P5, P4, P3 as a list."""
        p5, p4, p3 = self.model(x)
        return [p5, p4, p3]  # Return as list for Index module
