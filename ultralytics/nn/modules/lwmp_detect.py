"""
Lightweight detection head for LWMP-YOLO.
"""

import torch
import torch.nn as nn
from .conv import Conv, DWConv
from .modules import Detect


class LightweightDetect(Detect):
    """Lightweight detection head using depthwise separable convolutions."""
    
    def __init__(self, nc=80, ch=()):
        """Initialize lightweight detection head with reduced parameters."""
        super().__init__(nc, ch)
        
        # Replace standard convolutions with depthwise separable
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        
        for x in ch:
            # Use depthwise separable for bbox regression (4 outputs)
            # Original: Conv(x, c2, 3)
            # Now: DWConv -> 1x1 Conv
            c2 = max(16, x // 4)  # Reduce intermediate channels
            self.cv2.append(nn.Sequential(
                DWConv(x, c2, 3),
                Conv(c2, 4 * self.reg_max, 1)
            ))
            
            # Use 1x1 conv for classification
            self.cv3.append(Conv(x, self.nc, 1))
            
        # Override parent's cv2 and cv3
        self.m = nn.ModuleList(self.cv2 + self.cv3)