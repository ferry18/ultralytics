"""Custom selector module for LMWP-YOLO."""
import torch
import torch.nn as nn
from typing import List

class Selector(nn.Module):
    """Select a specific tensor from a list based on index. Custom module to avoid Ultralytics Index parsing issues."""
    
    def __init__(self, index=0):
        """Initialize selector with index."""
        super().__init__()
        self.index = index
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Select and return the tensor at the specified index."""
        return x[self.index]

# Aliases for different indices
class Select0(nn.Module):
    """Select first tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[0]

class Select1(nn.Module):
    """Select second tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[1]

class Select2(nn.Module):
    """Select third tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[2]
