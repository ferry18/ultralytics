"""
PP-LCNet x0.75 implementation for YOLO integration.
This module handles multi-scale feature extraction for LWMP-YOLO.
"""

import torch
import torch.nn as nn
from .pplcnet_exact import PPLCNet_x075


class PPLCNetYOLO(nn.Module):
    """
    PP-LCNet backbone for YOLO that provides multi-scale features.
    This implementation stores intermediate features that can be accessed
    by YOLO's head layers through a special mechanism.
    """
    
    # Class variable to store features globally for YOLO access
    _features_store = {}
    
    def __init__(self, ch=3, model_id=None):
        super().__init__()
        self.backbone = PPLCNet_x075(in_channels=ch)
        
        # Unique ID for this model instance
        self.model_id = model_id or id(self)
        
        # Output channels from PP-LCNet x0.75
        # P2: 36, P3: 72, P4: 144, P5: 288
        self.out_channels = self.backbone.out_channels
        
        # Add final 1280-d conv as mentioned in paper
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.out_channels[-1], 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
    def forward(self, x):
        """Forward pass that stores multi-scale features."""
        # Get all features [P2, P3, P4, P5]
        features = self.backbone(x)
        
        # Apply final conv to P5
        p5_final = self.final_conv(features[-1])
        
        # Store features for YOLO head access
        PPLCNetYOLO._features_store[self.model_id] = {
            'p2': features[0],  # 36 channels
            'p3': features[1],  # 72 channels  
            'p4': features[2],  # 144 channels
            'p5': features[3],  # 288 channels (before final conv)
            'p5_final': p5_final  # 1280 channels
        }
        
        # Return P5 final for next layers (SPPF)
        return p5_final
    
    @classmethod
    def get_features(cls, model_id, scale):
        """Get stored features for a specific scale."""
        if model_id in cls._features_store:
            return cls._features_store[model_id].get(scale)
        return None


class FeatureExtractor(nn.Module):
    """
    Helper module to extract stored features from PP-LCNet.
    This allows YOLO's head to access P2, P3, P4 features.
    """
    def __init__(self, model_id, scale, channels):
        super().__init__()
        self.model_id = model_id
        self.scale = scale
        self.channels = channels
        
    def forward(self, x):
        """Extract the stored feature for this scale."""
        feature = PPLCNetYOLO.get_features(self.model_id, self.scale)
        if feature is not None:
            return feature
        else:
            # Fallback: return zeros with expected shape
            b, _, h, w = x.shape
            scale_factor = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}[self.scale]
            h_out = h // scale_factor
            w_out = w // scale_factor
            return torch.zeros(b, self.channels, h_out, w_out, device=x.device)


class lcnet_075(nn.Module):
    """
    Wrapper for PP-LCNet x0.75 that matches author's YAML expectations.
    This implementation outputs P5 and stores other features for the head.
    """
    
    # Shared storage for features across layers
    _shared_features = {}
    
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        
        # PP-LCNet backbone
        self.backbone = PPLCNet_x075(in_channels=ch)
        
        # Final conv to 1280 dimensions (as mentioned in paper)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels[-1], 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Store this instance's features
        self.instance_id = id(self)
        
    def forward(self, x):
        """Forward pass that stores features and returns P5."""
        # Get all features [P2, P3, P4, P5]
        features = self.backbone(x)
        
        # Store features for head access
        lcnet_075._shared_features[self.instance_id] = {
            'p2': features[0],  # 36ch @ P2/4
            'p3': features[1],  # 72ch @ P3/8  
            'p4': features[2],  # 144ch @ P4/16
            'p5': features[3],  # 288ch @ P5/32
        }
        
        # Apply final conv to P5
        p5_final = self.final_conv(features[3])
        
        # Return P5 for SPPF (1280 channels)
        return p5_final
    
    def get_feature(self, scale):
        """Get a specific feature scale."""
        return lcnet_075._shared_features.get(self.instance_id, {}).get(scale)