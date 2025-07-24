"""
Area-weighted Wasserstein Loss for LWMP-YOLO.
Based on the paper "LWMP-YOLO: A Lightweight Multi-scale Small Target Detection Algorithm for UAVs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalizedWassersteinDistance(nn.Module):
    """Normalized Wasserstein Distance for bounding box similarity measurement."""
    
    def __init__(self, C=10.0):
        super().__init__()
        self.C = C  # Normalization constant
    
    def forward(self, pred_boxes, target_boxes):
        """
        Calculate normalized Wasserstein distance between predicted and target boxes.
        
        Args:
            pred_boxes: [N, 4] tensor (cx, cy, w, h)
            target_boxes: [N, 4] tensor (cx, cy, w, h)
        
        Returns:
            nwd: [N] tensor of normalized Wasserstein distances
        """
        # Convert box representation to Gaussian parameters
        # Center points
        pred_mu = pred_boxes[:, :2]
        target_mu = target_boxes[:, :2]
        
        # Covariance matrices (diagonal)
        # Sigma = diag(w^2/4, h^2/4)
        pred_sigma = torch.stack([
            pred_boxes[:, 2]**2 / 4,
            pred_boxes[:, 3]**2 / 4
        ], dim=-1)
        
        target_sigma = torch.stack([
            target_boxes[:, 2]**2 / 4,
            target_boxes[:, 3]**2 / 4
        ], dim=-1)
        
        # Calculate 2D Wasserstein distance
        # W2^2 = ||μ_A - μ_B||^2 + ||√Σ_A - √Σ_B||_F^2
        center_dist = torch.sum((pred_mu - target_mu)**2, dim=-1)
        
        # For diagonal covariance matrices, Frobenius norm simplifies
        sigma_dist = torch.sum((torch.sqrt(pred_sigma) - torch.sqrt(target_sigma))**2, dim=-1)
        
        w2_squared = center_dist + sigma_dist
        
        # Normalized Wasserstein distance
        nwd = torch.exp(-torch.sqrt(w2_squared) / np.sqrt(self.C))
        
        return nwd


class AreaWeightedWassersteinLoss(nn.Module):
    """Area-weighted Wasserstein Loss for small object detection."""
    
    def __init__(self, C=10.0, area_weight_power=2.0):
        super().__init__()
        self.nwd = NormalizedWassersteinDistance(C)
        self.area_weight_power = area_weight_power
    
    def forward(self, pred_boxes, target_boxes, target_areas=None):
        """
        Calculate area-weighted Wasserstein loss.
        
        Args:
            pred_boxes: [N, 4] tensor (cx, cy, w, h) normalized to [0, 1]
            target_boxes: [N, 4] tensor (cx, cy, w, h) normalized to [0, 1]
            target_areas: [N] tensor of target box areas (optional)
        
        Returns:
            loss: scalar loss value
        """
        # Calculate normalized Wasserstein distance
        nwd = self.nwd(pred_boxes, target_boxes)
        
        # Box loss = 1 - NWD
        box_loss = 1 - nwd
        
        # Apply area-based weighting if provided
        if target_areas is not None:
            # Sigmoid mapping for area weights
            # Smaller targets get higher weights
            area_weights = torch.sigmoid(-self.area_weight_power * (target_areas - 0.1))
            box_loss = box_loss * area_weights
        
        # Add scale difference term
        pred_wh = pred_boxes[:, 2:4]
        target_wh = target_boxes[:, 2:4]
        scale_diff = torch.abs(pred_wh - target_wh).sum(dim=-1)
        
        # Combined loss
        loss = box_loss.mean() + 0.1 * scale_diff.mean()
        
        return loss


class LWMPYOLOLoss(nn.Module):
    """Complete loss function for LWMP-YOLO including AWLoss."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        
        # Loss components
        self.box_loss = AreaWeightedWassersteinLoss(C=10.0)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_cls = 0.5
        self.lambda_obj = 1.0
        
    def forward(self, preds, batch):
        """
        Calculate total loss.
        
        Args:
            preds: Model predictions
            batch: Batch data including targets
            
        Returns:
            loss: Total loss
            loss_items: Individual loss components
        """
        # This is a simplified version - in practice, you would need to:
        # 1. Match predictions to targets using anchor assignment
        # 2. Calculate losses only for positive matches
        # 3. Apply proper loss scaling
        
        # For now, return a placeholder
        device = preds[0].device if isinstance(preds, list) else preds.device
        loss = torch.tensor(0.0, device=device)
        
        return loss, torch.zeros(3, device=device)  # box, cls, obj losses