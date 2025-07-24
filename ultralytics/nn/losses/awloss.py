"""
Area-weighted Wasserstein Loss (AWLoss) implementation for LWMP-YOLO.
Based on the paper's equations 10-18 and area-based dynamic weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalizedWassersteinDistance(nn.Module):
    """
    Compute Normalized Wasserstein Distance (NWD) between bounding boxes.
    Models bounding boxes as 2D Gaussian distributions.
    """
    
    def __init__(self, C=10.0):
        """
        Initialize NWD module.
        
        Args:
            C: Normalization constant (default=10.0, to be tuned based on dataset)
        """
        super().__init__()
        self.C = C
        
    def forward(self, pred_boxes, target_boxes):
        """
        Compute NWD between predicted and target boxes.
        
        Args:
            pred_boxes: Predicted boxes [batch, n, 4] in format (cx, cy, w, h)
            target_boxes: Target boxes [batch, n, 4] in format (cx, cy, w, h)
            
        Returns:
            nwd: Normalized Wasserstein Distance [batch, n]
        """
        # Extract box components
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes.split(1, dim=-1)
        target_cx, target_cy, target_w, target_h = target_boxes.split(1, dim=-1)
        
        # Model as Gaussian: μ = [cx, cy], Σ = diag(w²/4, h²/4)
        # For 2D Wasserstein distance between Gaussians (Eq. 13):
        # W²₂(NA, NB) = ||[cxA, cyA, wA/2, hA/2]ᵀ - [cxB, cyB, wB/2, hB/2]ᵀ||²₂
        
        # Compute normalized coordinates
        pred_vec = torch.cat([pred_cx, pred_cy, pred_w/2, pred_h/2], dim=-1)
        target_vec = torch.cat([target_cx, target_cy, target_w/2, target_h/2], dim=-1)
        
        # Compute squared Wasserstein distance
        w2_squared = torch.sum((pred_vec - target_vec) ** 2, dim=-1)
        
        # Compute NWD (Eq. 14)
        nwd = torch.exp(-torch.sqrt(w2_squared) / self.C)
        
        return nwd


class AreaWeighting(nn.Module):
    """
    Dynamic area-based weighting for small object prioritization.
    Uses sigmoid mapping to assign higher weights to smaller targets.
    """
    
    def __init__(self, alpha=10.0, beta=0.5):
        """
        Initialize area weighting module.
        
        Args:
            alpha: Scaling factor for sigmoid function
            beta: Offset for sigmoid function
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, target_boxes, image_size):
        """
        Compute area-based weights.
        
        Args:
            target_boxes: Target boxes [batch, n, 4] in format (cx, cy, w, h)
            image_size: Tuple of (height, width) of the image
            
        Returns:
            weights: Area-based weights [batch, n]
        """
        # Extract width and height
        _, _, w, h = target_boxes.split(1, dim=-1)
        
        # Compute relative area (normalized by image area)
        img_h, img_w = image_size
        relative_area = (w * h) / (img_h * img_w)
        relative_area = relative_area.squeeze(-1)
        
        # Apply sigmoid mapping: smaller areas get higher weights
        # weight = 2 - sigmoid(alpha * (area - beta))
        # This ensures weight ∈ (1, 2) with smaller areas getting values closer to 2
        weights = 2.0 - torch.sigmoid(self.alpha * (relative_area - self.beta))
        
        return weights


class ScaleDifference(nn.Module):
    """
    Relative scale difference term for width and height.
    Explicitly quantifies dimensional differences.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_boxes, target_boxes):
        """
        Compute scale difference loss.
        
        Args:
            pred_boxes: Predicted boxes [batch, n, 4] in format (cx, cy, w, h)
            target_boxes: Target boxes [batch, n, 4] in format (cx, cy, w, h)
            
        Returns:
            scale_loss: Scale difference loss [batch, n]
        """
        # Extract width and height
        _, _, pred_w, pred_h = pred_boxes.split(1, dim=-1)
        _, _, target_w, target_h = target_boxes.split(1, dim=-1)
        
        # Compute relative scale differences
        # Using log scale to handle different magnitudes
        w_diff = torch.abs(torch.log(pred_w + 1e-7) - torch.log(target_w + 1e-7))
        h_diff = torch.abs(torch.log(pred_h + 1e-7) - torch.log(target_h + 1e-7))
        
        # Combine width and height differences
        scale_loss = (w_diff + h_diff).squeeze(-1)
        
        return scale_loss


class AWLoss(nn.Module):
    """
    Area-weighted Wasserstein Loss for LWMP-YOLO.
    Combines NWD, area weighting, and scale difference.
    """
    
    def __init__(self, 
                 C=10.0,
                 area_alpha=10.0,
                 area_beta=0.5,
                 scale_weight=0.05,
                 box_weight=7.5,
                 cls_weight=0.5,
                 obj_weight=1.0):
        """
        Initialize AWLoss.
        
        Args:
            C: NWD normalization constant
            area_alpha: Area weighting sigmoid scale
            area_beta: Area weighting sigmoid offset
            scale_weight: Weight for scale difference term
            box_weight: Weight for box loss (λbox)
            cls_weight: Weight for classification loss (λcls)
            obj_weight: Weight for objectness loss (λobj)
        """
        super().__init__()
        
        # Loss components
        self.nwd = NormalizedWassersteinDistance(C)
        self.area_weight = AreaWeighting(area_alpha, area_beta)
        self.scale_diff = ScaleDifference()
        
        # Loss weights
        self.scale_weight = scale_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        
        # Standard losses for classification and objectness
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, predictions, targets, image_size=(640, 640)):
        """
        Compute AWLoss.
        
        Args:
            predictions: Dict with keys:
                - 'box': Predicted boxes [batch, n, 4]
                - 'cls': Classification logits [batch, n, num_classes]
                - 'obj': Objectness logits [batch, n]
            targets: Dict with keys:
                - 'box': Target boxes [batch, n, 4]
                - 'cls': Target classes [batch, n, num_classes] (one-hot)
                - 'obj': Target objectness [batch, n] (0 or 1)
                - 'mask': Valid target mask [batch, n]
            image_size: Tuple of (height, width)
            
        Returns:
            loss: Total loss
            loss_items: Dict of individual loss components
        """
        pred_boxes = predictions['box']
        pred_cls = predictions['cls']
        pred_obj = predictions['obj']
        
        target_boxes = targets['box']
        target_cls = targets['cls']
        target_obj = targets['obj']
        mask = targets['mask']
        
        # Box loss with NWD (Eq. 15)
        nwd_scores = self.nwd(pred_boxes, target_boxes)
        box_loss = 1.0 - nwd_scores
        
        # Area-based dynamic weighting
        area_weights = self.area_weight(target_boxes, image_size)
        box_loss = box_loss * area_weights
        
        # Scale difference term
        scale_loss = self.scale_diff(pred_boxes, target_boxes)
        box_loss = box_loss + self.scale_weight * scale_loss
        
        # Apply mask and reduce
        box_loss = (box_loss * mask).sum() / mask.sum().clamp(min=1)
        
        # Classification loss (Eq. 16)
        cls_loss = self.bce_cls(pred_cls, target_cls)
        cls_loss = (cls_loss.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        
        # Objectness loss (Eq. 17)
        obj_loss = self.bce_obj(pred_obj, target_obj)
        obj_loss = obj_loss.mean()
        
        # Total loss (Eq. 18)
        total_loss = (self.box_weight * box_loss + 
                     self.cls_weight * cls_loss + 
                     self.obj_weight * obj_loss)
        
        loss_items = {
            'box_loss': box_loss.detach(),
            'cls_loss': cls_loss.detach(),
            'obj_loss': obj_loss.detach(),
            'nwd_mean': nwd_scores[mask].mean().detach(),
            'area_weight_mean': area_weights[mask].mean().detach(),
            'scale_loss': (self.scale_weight * scale_loss * mask).sum().detach() / mask.sum().clamp(min=1)
        }
        
        return total_loss, loss_items


class YOLOv11AWLoss(nn.Module):
    """
    YOLO v11 detection loss with AWLoss integration.
    This replaces the standard YOLO loss computation.
    """
    
    def __init__(self, model, cfg):
        """
        Initialize YOLO AWLoss.
        
        Args:
            model: YOLO model
            cfg: Model configuration
        """
        super().__init__()
        
        # Get model parameters
        self.nc = cfg.get('nc', 80)  # number of classes
        self.no = self.nc + 5  # outputs per anchor (box[4] + obj[1] + classes[nc])
        self.stride = model.stride  # model strides
        
        # Initialize AWLoss
        self.awloss = AWLoss(
            C=cfg.get('nwd_C', 10.0),
            area_alpha=cfg.get('area_alpha', 10.0),
            area_beta=cfg.get('area_beta', 0.5),
            scale_weight=cfg.get('scale_weight', 0.05),
            box_weight=cfg.get('box_gain', 7.5),
            cls_weight=cfg.get('cls_gain', 0.5),
            obj_weight=cfg.get('obj_gain', 1.0)
        )
        
        # Anchor-related parameters
        self.balance = [4.0, 1.0, 0.4]  # P3-P5 layer balance
        self.ssi = 0  # stride 16 index
        
    def preprocess(self, predictions, targets, image_size):
        """
        Preprocess predictions and targets for AWLoss computation.
        
        Args:
            predictions: Raw model predictions
            targets: Ground truth targets
            image_size: Input image size
            
        Returns:
            Processed predictions and targets for AWLoss
        """
        # This is a simplified version - full implementation would handle:
        # - Anchor assignment
        # - Target matching
        # - Multi-scale processing
        # - Format conversion
        
        # For now, return placeholder
        # Full implementation would be integrated with YOLO's existing loss infrastructure
        raise NotImplementedError("Full YOLO preprocessing requires deep integration with YOLO's loss system")
        
    def forward(self, predictions, targets):
        """
        Compute YOLO loss with AWLoss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            loss: Total loss
            loss_items: Individual loss components
        """
        # This would integrate with YOLO's existing loss computation
        # replacing the IoU-based box loss with AWLoss
        
        # Placeholder for full implementation
        raise NotImplementedError("Full integration requires modification of YOLO's loss computation pipeline")