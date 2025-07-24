"""
Integration of AWLoss with YOLOv8/v11 detection loss system.
This module provides a drop-in replacement for v8DetectionLoss.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from .loss import BboxLoss, TaskAlignedAssigner, make_anchors, xywh2xyxy
from .metrics import bbox_iou
from .ops import dist2bbox
from ..nn.losses.awloss import AWLoss, NormalizedWassersteinDistance, AreaWeighting, ScaleDifference


class v8DetectionAWLoss:
    """
    YOLOv8/v11 detection loss with AWLoss integration.
    Replaces IoU-based box loss with Normalized Wasserstein Distance.
    """

    def __init__(self, model, tal_topk: int = 10):
        """Initialize v8DetectionAWLoss with model parameters."""
        device = next(model.parameters()).device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Task-aligned assigner for target assignment
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        
        # Original bbox loss for DFL component
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # AWLoss components
        self.nwd = NormalizedWassersteinDistance(C=getattr(h, 'nwd_C', 10.0))
        self.area_weight = AreaWeighting(
            alpha=getattr(h, 'area_alpha', 10.0),
            beta=getattr(h, 'area_beta', 0.5)
        )
        self.scale_diff = ScaleDifference()
        self.scale_weight = getattr(h, 'scale_weight', 0.05)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def xyxy2xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from xyxy to xywh format."""
        x1, y1, x2, y2 = boxes.split(1, dim=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.cat([cx, cy, w, h], dim=-1)

    def compute_awloss(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, 
                      target_scores: torch.Tensor, fg_mask: torch.Tensor, 
                      imgsz: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute AWLoss for bounding box regression.
        
        Args:
            pred_bboxes: Predicted boxes in xyxy format [batch, anchors, 4]
            target_bboxes: Target boxes in xyxy format [batch, anchors, 4]
            target_scores: Target scores [batch, anchors, nc]
            fg_mask: Foreground mask [batch, anchors]
            imgsz: Image size tensor [h, w]
            
        Returns:
            box_loss: Scalar box loss
            loss_items: Dictionary of loss components for logging
        """
        # Convert to xywh format for NWD computation
        pred_xywh = self.xyxy2xywh(pred_bboxes)
        target_xywh = self.xyxy2xywh(target_bboxes)
        
        # Get valid predictions (foreground)
        fg_mask_expanded = fg_mask.unsqueeze(-1).expand_as(pred_xywh)
        pred_fg = pred_xywh[fg_mask_expanded].view(-1, 4)
        target_fg = target_xywh[fg_mask_expanded].view(-1, 4)
        
        if pred_fg.shape[0] == 0:
            # No foreground predictions
            return torch.tensor(0., device=self.device), {}
        
        # Compute NWD
        nwd_scores = self.nwd(pred_fg.unsqueeze(0), target_fg.unsqueeze(0)).squeeze(0)
        box_loss = 1.0 - nwd_scores
        
        # Apply area-based weighting
        area_weights = self.area_weight(target_fg.unsqueeze(0), imgsz.tolist()).squeeze(0)
        box_loss = box_loss * area_weights
        
        # Add scale difference term
        scale_loss = self.scale_diff(pred_fg.unsqueeze(0), target_fg.unsqueeze(0)).squeeze(0)
        box_loss = box_loss + self.scale_weight * scale_loss
        
        # Weight by target scores
        target_scores_fg = target_scores[fg_mask].max(dim=1)[0]  # Get max score across classes
        box_loss = (box_loss * target_scores_fg).sum() / target_scores_fg.sum().clamp(min=1)
        
        # Prepare loss items for logging
        loss_items = {
            'nwd_mean': nwd_scores.mean().detach(),
            'area_weight_mean': area_weights.mean().detach(),
            'scale_loss_mean': scale_loss.mean().detach()
        }
        
        return box_loss, loss_items

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Target assignment using TaskAlignedAssigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss (standard BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Box loss using AWLoss
        if fg_mask.sum():
            # Scale target boxes back to feature map scale
            target_bboxes_scaled = target_bboxes / stride_tensor
            
            # Compute AWLoss
            awloss, loss_items = self.compute_awloss(
                pred_bboxes, 
                target_bboxes_scaled,
                target_scores,
                fg_mask,
                imgsz
            )
            
            loss[0] = awloss
            
            # DFL loss (if using DFL)
            if self.use_dfl:
                # Extract DFL component from original bbox loss
                _, dfl_loss = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, 
                    target_bboxes_scaled, target_scores, target_scores_sum, fg_mask
                )
                loss[2] = dfl_loss

        # Apply gains
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()


def create_awloss_model(model):
    """
    Create a model with AWLoss by replacing the standard loss computation.
    
    Args:
        model: YOLO model instance
        
    Returns:
        model: Modified model with AWLoss
    """
    # Store original loss class
    original_loss_class = model.model[-1].loss_class if hasattr(model.model[-1], 'loss_class') else None
    
    # Replace with AWLoss
    model.model[-1].loss_class = v8DetectionAWLoss
    
    # Store reference to original for potential restoration
    model.model[-1]._original_loss_class = original_loss_class
    
    return model