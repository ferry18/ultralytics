import torch
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss

__all__ = ("AWDetectionLoss",)


class AWDetectionLoss(v8DetectionLoss):
    """Adaptive weighted detection loss described in LWMP-YOLO paper.

    L = Σ (1/(2σ_i^2)) * L_i  +  log(1+σ_i^2)
    The learnable variables are σ_i (one per detection scale).
    """

    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)
        # log_sigma to guarantee positivity
        self.log_sigma = nn.Parameter(torch.zeros(3))

    def combine(self, losses: torch.Tensor):
        # losses: tensor of shape (3,)  [box,cls,dfl] aggregated by scale
        sigma2 = self.log_sigma.exp().pow(2)
        weighted = (losses / (2.0 * sigma2)).sum() + torch.log1p(sigma2).sum()
        return weighted

    def __call__(self, preds, batch):
        total, items = super().__call__(preds, batch)  # items detach
        # items is tensor of 3 elements (box, cls, dfl)
        total = self.combine(items)
        return total, items.detach()