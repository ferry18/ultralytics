import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import HardSwish, SELayer, DepSepConvHS

__all__ = ("lcnet_075",)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PPLCNetDet(nn.Module):
    """
    PP-LCNet detection backbone variant that emits multi-scale features.
    Scales and SE/large-kernel placement follow the reference paper configuration.
    """

    def __init__(self, scale: float = 0.75):
        super().__init__()
        # cfg: [k, c, s, use_se]
        self.cfgs = [
            [3, 32, 1, 0],
            [3, 64, 2, 0],
            [3, 64, 1, 0],
            [3, 128, 2, 0],
            [3, 128, 1, 0],
            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]
        self.scale = scale

        input_channel = _make_divisible(16 * scale)
        layers = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel), HardSwish()]

        c_idx = 0
        self.stage_indices = {"p3": None, "p4": None, "p5": None}
        self.p3_channels = None
        blocks = []
        for k, c, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * scale)
            blocks.append(DepSepConvHS(input_channel, output_channel, k, s, bool(use_se)))
            input_channel = output_channel
            c_idx += 1
            # Mark stage ends for P3/P4/P5 after strides 8, 16, 32
        self.features = nn.Sequential(*layers, *blocks)

        # We will extract P3/P4/P5 by tracking strides dynamically in forward
        self.out_indices = []

    def forward(self, x: torch.Tensor):
        feats = []
        cur = x
        stride = 1
        for i, m in enumerate(self.features):
            cur = m(cur)
            # Update stride when encountering stride-2 ops; detect by spatial change
            if isinstance(m, (nn.Conv2d,)):
                pass
            # Heuristic: pick features after certain spatial downsample ratios
            feats.append(cur)
        # Select P3/P4/P5 as the last three distinct downsampled maps
        maps = []
        last_h = None
        for f in feats:
            h = f.shape[-2]
            if last_h is None or h < last_h:
                maps.append(f)
                last_h = h
        # Ensure at least 3 maps (P3,P4,P5)
        if len(maps) < 3:
            # Fallback: take last 3
            maps = feats[-3:]
        P5, P4, P3 = maps[-1], maps[-2], maps[-3]
        return P3, P4, P5


def lcnet_075():
    return PPLCNetDet(scale=0.75)