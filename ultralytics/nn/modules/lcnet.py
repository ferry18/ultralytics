import torch
import torch.nn as nn
from .conv import Conv, DWConv, h_swish  # reuse existing helper blocks

__all__ = ("LCNet", "lcnet_075")


class DepthwiseSeparable(nn.Module):
    """Depthwise conv + pointwise conv (1x1). Expansion ratio fixed to 4 (as in PP-LCNet)"""

    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super().__init__()
        hidden_ch = int(in_ch * 4)
        self.conv = nn.Sequential(
            # 1x1 expansion
            Conv(in_ch, hidden_ch, k=1, s=1, act=act),
            # 3x3 depth-wise
            DWConv(hidden_ch, hidden_ch, k=3, s=stride, act=act),
            # 1x1 projection
            Conv(hidden_ch, out_ch, k=1, s=1, act=False),
        )

    def forward(self, x):
        return self.conv(x)


class LCNet(nn.Module):
    """Native PyTorch implementation of PP-LCNet backbone trimmed for YOLO (no classifier head).

    Outputs only the last feature map (P5), with internal channels controlled by `out_ch`.
    Width multiplier adjusts every stage proportionally. P2/P3/P4 are not returned because
    this simplified YAML wraps additional PAN neck layers afterwards.
    """

    def __init__(self, in_ch=3, out_ch=1024, scale=0.75):
        super().__init__()
        self.scale = scale
        # channel helper
        def c(c0):
            return int(c0 * scale + 0.5)

        act = h_swish()
        self.stem = Conv(in_ch, c(16), k=3, s=2, act=nn.ReLU())  # stride 2

        self.stage2 = nn.Sequential(
            DepthwiseSeparable(c(16), c(32), stride=1, act=act),
            DepthwiseSeparable(c(32), c(32), stride=2, act=act),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(c(32), c(64), stride=1, act=act),
            DepthwiseSeparable(c(64), c(64), stride=2, act=act),
            DepthwiseSeparable(c(64), c(64), stride=1, act=act),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparable(c(64), c(128), stride=1, act=act),
            DepthwiseSeparable(c(128), c(128), stride=2, act=act),
            DepthwiseSeparable(c(128), c(128), stride=1, act=act),
            DepthwiseSeparable(c(128), c(128), stride=1, act=act),
        )
        self.stage5 = nn.Sequential(
            DepthwiseSeparable(c(128), c(256), stride=1, act=act),
            DepthwiseSeparable(c(256), out_ch, stride=2, act=act),  # stride 32 output
        )
        self.out_channels = out_ch

    def forward(self, x):
        x = self.stem(x)      # P1 /2
        x = self.stage2(x)    # /4
        x = self.stage3(x)    # /8
        x = self.stage4(x)    # /16
        x = self.stage5(x)    # /32
        return x  # only P5 for YOLO head


def lcnet_075(c2=1024, pretrained=False):
    """Factory used by YAML. `c2` is the desired output channels (default 1024)."""
    # pretrained flag reserved for future extension
    return LCNet(out_ch=c2, scale=0.75)