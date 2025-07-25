import torch
import torch.nn as nn
from .conv import Conv, DWConv
from .lcnet import DepthwiseSeparable, h_swish

__all__ = ("lcnet_075_fixed",)


class LCNetFixed(nn.Module):
    """PP-LCNet backbone that outputs fixed channel counts regardless of width scaling."""

    def __init__(self, in_ch, out_ch, scale=0.75, pretrained=False):
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
            DepthwiseSeparable(c(256), 1024, stride=2, act=act),  # Always output 1024 channels
        )
        # Set fixed output channels
        self.out_channels = 1024  # Always 1024, ignore the out_ch parameter

    def forward(self, x):
        # Stem & early stages
        x = self.stem(x)      # P1 /2
        x = self.stage2(x)    # P2 /4

        # Produce multiscale feature maps
        p3 = self.stage3(x)   # P3 /8  (48 channels with scale=0.75)
        p4 = self.stage4(p3)  # P4 /16 (96 channels with scale=0.75)
        p5 = self.stage5(p4)  # P5 /32 (1024 channels fixed)

        # Return in order (high->low) as expected by the custom YAML
        return p5, p4, p3


class lcnet_075_fixed(LCNetFixed):
    """Fixed lcnet_075 that always outputs the correct channel counts."""

    def __init__(self, in_ch, out_ch, pretrained=False):
        super().__init__(in_ch=in_ch, out_ch=out_ch, scale=0.75, pretrained=pretrained)