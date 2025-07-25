import torch
import torch.nn as nn
from .conv import Conv, DWConv  # removed h_swish import


# hard-swish activation
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.relu6(x + 3.0, inplace=self.inplace) / 6.0

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
        # Expose output channel count for Ultralytics parser compatibility
        self.out_channels = out_ch

    def forward(self, x):
        return self.conv(x)


class LCNet(nn.Module):
    """PP-LCNet backbone trimmed for YOLO. Accepts `in_ch` and `out_ch` to align with Ultralytics parser."""

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
            DepthwiseSeparable(c(256), out_ch, stride=2, act=act),  # stride 32 output
        )
        self.out_channels = out_ch

        # Optionally load weights here if supplied later (placeholder)
        if pretrained:
            pass  # TODO: load when available

    def forward(self, x):
        # Stem & early stages
        x = self.stem(x)      # P1 /2
        x = self.stage2(x)    # P2 /4

        # Produce multiscale feature maps
        p3 = self.stage3(x)   # P3 /8  (low-level features, smaller channels)
        p4 = self.stage4(p3)  # P4 /16
        p5 = self.stage5(p4)  # P5 /32 (highest-level features)

        # Return in order (high->low) as expected by the custom YAML
        return p5, p4, p3


class lcnet_075(LCNet):
    """Wrapper class so the YAML string 'lcnet_075' resolves to a nn.Module."""

    def __init__(self, in_ch, out_ch, pretrained=False):
        super().__init__(in_ch=in_ch, out_ch=out_ch, scale=0.75, pretrained=pretrained)

# keep function alias for convenience
def lcnet_075_fn(c2=1024, pretrained=False):
    return LCNet(in_ch=3, out_ch=c2, scale=0.75, pretrained=pretrained)

__all__ += ("lcnet_075_fn",)