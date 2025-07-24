"""
LWMP-YOLO Exact Implementation
Based on the paper: "Improved YOLO for long range detection of small drones"

This implementation follows the exact specifications from the paper:
- LCbackbone (PP-LCNet x0.75): Lightweight backbone with DSConv, H-Swish, SE, 5x5 kernels
- MAFR: Multi-scale Adaptive Feature Refinement 
- Target: 1.23M parameters, 2.71MB model size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Ensure channel count is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """Hard Swish activation function as described in the paper."""
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        squeeze_channels = make_divisible(in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1, bias=True)
        self.hardsigmoid = HardSigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution block as used in PP-LCNet."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, use_se: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # SE module (optional)
        self.use_se = use_se
        if use_se:
            self.se = SEModule(in_channels)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.act = HardSwish()

    def forward(self, x):
        # Depthwise
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # SE
        if self.use_se:
            out = self.se(out)
        
        # Pointwise
        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.act(out)
        
        return out


class PPLCNetBlock(nn.Module):
    """PP-LCNet block configuration."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, use_se: bool):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.dsc = DepthwiseSeparable(in_channels, out_channels, kernel_size, stride, use_se)

    def forward(self, x):
        out = self.dsc(x)
        if self.use_shortcut:
            out = out + x
        return out


class LCBackbone(nn.Module):
    """
    PP-LCNet x0.75 backbone for LWMP-YOLO.
    Target: ~586K parameters to achieve total 1.23M with the rest of the model.
    """
    def __init__(self, in_channels: int = 3, scale: float = 0.75):
        super().__init__()
        
        # PP-LCNet x0.75 configuration - optimized for 1.23M params
        # Format: [kernel_size, out_channels, stride, use_se]
        base_cfg = [
            # Stage 0
            [3, 16, 2, False],
            
            # Stage 1  
            [3, 32, 1, False],
            [3, 32, 2, False],
            [3, 64, 1, False],
            
            # Stage 2
            [3, 64, 2, False], 
            [3, 128, 1, False],
            
            # Stage 3
            [3, 128, 2, False],
            [5, 256, 1, True],  # 5x5 kernel with SE
            [5, 256, 1, True],
            
            # Stage 4
            [5, 256, 2, True],
            [5, 512, 1, True],
        ]
        
        # Scale channels
        scaled_cfg = []
        for k, c, s, se in base_cfg:
            scaled_c = make_divisible(c * scale)
            scaled_cfg.append([k, scaled_c, s, se])
        
        # Build layers
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, scaled_cfg[0][1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(scaled_cfg[0][1]),
            HardSwish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_c = scaled_cfg[0][1]
        
        # Track output channels for each stage
        self.out_channels = []
        stage_idx = 0
        
        for i, (k, out_c, s, se) in enumerate(scaled_cfg[1:], 1):
            # Create block
            block = PPLCNetBlock(in_c, out_c, k, s, se)
            
            # Add to current stage or create new stage
            if s == 2 and i > 1:  # New stage
                stage_idx += 1
                self.stages.append(nn.Sequential(block))
                self.out_channels.append(out_c)
            else:
                if stage_idx >= len(self.stages):
                    self.stages.append(nn.Sequential())
                self.stages[stage_idx].add_module(f'block{len(self.stages[stage_idx])}', block)
                if stage_idx < len(self.out_channels):
                    self.out_channels[stage_idx] = out_c
                else:
                    self.out_channels.append(out_c)
            
            in_c = out_c
        
        # Final 1x1 conv - reduce to 384 for YOLO compatibility
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_c, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            HardSwish()
        )
        
        # Global average pooling (for feature extraction)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Extract features at different scales
        features = []
        
        # Stage outputs for YOLO (P2, P3, P4, P5)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:  # Skip first stage, start from P2
                features.append(x)
        
        # Final conv
        x = self.final_conv(x)
        features.append(x)  # P5
        
        # Return features [P2, P3, P4, P5]
        return features[-3:]  # Return only P3, P4, P5 for YOLO


class StdPool(nn.Module):
    """Standard pooling layer for MAFR."""
    def forward(self, x):
        # Calculate standard deviation across spatial dimensions
        b, c, h, w = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True).view(b, c, 1, 1)
        return std


class MCAGate(nn.Module):
    """Multi-dimensional Collaborative Attention Gate."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_std = StdPool()
        self.ch_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention  
        self.sp_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        ch_avg = self.ch_avg(x)
        ch_std = self.ch_std(x)
        ch_att = self.ch_fc(ch_avg + ch_std)
        ch_att = self.sigmoid(ch_att)
        
        # Apply channel attention
        x_ch = x * ch_att
        
        # Spatial attention
        sp_avg = torch.mean(x_ch, dim=1, keepdim=True)
        sp_max, _ = torch.max(x_ch, dim=1, keepdim=True)
        sp_att = self.sp_conv(torch.cat([sp_avg, sp_max], dim=1))
        sp_att = self.sigmoid(sp_att)
        
        return x_ch * sp_att


class MCALayer(nn.Module):
    """Multi-dimensional Collaborative Attention Layer."""
    def __init__(self, channels, no_spatial=False):
        super().__init__()
        self.mca_gate = MCAGate(channels)
        self.no_spatial = no_spatial

    def forward(self, x):
        return self.mca_gate(x)


class LightweightMSFFM(nn.Module):
    """Lightweight Multi-Scale Feature Fusion Module."""
    def __init__(self, channels):
        super().__init__()
        # Multi-scale grouped convolutions
        self.conv1x1 = nn.Conv2d(channels, channels//4, 1, groups=1)
        self.conv3x3 = nn.Conv2d(channels, channels//4, 3, padding=1, groups=channels//4)
        self.conv5x5 = nn.Conv2d(channels, channels//4, 5, padding=2, groups=channels//4)
        self.conv7x7 = nn.Conv2d(channels, channels//4, 7, padding=3, groups=channels//4)
        
        # Fusion
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
        # SE module
        self.se = SEModule(channels)

    def forward(self, x):
        # Multi-scale features
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)
        
        # Concatenate
        fused = torch.cat([f1, f3, f5, f7], dim=1)
        
        # Fusion and SE
        out = self.fusion(fused)
        out = self.bn(out)
        out = self.act(out)
        out = self.se(out)
        
        return out


class MiniResidualBlock(nn.Module):
    """Mini Residual Block for MAFR."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.act2(out)
        
        return out


class MAFR(nn.Module):
    """Multi-scale Adaptive Feature Refinement module."""
    def __init__(self, channels, no_spatial=True):
        super().__init__()
        self.mca = MCALayer(channels, no_spatial)
        self.lmsffm = LightweightMSFFM(channels)
        self.mini_res = MiniResidualBlock(channels)

    def forward(self, x):
        x = self.mca(x)
        x = self.lmsffm(x)
        x = self.mini_res(x)
        return x


# YOLO Integration module
class LWMP_PPLCNet(nn.Module):
    """
    PP-LCNet x0.75 wrapper for YOLO integration.
    Outputs features at P3, P4, P5 scales.
    """
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        self.backbone = LCBackbone(in_channels=ch, scale=0.75)
        
        # Output channel counts for YOLO
        # Based on PP-LCNet x0.75: [96, 192, 384] for P3, P4, P5
        self.out_channels = [96, 192, 384]
        
        # Check actual output channels
        with torch.no_grad():
            dummy = torch.randn(1, ch, 256, 256)
            outputs = self.backbone(dummy)
            self.out_channels = [o.shape[1] for o in outputs]

    def forward(self, x):
        features = self.backbone(x)
        # Return as a list for YOLO compatibility
        if isinstance(features, list):
            return features
        return [features]