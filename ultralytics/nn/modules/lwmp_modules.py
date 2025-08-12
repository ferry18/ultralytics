"""
LWMP-YOLO Modules for Ultralytics.

This module implements lightweight and efficient components for drone detection:
- PP-LCNet backbone with depthwise separable convolutions
- MAFR (Multidimensional Attention Feature Refinement) module
- C3TR module with transformer encoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .conv import Conv, DWConv, autopad
from .block import Bottleneck, C3


__all__ = ['HardSigmoid', 'HardSwish', 'SELayer', 'DepSepConv', 'PPLCNet', 'lcnet_075', 
           'MAFR', 'MultiScaleFusion', 'MicroResidualBlock', 'C3TR_LWMP', 'AWLoss']


# PP-LCNet Components
class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class HardSwish(nn.Module):
    """Hard Swish activation function as per Eq. 2: H-swish(x) = x * ReLU6(x+3) / 6."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


def _make_divisible(v, divisor=8, min_value=None):
    """Ensure channel number is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    """Depthwise Separable Convolution block for PP-LCNet."""
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super().__init__()
        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2

        layers = [
            # Depthwise convolution
            nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            HardSwish(),
        ]
        
        # SE module (optional)
        if use_se:
            layers.append(SELayer(inp, inp))
        
        # Pointwise convolution
        layers.extend([
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            HardSwish(),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class PPLCNet(nn.Module):
    """PP-LCNet backbone implementation for lightweight feature extraction."""
    def __init__(self, scale=0.75, in_channels=3):
        super().__init__()
        self.scale = scale
        
        # Network configuration: [kernel_size, out_channels, stride, use_SE]
        self.cfgs = [
            [3, 32, 1, 0],    # Stage 1
            [3, 64, 2, 0],
            [3, 64, 1, 0],
            [3, 128, 2, 0],   # Stage 2
            [3, 128, 1, 0],
            [5, 256, 2, 0],   # Stage 3
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 512, 2, 1],   # Stage 4 with SE
            [5, 512, 1, 1],
        ]
        
        # Initial stem
        input_channel = _make_divisible(16 * scale)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        features = []
        
        for i, (k, c, s, use_se) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * scale)
            features.append(DepSepConv(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
            
            # Create stage at stride transitions
            if s == 2 and i > 0 or i == len(self.cfgs) - 1:
                self.stages.append(nn.Sequential(*features))
                features = []
        
        # Final layers (not used in detection, but kept for completeness)
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, 1280, 1, 1, 0),
            HardSwish()
        )
        
        # Store channel info for YOLO integration
        self.out_channels = [
            _make_divisible(64 * scale),   # P2/4
            _make_divisible(128 * scale),  # P3/8
            _make_divisible(256 * scale),  # P4/16
            _make_divisible(512 * scale)   # P5/32
        ]

    def forward(self, x):
        outputs = []
        
        # Stem
        x = self.stem(x)  # P1/2
        
        # Stage 1 (P2/4)
        x = self.stages[0](x)
        outputs.append(x)  # P2/4
        
        # Stage 2 (P3/8)
        x = self.stages[1](x)
        outputs.append(x)  # P3/8
        
        # Stage 3 (P4/16)
        x = self.stages[2](x)
        outputs.append(x)  # P4/16
        
        # Stage 4 (P5/32)
        x = self.stages[3](x)
        outputs.append(x)  # P5/32
        
        return outputs


def lcnet_075(c1=3, pretrained=False):
    """
    Create PP-LCNet with 0.75x scale factor.
    
    Returns a backbone that outputs a list of features instead of a single tensor.
    This matches YOLO's multi-scale feature extraction pattern.
    """
    class LCNetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = PPLCNet(scale=0.75, in_channels=c1)
            
        def forward(self, x):
            # Get multi-scale features
            features = self.backbone(x)
            # Return the last feature (P5) for compatibility with YOLO head structure
            # The intermediate features can be accessed via indexing in the yaml
            return features[-1]  # Return P5/32
    
    return LCNetBackbone()


# MAFR Module Components
class MultidimensionalAttention(nn.Module):
    """Multidimensional Collaborative Attention mechanism."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Channel attention - processes concatenated avg and std
        self.channel_fc1 = nn.Conv2d(channels * 2, channels // 16, 1)
        self.channel_fc2 = nn.Conv2d(channels // 16, channels, 1)
        
        # Height attention
        self.height_conv = nn.Conv2d(channels, 1, kernel_size=1)
        
        # Width attention  
        self.width_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention (Eq. 3-5)
        # Eq. 3: Average pooling
        channel_avg = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        
        # Eq. 4: Standard deviation pooling
        channel_std = torch.std(x.view(b, c, -1), dim=2, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Concatenate avg and std as per paper
        channel_stats = torch.cat([channel_avg, channel_std], dim=1)  # [B, 2C, 1, 1]
        
        # Eq. 5: Generate channel attention through FC layers
        channel_attention = F.relu(self.channel_fc1(channel_stats))
        channel_attention = torch.sigmoid(self.channel_fc2(channel_attention))
        x_channel = x * channel_attention
        
        # Height attention (Eq. 6)
        x_h = x.permute(0, 3, 2, 1)  # [B, W, H, C]
        height_avg = torch.mean(x_h, dim=1, keepdim=True)  # [B, 1, H, C]
        height_attention = self.height_conv(height_avg.permute(0, 3, 2, 1))
        height_attention = torch.sigmoid(height_attention)
        x_height = x * height_attention
        
        # Width attention (Eq. 7)
        x_w = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        width_avg = torch.mean(x_w, dim=1, keepdim=True)  # [B, 1, W, C]
        width_attention = self.width_conv(width_avg.permute(0, 3, 2, 1))
        width_attention = torch.sigmoid(width_attention)
        x_width = x * width_attention
        
        # Fusion (Eq. 9)
        out = (x_channel + x_height + x_width) / 3
        
        return out


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion module with grouped convolutions."""
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        mid_channels = channels // 4
        
        # Multi-scale branches
        self.branch1 = nn.Conv2d(channels, mid_channels, 1, groups=groups)
        self.branch2 = nn.Conv2d(channels, mid_channels, 3, padding=1, groups=groups)
        self.branch3 = nn.Conv2d(channels, mid_channels, 5, padding=2, groups=groups)
        self.branch4 = nn.Conv2d(channels, mid_channels, 7, padding=3, groups=groups)
        
        # Fusion
        self.fusion = nn.Conv2d(mid_channels * 4, channels, 1)
        self.se = SELayer(channels, channels)

    def forward(self, x):
        # Multi-scale extraction
        x1 = self.branch1(x)
        x2 = self.branch2(x) 
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        # Concatenate and fuse
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_fused = self.fusion(x_cat)
        
        # SE attention
        x_att = self.se(x_fused)
        
        # Residual connection
        return x + x_att


class MicroResidualBlock(nn.Module):
    """Micro residual block for fine-grained feature extraction."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class MAFR(nn.Module):
    """Multidimensional Attention Feature Refinement module."""
    def __init__(self, channels):
        super().__init__()
        self.multidim_attention = MultidimensionalAttention(channels)
        self.multiscale_fusion = MultiScaleFusion(channels)
        self.micro_residual = MicroResidualBlock(channels)

    def forward(self, x):
        # Apply multidimensional attention
        x = self.multidim_attention(x)
        
        # Multi-scale fusion
        x = self.multiscale_fusion(x)
        
        # Micro residual refinement
        x = self.micro_residual(x)
        
        return x


# C3TR Module
class TransformerLayer_LWMP(nn.Module):
    """Transformer layer for C3TR module with self-attention."""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        # Self-attention with Q, K, V
        q = self.linear_q(src)
        k = self.linear_k(src)
        v = self.linear_v(src)
        
        src2, _ = self.self_attn(q, k, v)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class C3TR_LWMP(nn.Module):
    """C3TR module with Transformer encoder for global feature modeling."""
    def __init__(self, c1, c2, n=1, num_heads=8, dim_feedforward=2048, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        # Bottleneck blocks
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
        
        # Transformer components
        self.linear = nn.Linear(c2, c2)
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer_LWMP(c2, num_heads, dim_feedforward) for _ in range(n)]
        )
        
    def forward(self, x):
        # Split and process through CBS blocks
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        
        # Bottleneck processing
        y1 = self.m(y1)
        
        # Concatenate and process
        y = torch.cat((y1, y2), 1)
        y = self.cv3(y)
        
        # Transformer processing
        b, c, h, w = y.shape
        y_reshape = y.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # Linear shortcut
        y_linear = self.linear(y_reshape)
        
        # Transformer layers
        y_trans = self.transformer_layers(y_reshape)
        
        # Combine linear and transformer outputs
        y_out = y_trans + y_linear
        
        # Reshape back
        y_out = y_out.permute(1, 2, 0).reshape(b, c, h, w)
        
        return y_out


# Pruning Implementation
class L1FilterPruner:
    """
    L1-norm based filter pruning for lightweight model optimization.
    
    Implements Eq. 19: ||F_{i,j}||_1 = sum_{l=1}^{n_i} sum_{m=1}^{k} sum_{n=1}^{k} |K_{l,m,n}|
    """
    
    def __init__(self, model, pruning_ratio=0.3):
        """
        Initialize pruner with model and pruning ratio.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of filters to prune (0-1)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        
    def calculate_filter_importance(self, conv_layer):
        """
        Calculate L1-norm importance for each filter in a convolutional layer.
        
        Implements Eq. 19 from the paper.
        """
        weights = conv_layer.weight.data
        num_filters = weights.shape[0]
        
        # Calculate L1-norm for each filter
        importance_scores = torch.zeros(num_filters)
        for i in range(num_filters):
            # L1-norm: sum of absolute values
            importance_scores[i] = torch.sum(torch.abs(weights[i]))
            
        return importance_scores
    
    def get_pruning_plan(self):
        """
        Analyze model and create pruning plan for all conv layers.
        
        Returns:
            pruning_plan: Dict with layer names and filters to prune
        """
        pruning_plan = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 16:  # Don't prune too small layers
                importance = self.calculate_filter_importance(module)
                
                # Sort filters by importance
                sorted_idx = torch.argsort(importance)
                
                # Determine number of filters to prune
                num_filters = len(importance)
                num_to_prune = int(num_filters * self.pruning_ratio)
                
                # Keep minimum number of filters
                num_to_prune = min(num_to_prune, num_filters - 16)
                
                if num_to_prune > 0:
                    # Select filters with lowest importance
                    filters_to_prune = sorted_idx[:num_to_prune].tolist()
                    pruning_plan[name] = {
                        'filters_to_prune': filters_to_prune,
                        'original_out_channels': module.out_channels,
                        'new_out_channels': module.out_channels - num_to_prune
                    }
                    
        return pruning_plan
    
    def apply_pruning(self, pruning_plan):
        """
        Apply pruning to model based on pruning plan.
        
        Creates new model with reduced filters.
        """
        # Track conv layer connections for adjusting subsequent layers
        conv_connections = self._trace_conv_connections()
        
        for layer_name, prune_info in pruning_plan.items():
            filters_to_keep = [i for i in range(prune_info['original_out_channels']) 
                             if i not in prune_info['filters_to_prune']]
            
            # Get the module to prune
            module = dict(self.model.named_modules())[layer_name]
            
            # Create new pruned conv layer
            new_conv = nn.Conv2d(
                module.in_channels,
                prune_info['new_out_channels'],
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None
            )
            
            # Copy weights for kept filters
            new_conv.weight.data = module.weight.data[filters_to_keep]
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[filters_to_keep]
            
            # Replace module in model
            self._replace_module(layer_name, new_conv)
            
            # Adjust BatchNorm if exists
            bn_name = self._find_subsequent_bn(layer_name)
            if bn_name:
                self._prune_batchnorm(bn_name, filters_to_keep)
            
            # Adjust subsequent conv layers that use this layer's output
            self._adjust_subsequent_convs(layer_name, filters_to_keep, conv_connections)
    
    def _trace_conv_connections(self):
        """Trace connections between conv layers in the model."""
        connections = {}
        prev_conv_name = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if prev_conv_name:
                    connections[name] = prev_conv_name
                prev_conv_name = name
                
        return connections
    
    def _replace_module(self, module_name, new_module):
        """Replace a module in the model."""
        names = module_name.split('.')
        parent = self.model
        
        for name in names[:-1]:
            parent = getattr(parent, name)
            
        setattr(parent, names[-1], new_module)
    
    def _find_subsequent_bn(self, conv_name):
        """Find BatchNorm layer that follows a conv layer."""
        found_conv = False
        
        for name, module in self.model.named_modules():
            if name == conv_name:
                found_conv = True
            elif found_conv and isinstance(module, nn.BatchNorm2d):
                return name
            elif found_conv and isinstance(module, nn.Conv2d):
                break
                
        return None
    
    def _prune_batchnorm(self, bn_name, keep_indices):
        """Prune BatchNorm layer to match pruned conv layer."""
        bn_module = dict(self.model.named_modules())[bn_name]
        
        new_bn = nn.BatchNorm2d(len(keep_indices))
        new_bn.weight.data = bn_module.weight.data[keep_indices]
        new_bn.bias.data = bn_module.bias.data[keep_indices]
        new_bn.running_mean.data = bn_module.running_mean.data[keep_indices]
        new_bn.running_var.data = bn_module.running_var.data[keep_indices]
        
        self._replace_module(bn_name, new_bn)
    
    def _adjust_subsequent_convs(self, pruned_conv_name, keep_indices, connections):
        """Adjust input channels of conv layers that depend on pruned layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in connections:
                if connections[name] == pruned_conv_name:
                    # This conv uses the pruned conv's output
                    new_conv = nn.Conv2d(
                        len(keep_indices),
                        module.out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        module.bias is not None
                    )
                    
                    # Adjust weights for new input channels
                    new_conv.weight.data = module.weight.data[:, keep_indices]
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data
                    
                    self._replace_module(name, new_conv)


def prune_model(model, pruning_ratio=0.3):
    """
    Apply L1-norm based pruning to a model.
    
    Args:
        model: Model to prune
        pruning_ratio: Fraction of filters to remove
        
    Returns:
        pruned_model: Model with pruned filters
    """
    pruner = L1FilterPruner(model, pruning_ratio)
    
    # Get pruning plan
    pruning_plan = pruner.get_pruning_plan()
    
    # Log pruning statistics
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # Apply pruning
    pruner.apply_pruning(pruning_plan)
    
    total_params_after = sum(p.numel() for p in model.parameters())
    
    print(f"Pruning completed:")
    print(f"Parameters before: {total_params_before:,}")
    print(f"Parameters after: {total_params_after:,}")
    print(f"Reduction: {(1 - total_params_after/total_params_before)*100:.1f}%")
    
    return model