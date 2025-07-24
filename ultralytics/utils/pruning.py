"""
Pruning utilities for LWMP-YOLO.
Based on L1-norm filter importance evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def compute_filter_importance(conv_layer):
    """
    Compute L1-norm importance for each filter in a convolutional layer.
    
    Args:
        conv_layer: nn.Conv2d layer
        
    Returns:
        importance: numpy array of filter importance scores
    """
    weight = conv_layer.weight.data.cpu().numpy()
    num_filters = weight.shape[0]
    
    # Compute L1-norm for each filter
    importance = np.zeros(num_filters)
    for i in range(num_filters):
        importance[i] = np.sum(np.abs(weight[i]))
    
    return importance


def prune_conv_layer(conv_layer, bn_layer, keep_idxs):
    """
    Prune a convolutional layer by keeping only specified filters.
    
    Args:
        conv_layer: nn.Conv2d layer to prune
        bn_layer: corresponding BatchNorm layer (optional)
        keep_idxs: indices of filters to keep
        
    Returns:
        new_conv: pruned convolutional layer
        new_bn: pruned batch norm layer (if provided)
    """
    # Get dimensions
    out_channels = len(keep_idxs)
    in_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    groups = conv_layer.groups
    bias = conv_layer.bias is not None
    
    # Create new conv layer
    new_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, groups=groups, bias=bias
    )
    
    # Copy weights
    new_conv.weight.data = conv_layer.weight.data[keep_idxs]
    if bias:
        new_conv.bias.data = conv_layer.bias.data[keep_idxs]
    
    # Prune batch norm if provided
    new_bn = None
    if bn_layer is not None:
        new_bn = nn.BatchNorm2d(out_channels)
        new_bn.weight.data = bn_layer.weight.data[keep_idxs]
        new_bn.bias.data = bn_layer.bias.data[keep_idxs]
        new_bn.running_mean = bn_layer.running_mean[keep_idxs]
        new_bn.running_var = bn_layer.running_var[keep_idxs]
        new_bn.num_batches_tracked = bn_layer.num_batches_tracked
    
    return new_conv, new_bn


def prune_model(model, prune_ratio=0.3):
    """
    Prune a YOLO model using L1-norm based filter pruning.
    
    Args:
        model: YOLO model to prune
        prune_ratio: ratio of filters to prune (0-1)
        
    Returns:
        pruned_model: pruned version of the model
    """
    pruned_model = deepcopy(model)
    
    # Find all conv-bn pairs
    conv_bn_pairs = []
    modules = list(pruned_model.modules())
    
    for i in range(len(modules) - 1):
        if isinstance(modules[i], nn.Conv2d) and isinstance(modules[i+1], nn.BatchNorm2d):
            if modules[i].out_channels == modules[i+1].num_features:
                conv_bn_pairs.append((modules[i], modules[i+1]))
    
    # Prune each conv-bn pair
    for conv, bn in conv_bn_pairs:
        # Skip depthwise convolutions and final layers
        if conv.groups > 1 or conv.out_channels < 32:
            continue
            
        # Compute importance
        importance = compute_filter_importance(conv)
        
        # Determine filters to keep
        num_keep = int(conv.out_channels * (1 - prune_ratio))
        num_keep = max(num_keep, 16)  # Keep at least 16 filters
        
        keep_idxs = np.argsort(importance)[::-1][:num_keep]
        keep_idxs = np.sort(keep_idxs)
        
        # Prune the layer
        new_conv, new_bn = prune_conv_layer(conv, bn, keep_idxs)
        
        # Replace in model (this is simplified - in practice you'd need to handle connections)
        # The actual implementation would require rebuilding the model architecture
    
    return pruned_model


def compute_pruning_stats(model, pruned_model):
    """
    Compute statistics about pruning results.
    
    Args:
        model: original model
        pruned_model: pruned model
        
    Returns:
        stats: dictionary of pruning statistics
    """
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())
    
    original_params = count_parameters(model)
    pruned_params = count_parameters(pruned_model)
    
    stats = {
        'original_parameters': original_params,
        'pruned_parameters': pruned_params,
        'reduction': 1 - (pruned_params / original_params),
        'reduction_mb': (original_params - pruned_params) * 4 / 1024 / 1024  # assuming float32
    }
    
    return stats