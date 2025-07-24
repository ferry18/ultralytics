"""
PP-LCNet x0.75 with multi-output support for YOLO.
This implementation allows YOLO to access intermediate features through special indexing.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
from .lcnet_final import lcnet_075 as lcnet_075_base


class lcnet_075(nn.Module):
    """
    PP-LCNet x0.75 that returns multiple outputs for YOLO's indexing system.
    
    This module is designed to work with the author's YAML structure where:
    - The module itself is at index 0
    - Internal features P2, P3, P4 can be accessed at indices 1, 2, 3
    """
    
    def __init__(self, ch=3, pretrained=True):
        super().__init__()
        
        # Handle YOLO's parameter passing
        if isinstance(ch, list):
            ch = ch[0] if ch else 3
        if isinstance(pretrained, list):
            pretrained = pretrained[0] if pretrained else True
            
        # Create the base PP-LCNet backbone
        self.backbone = lcnet_075_base(ch=ch, pretrained=pretrained)
        
        # These will be populated during forward pass
        self._features = {}
        
        # Mark this module as multi-output for YOLO
        self.multi_output = True
        
    def forward(self, x):
        """
        Forward pass that returns P5 and makes P2, P3, P4 accessible.
        
        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: P5 feature or list of features
        """
        # Run backbone forward pass
        p5 = self.backbone(x)
        
        # Get stored features from backbone
        p2 = self.backbone.p2
        p3 = self.backbone.p3  
        p4 = self.backbone.p4
        
        # Store for external access
        self._features = {
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5
        }
        
        # Return as a list for YOLO to handle
        # This allows indexing: [0] = full list, [1] = p2, [2] = p3, [3] = p4
        return [p5, p2, p3, p4]


class MultiOutputHandler:
    """
    Helper class to handle multi-output modules in YOLO's architecture.
    This modifies how YOLO processes modules that return multiple features.
    """
    
    @staticmethod
    def patch_yolo_forward():
        """Patch YOLO's forward method to handle multi-output modules."""
        from ultralytics.nn.tasks import BaseModel
        
        # Store original method
        original_predict_once = BaseModel._predict_once
        
        def _predict_once_with_multi_output(self, x, profile=False, visualize=False, embed=None):
            """Modified predict_once that handles multi-output modules."""
            y, dt, embeddings = [], [], []  # outputs
            embed = frozenset(embed) if embed is not None else {-1}
            max_idx = max(embed)
            
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
                if profile:
                    self._profile_one_layer(m, x, dt)
                
                # Run module
                x = m(x)
                
                # Handle multi-output modules
                if hasattr(m, 'multi_output') and m.multi_output and isinstance(x, list):
                    # Store the main output (first element)
                    y.append(x[0] if m.i in self.save else None)
                    
                    # Store additional outputs at subsequent indices
                    for j, feat in enumerate(x[1:], start=1):
                        if len(y) <= m.i + j:
                            y.extend([None] * (m.i + j - len(y) + 1))
                        y[m.i + j] = feat
                else:
                    # Normal single output
                    y.append(x if m.i in self.save else None)
                
                if visualize:
                    feature_visualization(x[0] if isinstance(x, list) else x, m.type, m.i, save_dir=visualize)
                
                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(
                        x[0] if isinstance(x, list) else x, (1, 1)
                    ).squeeze(-1).squeeze(-1))  # flatten
                    if m.i == max_idx:
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
            
            return x[0] if isinstance(x, list) else x
        
        # Apply patch
        BaseModel._predict_once = _predict_once_with_multi_output
        
        return original_predict_once
    
    @staticmethod
    def patch_parse_model():
        """Patch parse_model to handle multi-output modules in YAML parsing."""
        from ultralytics.nn.tasks import parse_model
        import ultralytics.nn.tasks as tasks
        
        # Store original function
        original_parse_model = tasks.parse_model
        
        def parse_model_with_multi_output(d, ch, verbose=True):
            """Modified parse_model that pre-allocates space for multi-output modules."""
            import copy
            from ultralytics.utils import LOGGER
            import torch
            # Import specific modules needed
            from ultralytics.nn.modules import (Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, 
                                               SPP, SPPF, C2f, DWConv, Focus, BottleneckCSP, C1, C2, C3, C3TR, 
                                               C3Ghost, nn, DWConvTranspose2d, C3x, C2fPSA, C2PSA, RepC3, C2fCIB, 
                                               C3k2, ELAN, AConv, ADown, RepNCSPELAN4, SPPELAN, C2fAttn, SELayer, 
                                               LightweightMSFFM, MiniResidualBlock, MAFR, MCALayer, lcnet_075, 
                                               AIFI, HGStem, HGBlock, ResNetLayer, Concat, Detect, WorldDetect, 
                                               Segment, Pose, OBB, ImagePoolingAttn, v10Detect, YOLOEDetect, 
                                               YOLOESegment, RTDETRDecoder, CBLinear, CBFuse, TorchVision, Index, make_divisible)
            
            # Parse the YAML
            if verbose:
                LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
            
            nc, max_channels = d["nc"], d.get("max_channels", 1000)
            layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
            
            # Scan for multi-output modules first
            multi_output_indices = {}
            temp_d = copy.deepcopy(d)
            
            for i, (f, n, m, args) in enumerate(temp_d["backbone"] + temp_d["head"]):
                m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
                
                # Check if this is our multi-output lcnet_075
                if m.__name__ == 'lcnet_075' and hasattr(m, '__module__'):
                    if 'lcnet_multi_output' in m.__module__:
                        # This module will output 4 features
                        multi_output_indices[i] = 3  # 3 additional outputs
            
            # Now parse normally but adjust indices
            index_offset = 0
            
            for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
                # Adjust from indices based on offset
                if isinstance(f, list):
                    f = [x + index_offset if x >= 0 else x for x in f]
                elif f >= 0:
                    f = f + index_offset
                
                # Parse module
                m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
                for j, a in enumerate(args):
                    if isinstance(a, str):
                        args[j] = locals()[a] if a in locals() else a
                
                n = n_ = max(round(n * d["scales"].get("depth", 1.0)), 1) if n > 1 else n
                
                if m in {Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2f, DWConv,
                        Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d,
                        C3x, C2fPSA, C2PSA, RepC3, C2fCIB, C3k2, ELAN, AConv, ADown, RepNCSPELAN4, SPPELAN, C2fAttn,
                        SELayer, LightweightMSFFM, MiniResidualBlock, MAFR, MCALayer, lcnet_075}:
                    c1, c2 = ch[f], args[0]
                    if c2 != max_channels:
                        c2 = make_divisible(min(c2, max_channels) * d["scales"].get("width", 1.0), 8)
                    
                    args = [c1, c2, *args[1:]]
                    if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2PSA, 
                            C2fCIB, C3k2}:
                        args.insert(2, n)
                        n = 1
                elif m is AIFI:
                    args = [ch[f], *args]
                elif m in {HGStem, HGBlock}:
                    c1, cm, c2 = ch[f], args[0], args[1]
                    if c2 != max_channels:
                        c2 = make_divisible(min(c2, max_channels) * d["scales"].get("width", 1.0), 8)
                    
                    args = [c1, cm, c2, *args[2:]]
                    if m is HGBlock:
                        args.insert(4, n)
                        n = 1
                elif m is ResNetLayer:
                    c2 = args[1] if args[3] else args[1] * 4
                elif m is nn.BatchNorm2d:
                    args = [ch[f]]
                elif m is Concat:
                    c2 = sum(ch[x + index_offset if x >= 0 else x] for x in f if x != -1)
                elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
                    args.append([ch[x + index_offset if x >= 0 else x] for x in f if x != -1])
                    if m is Segment:
                        args[2] = make_divisible(min(args[2], max_channels) * d["scales"].get("width", 1.0), 8)
                elif m is RTDETRDecoder:
                    args.insert(1, [ch[x + index_offset if x >= 0 else x] for x in f if x != -1])
                elif m is CBLinear:
                    c2 = args[0]
                    c1 = ch[f[0] + index_offset if f[0] >= 0 else f[0]] if isinstance(f, list) else ch[f + index_offset if f >= 0 else f]
                    args = [c1, c2, *args[1:]]
                elif m is CBFuse:
                    c2 = ch[f[-1] + index_offset if f[-1] >= 0 else f[-1]]
                else:
                    c2 = ch[f + index_offset if f >= 0 else f]
                
                # Build module
                m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
                t = str(m)[8:-2].replace("__main__.", "")
                m_.np = sum(x.numel() for x in m_.parameters())
                m_.i, m_.f, m_.type = i + index_offset, f, t
                
                if verbose:
                    LOGGER.info(f"{i + index_offset:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
                
                # Update save list
                save.extend(x % (i + index_offset) for x in ([f] if isinstance(f, int) else f) if x != -1)
                
                layers.append(m_)
                if i == 0:
                    ch = []
                
                # Handle multi-output modules
                if i in multi_output_indices:
                    # This module outputs multiple features
                    num_extra = multi_output_indices[i]
                    ch.append(c2)  # Main output
                    
                    # Add placeholders for additional outputs
                    for j in range(num_extra):
                        ch.append(None)  # Will be filled during forward pass
                        save.append(i + index_offset + j + 1)
                    
                    # Update offset for subsequent modules
                    index_offset += num_extra
                else:
                    ch.append(c2)
            
            return torch.nn.Sequential(*layers), sorted(save)
        
        # Apply patch
        tasks.parse_model = parse_model_with_multi_output
        
        return original_parse_model


# Auto-patch when module is imported
def auto_patch():
    """Automatically patch YOLO when this module is imported."""
    try:
        MultiOutputHandler.patch_yolo_forward()
        MultiOutputHandler.patch_parse_model()
        print("✓ YOLO patched for multi-output support")
    except Exception as e:
        print(f"Warning: Failed to patch YOLO: {e}")


# Uncomment to auto-patch on import
# auto_patch()