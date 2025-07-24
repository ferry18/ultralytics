"""
Patch for YOLO's parse_model to handle lcnet_075's special indexing requirements.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics.utils import LOGGER


def patch_parse_model():
    """Patch parse_model to handle lcnet_075's multi-output architecture."""
    import ultralytics.nn.tasks as tasks
    
    # Store original
    original_parse_model = tasks.parse_model
    
    def parse_model_patched(d, ch, verbose=True):
        """Modified parse_model that handles lcnet_075 special indexing."""
        # Import necessary modules
        import torch.nn as nn
        from ultralytics.utils.ops import make_divisible
        from ultralytics.nn.modules import (
            Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
            C2f, DWConv, Focus, BottleneckCSP, C1, C2, C3, C3TR, C3Ghost,
            DWConvTranspose2d, C3x, C2fPSA, C2PSA, RepC3, C2fCIB, C3k2,
            AConv, ADown, RepNCSPELAN4, SPPELAN, C2fAttn, AIFI, HGStem,
            HGBlock, ResNetLayer, Concat, Detect, WorldDetect, Segment, Pose,
            OBB, ImagePoolingAttn, v10Detect, RTDETRDecoder, CBLinear, CBFuse,
            lcnet_075, MAFR, SELayer
        )
        try:
            from ultralytics.nn.modules import ELAN
        except ImportError:
            ELAN = None
        try:
            from ultralytics.nn.modules import TorchVision
        except ImportError:
            TorchVision = None
        
        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        
        max_channels = d.get("max_channels", float("inf"))
        nc = d['nc']
        
        # First pass: build model and identify lcnet_075 location
        ch = [ch]  # Convert to list format as expected by parse_model
        layers, save, c2 = [], [], ch[-1]
        
        # Virtual indices for lcnet features
        lcnet_virtual_indices = {}
        
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m_str = m
            # Resolve module name to actual class
            if isinstance(m, str):
                if "nn." in m:
                    m = getattr(torch.nn, m[3:])
                else:
                    # Try to get from imported modules
                    m = locals().get(m) or globals().get(m) or eval(m)
            
            # Resolve string references
            for j, a in enumerate(args):
                if isinstance(a, str):
                    args[j] = locals()[a] if a in locals() else a
                    
            n = n_ = max(round(n * d.get("depth", 1.0)), 1) if n > 1 else n
            
            # Prepare module arguments
            # Build list of valid modules
            valid_modules = {Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
                    C2f, DWConv, Focus, BottleneckCSP, C1, C2, C3, C3TR, C3Ghost,
                    nn.ConvTranspose2d, DWConvTranspose2d, C3x, C2fPSA, C2PSA, RepC3,
                    C2fCIB, C3k2, AConv, ADown, RepNCSPELAN4, SPPELAN, C2fAttn,
                    SELayer, MAFR}
            if ELAN is not None:
                valid_modules.add(ELAN)
                
            if m in valid_modules:
                c1, c2 = ch[f], args[0]
                if c2 != max_channels:
                    c2 = make_divisible(min(c2, max_channels) * d.get("width", 1.0), 8)
                args = [c1, c2, *args[1:]]
                if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x,
                        RepC3, C2fPSA, C2PSA, C2fCIB, C3k2}:
                    args.insert(2, n)
                    n = 1
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in {HGStem, HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                if c2 != max_channels:
                    c2 = make_divisible(min(c2, max_channels) * d.get("width", 1.0), 8)
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
                args.append([ch[x] for x in f])
                if m is Segment:
                    args[2] = make_divisible(min(args[2], max_channels) * d.get("width", 1.0), 8)
            elif m is RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            elif m is lcnet_075:
                # Special handling for lcnet_075
                c2 = 288  # PP-LCNet x0.75 outputs 288 channels
                args = [3, *args]  # Ensure channel argument
                
                # Mark virtual indices for P2, P3, P4
                if i == 0:  # First module in backbone
                    lcnet_virtual_indices = {
                        1: (40, 'p2'),   # P2: 40 channels (corrected from lcnet_final.py)
                        2: (72, 'p3'),   # P3: 72 channels
                        3: (144, 'p4')   # P4: 144 channels
                    }
            else:
                c2 = ch[f]
                
            # Build module
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
                
            # Update save list
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            
            if i == 0:
                ch = []
                
            # Add channels
            ch.append(c2)
            
            # Add virtual channels for lcnet
            if m_str == 'lcnet_075' and lcnet_virtual_indices:
                for idx, (channels, name) in lcnet_virtual_indices.items():
                    if len(ch) <= idx:
                        ch.extend([None] * (idx - len(ch) + 1))
                    ch[idx] = channels
                    save.append(idx)
                    
        # Create a custom model class that handles lcnet feature access
        class LCNetModel(nn.Sequential):
            def __init__(self, layers):
                super().__init__(*layers)
                self.lcnet_features = {}
                
            def forward(self, x):
                y = []
                for m in self:
                    if m.f != -1:  # if not from previous layer
                        if isinstance(m.f, list):
                            x_list = []
                            for j in m.f:
                                if j == -1:
                                    x_list.append(x)
                                elif j in [1, 2, 3] and hasattr(self[0], 'p2'):
                                    # Access lcnet features
                                    feature_map = {1: self[0].p2, 2: self[0].p3, 3: self[0].p4}
                                    x_list.append(feature_map.get(j, y[j] if j < len(y) else None))
                                else:
                                    x_list.append(y[j] if j < len(y) else None)
                            x = x_list
                        else:
                            if m.f in [1, 2, 3] and hasattr(self[0], 'p2'):
                                feature_map = {1: self[0].p2, 2: self[0].p3, 3: self[0].p4}
                                x = feature_map.get(m.f, y[m.f] if m.f < len(y) else None)
                            else:
                                x = y[m.f] if m.f < len(y) else None
                    
                    x = m(x)  # run
                    y.append(x if m.i in save else None)  # save output
                    
                return x
                
        return LCNetModel(layers), sorted(save)
    
    # Apply patch
    tasks.parse_model = parse_model_patched
    return original_parse_model


def unpatch_parse_model(original):
    """Restore original parse_model."""
    import ultralytics.nn.tasks as tasks
    tasks.parse_model = original