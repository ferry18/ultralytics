"""
Patch for YOLO to support multi-output lcnet_075 module.
This allows the author's YAML to work correctly.
"""

import torch
from typing import List, Union


def patch_yolo_for_lcnet():
    """
    Patch YOLO's BaseModel to handle lcnet_075's multi-output features.
    This allows indices 1, 2, 3 to access P2, P3, P4 from lcnet_075.
    """
    # Delayed import to avoid circular dependency
    import ultralytics.nn.tasks
    BaseModel = ultralytics.nn.tasks.BaseModel
    
    # Store original _predict_once
    original_predict_once = BaseModel._predict_once
    
    # Special storage for lcnet features
    lcnet_features = {}
    
    def _predict_once_patched(self, x, profile=False, visualize=False, embed=None):
        """Modified predict_once that handles lcnet_075 multi-outputs."""
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # Special handling for accessing lcnet features
                if isinstance(m.f, list):
                    x = []
                    for j in m.f:
                        if j == -1:
                            x.append(y[-1] if y else None)
                        elif j in [1, 2, 3] and hasattr(self, '_lcnet_features'):
                            # Map to lcnet features: 1->P2, 2->P3, 3->P4
                            feature_map = {1: 'p2', 2: 'p3', 3: 'p4'}
                            x.append(self._lcnet_features.get(feature_map.get(j)))
                        elif j < len(y):
                            x.append(y[j])
                        else:
                            # Handle out of range indices gracefully
                            x.append(None)
                else:
                    if m.f in [1, 2, 3] and hasattr(self, '_lcnet_features'):
                        feature_map = {1: 'p2', 2: 'p3', 3: 'p4'}
                        x = self._lcnet_features.get(feature_map.get(m.f))
                    else:
                        x = y[m.f] if m.f < len(y) else None
            
            if profile:
                self._profile_one_layer(m, x, dt)
            
            # Run module
            x = m(x)
            
            # Special handling for lcnet_075
            if hasattr(m, '__class__') and m.__class__.__name__ == 'lcnet_075':
                # If lcnet_075 has stored features, make them available
                if hasattr(m, 'p2') and hasattr(m, 'p3') and hasattr(m, 'p4'):
                    # Store features at the expected indices
                    lcnet_features['p2'] = m.p2
                    lcnet_features['p3'] = m.p3  
                    lcnet_features['p4'] = m.p4
                    
                    # lcnet_075 is at index 0, but we need to insert at the right positions
                    # The author's YAML expects: index 1 = P2, index 2 = P3, index 3 = P4
                    # But with current indexing, we have: 0=lcnet, 1=SPPF, 2=C2PSA
                    # So we need to inject P2,P3,P4 features at positions 1,2,3
                    
                    # Store the features globally for later access
                    if not hasattr(self, '_lcnet_features'):
                        self._lcnet_features = {}
                    self._lcnet_features['p2'] = m.p2
                    self._lcnet_features['p3'] = m.p3
                    self._lcnet_features['p4'] = m.p4
                    
                    # Modify save list to include these indices
                    if hasattr(self, 'save'):
                        for idx in [1, 2, 3]:
                            if idx not in self.save:
                                self.save.append(idx)
                        self.save = sorted(self.save)
            
            # Save output normally
            y.append(x if m.i in self.save else None)
            
            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        
        return x
    
    # Apply the patch
    BaseModel._predict_once = _predict_once_patched
    
    print("✓ YOLO patched for lcnet_075 multi-output support")
    
    return original_predict_once


def unpatch_yolo(original_method):
    """Restore original YOLO behavior."""
    import ultralytics.nn.tasks
    ultralytics.nn.tasks.BaseModel._predict_once = original_method
    print("✓ YOLO patch removed")


# Export functions
__all__ = ['patch_yolo_for_lcnet', 'unpatch_yolo']