# YOLO Core Modifications for LWMP-YOLO

## Overview

To support the author's YAML structure for LWMP-YOLO, we've implemented modifications to YOLO's core that allow the `lcnet_075` module to expose multi-scale features (P2, P3, P4) that can be accessed by subsequent layers.

## The Challenge

The author's YAML expects:
```yaml
backbone:
  - [-1, 1, lcnet_075, [True]]           # 0 - PP-LCNet backbone
  - [-1, 1, SPPF, [1024, 5]]             # 1
  - [-1, 2, C2PSA, [1024]]               # 2

head:
  # ...
  - [[-1, 1], 1, Concat, [1]]   # Concat with index 1 (expects P2)
  - [[-1, 2], 1, Concat, [1]]   # Concat with index 2 (expects P3)  
  - [[-1, 3], 1, Concat, [1]]   # Concat with index 3 (expects P4)
```

Standard YOLO only has 3 backbone layers (indices 0, 1, 2), so accessing index 3 causes an error.

## Solution: Multi-Output Module Support

### 1. Modified `lcnet_075` Module
The `lcnet_075` module stores intermediate features:
```python
class lcnet_075(nn.Module):
    def forward(self, x):
        # ... backbone computation ...
        
        # Store features for later access
        self.p2 = features['p2']  # P2/4
        self.p3 = features['p3']  # P3/8  
        self.p4 = features['p4']  # P4/16
        
        return p5  # P5/32
```

### 2. YOLO Core Patch
The `patch_yolo_for_lcnet()` function modifies YOLO's `_predict_once` method:

```python
def _predict_once_patched(self, x, ...):
    # ... normal processing ...
    
    # Special handling for lcnet_075
    if m.__class__.__name__ == 'lcnet_075':
        if hasattr(m, 'p2') and hasattr(m, 'p3') and hasattr(m, 'p4'):
            # Make P2, P3, P4 available at indices 1, 2, 3
            if len(y) == 1:  # Right after lcnet_075
                y.extend([m.p2, m.p3, m.p4])
                # Update save list
                for idx in [1, 2, 3]:
                    if idx not in self.save:
                        self.save.append(idx)
```

This allows the Concat operations in the head to access:
- Index 1 → P2 features
- Index 2 → P3 features
- Index 3 → P4 features

### 3. Implementation Details

**Key Changes:**
1. The patch intercepts YOLO's forward pass
2. When `lcnet_075` is detected, it extracts stored features
3. These features are added to YOLO's output list at the expected indices
4. The save list is updated dynamically

**Benefits:**
- ✓ No modification to YOLO's source code required
- ✓ Author's YAML works without changes
- ✓ Clean and reversible (can unpatch)
- ✓ Maintains compatibility with standard YOLO

**Limitations:**
- Requires patching before model creation
- Specific to `lcnet_075` module name
- Assumes features are stored as attributes

## Usage

```python
# 1. Apply patch
from tasks_patch import patch_yolo_for_lcnet
original = patch_yolo_for_lcnet()

# 2. Use YOLO normally
from ultralytics import YOLO
model = YOLO('yolo11-lwmp-author.yaml')
model.train(...)

# 3. Restore original behavior
from tasks_patch import unpatch_yolo
unpatch_yolo(original)
```

## Alternative Approaches Considered

1. **Modify parse_model**: More complex, requires understanding YOLO's parsing
2. **Custom YAML format**: Would deviate from author's structure
3. **Multi-output list return**: Conflicts with YOLO's architecture
4. **Split modules**: Too verbose, loses encapsulation

## Conclusion

This solution provides the minimal necessary changes to support the author's multi-output backbone design while maintaining YOLO's architecture integrity. It demonstrates how framework limitations can be overcome with targeted patches when modifying core code isn't feasible.