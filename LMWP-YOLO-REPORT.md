# LMWP-YOLO Network Architecture Issues - Comprehensive Report

## Executive Summary

The LMWP-YOLO model implementation has fundamental channel mismatch issues due to the interaction between:
1. Width scaling applied by the Ultralytics framework
2. Multi-output backbone (lcnet_075) 
3. Index module channel tracking
4. Model scale inference from filename

## Issues Identified

### 1. Width Scaling Applied to All Layers
- The Ultralytics framework applies width_multiple to ALL layers, including backbone outputs
- With default scale 'n' (width=0.25), a 1024-channel output becomes 256 channels
- This causes mismatches throughout the network

### 2. Index Module Channel Tracking
```python
# In parse_model:
elif m in frozenset({TorchVision, Index}):
    c2 = args[0]  # Takes first argument as output channels
    c1 = ch[f]
    args = [*args[1:]]
```
- The Index module uses args[0] as output channels
- Original YAML: `Index, [256, 0]` means 256 output channels, index 0
- But lcnet actually outputs 1024 channels at index 0

### 3. Model Scale Inference
```python
def guess_model_scale(model_path):
    return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
```
- Scale is inferred from filename, not YAML content
- `yolov11-lcnet-mafrneck.yaml` → defaults to scale 'n'
- `yolov11l-lcnet-mafrneck.yaml` → uses scale 'l'

### 4. Multi-Output Backbone Handling
- lcnet_075 returns tuple (P5, P4, P3) with channels (1024, 96, 48)
- But width scaling reduces these to (256, 24, 12) with scale 'n'
- Index modules then specify different channels, causing mismatches

## Root Cause Analysis

The architecture assumes fixed channel counts from the backbone, but the framework applies global width scaling. The Index module's channel specification conflicts with actual backbone outputs after scaling.

## Attempted Solutions

1. **Fixed backbone (lcnet_075_fixed)**: Created version with hardcoded output channels
   - Still affected by width scaling in subsequent layers

2. **Scale 'l' usage**: Tried using scale='l' with width=1.0
   - Partially works but max_channels=512 causes other issues
   - Filename must contain 'l' to trigger this scale

3. **Modified YAML channel specs**: Adjusted Index module arguments
   - Complex to maintain consistency across all layers

## Recommended Solution

### Option 1: Proper Scale Configuration
1. Rename YAML to include scale: `yolov11s-lcnet-mafrneck.yaml`
2. Use scale 's' (width=0.5) or 'm' (width=0.75) for better balance
3. Adjust all channel specifications accordingly

### Option 2: Custom parse_model
Create a custom parse_model function that:
- Recognizes multi-output backbones
- Doesn't apply width scaling to backbone outputs
- Properly tracks channels through Index modules

### Option 3: Backbone Wrapper
Create a wrapper module that:
- Internally uses lcnet_075
- Applies inverse width scaling to restore original channels
- Presents consistent interface to the framework

## Implementation Recommendations

For immediate use:
```bash
# Rename the file to use scale 's' (width=0.5)
cp LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml yolov11s-lcnet-mafrneck.yaml

# Modify channel specifications in YAML:
# - Index modules: [512, 0], [48, 1], [24, 2]  # Half of original
# - Subsequent layers: adjust accordingly
```

For production:
- Implement custom backbone module that handles width scaling internally
- Submit PR to Ultralytics to better support multi-output backbones
- Document channel flow explicitly in configuration

## Conclusion

The LMWP-YOLO architecture is sound, but the implementation conflicts with Ultralytics' width scaling mechanism. The issue is not with the architecture itself but with how the framework handles multi-output backbones and channel tracking through Index modules.

The simplest immediate solution is to use an appropriate scale (s/m/l) that minimizes channel mismatches, or implement a custom backbone wrapper that manages scaling internally.