"""
Test script to verify LWMP-YOLO imports are working correctly.
"""

print("Testing imports...")

try:
    from ultralytics import YOLO
    print("✓ YOLO import successful")
except ImportError as e:
    print(f"✗ YOLO import failed: {e}")

try:
    from ultralytics.utils.loss import AWDetectionLoss, AWLoss
    print("✓ AWDetectionLoss and AWLoss import successful")
except ImportError as e:
    print(f"✗ AWDetectionLoss/AWLoss import failed: {e}")

try:
    from ultralytics.nn.modules import prune_model, L1FilterPruner
    print("✓ prune_model and L1FilterPruner import successful")
except ImportError as e:
    print(f"✗ prune_model/L1FilterPruner import failed: {e}")

try:
    from ultralytics.nn.modules import (
        HardSigmoid, HardSwish, SELayer, DepSepConv, 
        PPLCNet, lcnet_075, MAFR, MultiScaleFusion, 
        MicroResidualBlock, C3TR_LWMP
    )
    print("✓ All LWMP module imports successful")
except ImportError as e:
    print(f"✗ LWMP module imports failed: {e}")

print("\nTesting YAML configuration...")
try:
    import os
    yaml_path = 'ultralytics/cfg/models/11/yolo11-lwmp.yaml'
    if os.path.exists(yaml_path):
        print(f"✓ YAML file exists: {yaml_path}")
        # Try to load the model
        model = YOLO(yaml_path)
        print("✓ Model loaded successfully")
    else:
        print(f"✗ YAML file not found: {yaml_path}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

print("\nAll tests completed!")