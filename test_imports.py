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
        print("  Loading model...")
        model = YOLO(yaml_path)
        print("✓ Model loaded successfully")
        
        # Print model info
        print("\nModel Information:")
        print(f"  Number of parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"  Number of layers: {len(list(model.model.modules()))}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        import torch
        dummy_input = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"✓ Forward pass successful, output shapes: {[o.shape for o in output] if isinstance(output, (list, tuple)) else output.shape}")
        
    else:
        print(f"✗ YAML file not found: {yaml_path}")
except Exception as e:
    print(f"✗ Model loading/testing failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")