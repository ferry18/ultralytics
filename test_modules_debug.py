"""
Debug script to test LWMP modules individually.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing individual LWMP modules...")

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from ultralytics.nn.modules import lcnet_075, MAFR, C3TR_LWMP
    print("✓ Module imports successful")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    exit(1)

# Test 2: Create lcnet_075
print("\n2. Testing lcnet_075 creation...")
try:
    # Test with proper arguments
    model = lcnet_075(c1=3, c2=None)
    print(f"✓ lcnet_075 created successfully")
    print(f"  Output channels: {model.out_channels}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
except Exception as e:
    print(f"✗ lcnet_075 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Create MAFR
print("\n3. Testing MAFR creation...")
try:
    mafr = MAFR(c1=256, c2=256)
    print("✓ MAFR created successfully")
    
    # Test forward pass
    x = torch.randn(1, 256, 40, 40)
    y = mafr(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
except Exception as e:
    print(f"✗ MAFR failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Create C3TR_LWMP
print("\n4. Testing C3TR_LWMP creation...")
try:
    c3tr = C3TR_LWMP(c1=256, c2=256)
    print("✓ C3TR_LWMP created successfully")
    
    # Test forward pass
    x = torch.randn(1, 256, 40, 40)
    y = c3tr(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
except Exception as e:
    print(f"✗ C3TR_LWMP failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test parse_model compatibility
print("\n5. Testing YOLO parse_model compatibility...")
try:
    from ultralytics.nn.tasks import parse_model
    from copy import deepcopy
    
    # Simple test config
    test_yaml = {
        'nc': 80,
        'scale': 'n',
        'scales': {'n': [0.33, 0.25, 1024]},
        'backbone': [
            [-1, 1, 'lcnet_075', []],
        ],
        'head': [
            [-1, 1, 'MAFR', []],
        ]
    }
    
    ch = 3  # Input channels (integer, not list!)
    model, save = parse_model(deepcopy(test_yaml), ch=ch, verbose=True)
    print("✓ parse_model successful")
    
except Exception as e:
    print(f"✗ parse_model failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug tests completed!")