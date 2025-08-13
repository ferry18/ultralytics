"""
Debug parse_model step by step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Mock the parse process
ch = [3]
f = -1  # from index
args = []  # empty args from yaml

print(f"Initial values:")
print(f"  ch = {ch}")
print(f"  f = {f}")
print(f"  ch[f] = {ch[f]}")
print(f"  type(ch[f]) = {type(ch[f])}")

# Test what parse_model would do
c1 = ch[f]
c2 = None
args_to_module = [c1, c2, *args]

print(f"\nArgs to lcnet_075:")
print(f"  args = {args_to_module}")
print(f"  c1 = {c1}, type = {type(c1)}")
print(f"  c2 = {c2}, type = {type(c2)}")

# Test direct module creation
print("\nTesting direct module creation:")
from ultralytics.nn.modules import lcnet_075

try:
    model = lcnet_075(c1, c2)
    print("✓ Module created successfully")
except Exception as e:
    print(f"✗ Module creation failed: {e}")
    import traceback
    traceback.print_exc()