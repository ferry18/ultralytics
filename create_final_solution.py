import yaml
import os
import shutil

def create_custom_selector_module():
    """Create a custom selector module that doesn't have the Index parsing issue."""
    
    selector_code = '''"""Custom selector module for LMWP-YOLO."""
import torch
import torch.nn as nn
from typing import List

class Selector(nn.Module):
    """Select a specific tensor from a list based on index. Custom module to avoid Ultralytics Index parsing issues."""
    
    def __init__(self, index=0):
        """Initialize selector with index."""
        super().__init__()
        self.index = index
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Select and return the tensor at the specified index."""
        return x[self.index]

# Aliases for different indices
class Select0(nn.Module):
    """Select first tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[0]

class Select1(nn.Module):
    """Select second tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[1]

class Select2(nn.Module):
    """Select third tensor from list."""
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[2]
'''
    
    # Write to custom modules
    with open('ultralytics/nn/modules/selector.py', 'w') as f:
        f.write(selector_code)
    
    print("Created: ultralytics/nn/modules/selector.py")
    
    # Update __init__.py
    init_path = 'ultralytics/nn/modules/__init__.py'
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Add import
    if 'from .selector import' not in content:
        import_line = 'from .selector import Select0, Select1, Select2, Selector'
        # Find the last import line
        lines = content.split('\n')
        import_idx = -1
        for i in range(len(lines)-1, -1, -1):
            if lines[i].startswith('from .'):
                import_idx = i
                break
        
        if import_idx >= 0:
            lines.insert(import_idx + 1, import_line)
        
        # Add to __all__
        all_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('__all__'):
                all_idx = i
                break
        
        if all_idx >= 0:
            # Find the closing bracket
            j = all_idx
            while j < len(lines) and ']' not in lines[j]:
                j += 1
            if j < len(lines):
                # Add before the closing bracket
                lines[j] = lines[j].replace(']', ', "Select0", "Select1", "Select2", "Selector"]')
        
        content = '\n'.join(lines)
        
        with open(init_path, 'w') as f:
            f.write(content)
        
        print("Updated: ultralytics/nn/modules/__init__.py")

def update_tasks_py():
    """Update tasks.py to include the new selector modules."""
    
    tasks_path = 'ultralytics/nn/tasks.py'
    with open(tasks_path, 'r') as f:
        content = f.read()
    
    # Add imports
    if 'Select0' not in content:
        # Find the imports section
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'from ultralytics.nn.modules import' in line and 'Select0' not in line:
                # Check if it ends with a parenthesis (multi-line import)
                if line.rstrip().endswith(')'):
                    lines[i] = line.rstrip()[:-1] + ', Select0, Select1, Select2, Selector)'
                else:
                    lines[i] = line.rstrip() + ', Select0, Select1, Select2, Selector'
                break
        
        # Also add to base_modules
        base_modules_idx = -1
        for i, line in enumerate(lines):
            if 'base_modules =' in line:
                base_modules_idx = i
                break
        
        if base_modules_idx >= 0:
            # Find the end of the tuple
            j = base_modules_idx
            while j < len(lines) and ')' not in lines[j]:
                j += 1
            if j < len(lines):
                # Add before the closing parenthesis
                lines[j] = lines[j].replace(')', ', Select0, Select1, Select2, Selector)')
        
        content = '\n'.join(lines)
        
        with open(tasks_path, 'w') as f:
            f.write(content)
        
        print("Updated: ultralytics/nn/tasks.py")

def create_working_yaml():
    """Create the final working YAML using custom selector modules."""
    
    yaml_content = """# Ultralytics YOLOv11 🚀, GPL-3.0 license
# LMWP-YOLO: Lightweight Mamba-assisted FPN with Receptive-field Attention Neck
# Final working version using custom selector modules

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants
  # [depth, width, max_channels]
  l: [1.00, 1.00, 512]

# Architecture
backbone:
  # [from, repeats, module, args]
  - [-1, 1, lcnet_075, [1024]]       # 0  outputs (P5, P4, P3) = (1024, 96, 48)
  - [0, 1, Select0, []]              # 1  P5 (1024 channels)
  - [0, 1, Select1, []]              # 2  P4 (96 channels)  
  - [0, 1, Select2, []]              # 3  P3 (48 channels)
  - [1, 1, SPPF, [512, 5]]           # 4  P5' (1024 -> 512 channels)
  - [4, 1, C2PSA, [512]]             # 5  refined P5' (512 channels)

head:
  - [5, 1, nn.Upsample, [None, 2, "nearest"]]    # 6  upsample P5' (512 channels)
  - [[6, 2], 1, Concat, [1]]                     # 7  cat(P5', P4) = 512 + 96 = 608 channels
  - [7, 1, C3k2, [256, False]]                   # 8  P4' (608 -> 256 channels)
  
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]    # 9  upsample P4' (256 channels)
  - [[9, 3], 1, Concat, [1]]                     # 10 cat(P4', P3) = 256 + 48 = 304 channels
  - [10, 1, C3k2, [128, False]]                  # 11 P3' (304 -> 128 channels)
  
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]   # 12 upsample P3' (128 channels)
  - [12, 1, C3k2, [64, False]]                   # 13 P2 (128 -> 64 channels)
  - [13, 1, MAFR, [64]]                          # 14 enhanced P2 (64 channels)
  
  - [[14, 11, 8], 1, Detect, [nc]]              # 15 Detect(P2, P3', P4')
"""
    
    # Save with 'l' in filename to force scale=l
    with open('yolov11l-lcnet-mafrneck-final.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created: yolov11l-lcnet-mafrneck-final.yaml")

def create_test_script():
    """Create a test script for the final solution."""
    
    test_code = '''import sys
sys.path.insert(0, '/workspace')
import torch
from ultralytics import YOLO

def test_final_model():
    """Test the final LMWP-YOLO model."""
    print("Testing Final LMWP-YOLO Model")
    print("="*60)
    
    config = 'yolov11l-lcnet-mafrneck-final.yaml'
    
    try:
        # Load model
        print(f"Loading: {config}")
        model = YOLO(config)
        print("✓ Model loaded successfully!")
        
        # Test forward pass
        print("\\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(dummy_input)
        print("✓ Forward pass successful!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"\\nTotal parameters: {total_params:,}")
        
        # Try minimal training
        print("\\nTesting training capability...")
        results = model.train(
            data='coco8.yaml',
            epochs=1,
            imgsz=640,
            batch=1,
            device='cpu',
            verbose=False,
            workers=0,
        )
        print("✓ Training test successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_final_model():
        print("\\n" + "="*60)
        print("✓ SUCCESS! LMWP-YOLO model is working!")
        print("\\nYou can now run the full training pipeline:")
        print("python3 train_lmwp_fixed.py")
        print("="*60)
    else:
        print("\\n✗ Model test failed")
'''
    
    with open('test_final_solution.py', 'w') as f:
        f.write(test_code)
    
    print("Created: test_final_solution.py")

def main():
    """Create the final working solution."""
    print("Creating Final LMWP-YOLO Solution")
    print("="*60)
    
    try:
        # Step 1: Create custom selector modules
        create_custom_selector_module()
        
        # Step 2: Update tasks.py
        update_tasks_py()
        
        # Step 3: Create working YAML
        create_working_yaml()
        
        # Step 4: Create test script
        create_test_script()
        
        print("\n✓ Final solution created!")
        print("\nTo test the model:")
        print("python3 test_final_solution.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()