import yaml

def create_working_yaml():
    """Create a working YAML with proper channel specifications for scale 'l'."""
    
    # For scale 'l', width_multiple = 1.0
    # So we need to ensure all channel specs match the actual unscaled values
    
    yaml_content = """# Ultralytics YOLOv11 🚀, GPL-3.0 license
# LMWP-YOLO: Lightweight Mamba-assisted FPN with Receptive-field Attention Neck
# Working version with correct unscaled channel specifications

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
  - [1, 1, SPPF, [1024, 5]]          # 4  P5' (keeps 1024 channels)
  - [4, 1, C2PSA, [1024]]            # 5  refined P5' (1024 channels)

head:
  - [5, 1, nn.Upsample, [None, 2, "nearest"]]    # 6  upsample P5' (1024 channels)
  - [[6, 2], 1, Concat, [1]]                     # 7  cat(P5', P4) = 1024 + 96 = 1120 channels
  - [7, 1, C3k2, [512, False]]                   # 8  P4' (1120 -> 512 channels)
  
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]    # 9  upsample P4' (512 channels)
  - [[9, 3], 1, Concat, [1]]                     # 10 cat(P4', P3) = 512 + 48 = 560 channels
  - [10, 1, C3k2, [256, False]]                  # 11 P3' (560 -> 256 channels)
  
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]   # 12 upsample P3' (256 channels)
  - [12, 1, C3k2, [128, False]]                  # 13 P2 (256 -> 128 channels)
  - [13, 1, MAFR, [128]]                         # 14 enhanced P2 (128 channels)
  
  - [[14, 11, 8], 1, Detect, [nc]]              # 15 Detect(P2, P3', P4')
"""
    
    # Save with 'l' in filename to force scale=l
    with open('yolov11l-lcnet-mafrneck-working.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created: yolov11l-lcnet-mafrneck-working.yaml")
    
    # Also create a test script
    test_code = '''import sys
sys.path.insert(0, '/workspace')
import torch
from ultralytics import YOLO

print("Testing LMWP-YOLO Working Model")
print("="*60)

try:
    # Load model
    model = YOLO('yolov11l-lcnet-mafrneck-working.yaml')
    print("✓ Model loaded successfully!")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(dummy_input)
    print("✓ Forward pass successful!")
    
    # Start training
    print("\\nStarting training on COCO8 grayscale...")
    results = model.train(
        data='coco8.yaml',
        epochs=30,
        imgsz=640,
        batch=4,
        device='cpu',
        workers=0,
        project='runs/lmwp-train',
        name='grayscale',
        exist_ok=True,
    )
    
    print("\\n✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('train_lmwp_working.py', 'w') as f:
        f.write(test_code)
    
    print("Created: train_lmwp_working.py")

if __name__ == "__main__":
    create_working_yaml()