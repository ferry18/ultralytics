import sys
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
    print("\nStarting training on COCO8 grayscale...")
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
    
    print("\n✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
