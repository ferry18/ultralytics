import sys
sys.path.insert(0, '/workspace')
import torch
from ultralytics import YOLO

def test_model():
    """Test the fixed LMWP-YOLO model."""
    print("Testing Fixed LMWP-YOLO Model V2")
    print("="*60)
    
    configs = [
        'yolov11l-lcnet-mafrneck-fixed-v2.yaml',
        'yolov11-lcnet-mafrneck-fixed-v2.yaml'
    ]
    
    for config in configs:
        print(f"\nTesting: {config}")
        print("-"*40)
        
        try:
            # Load model
            model = YOLO(config)
            print("✓ Model loaded successfully!")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                output = model.model(dummy_input)
            
            print("✓ Forward pass successful!")
            
            # Try minimal training
            print("\nTesting training capability...")
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
            
            return config
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return None

if __name__ == "__main__":
    working_config = test_model()
    
    if working_config:
        print(f"\n✓ SUCCESS! Working configuration: {working_config}")
        print("\nNow you can run the full training with:")
        print(f"python3 train_lmwp_fixed.py")
    else:
        print("\n✗ No working configuration found")