import sys
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
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(dummy_input)
        print("✓ Forward pass successful!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
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
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_final_model():
        print("\n" + "="*60)
        print("✓ SUCCESS! LMWP-YOLO model is working!")
        print("\nYou can now run the full training pipeline:")
        print("python3 train_lmwp_fixed.py")
        print("="*60)
    else:
        print("\n✗ Model test failed")
