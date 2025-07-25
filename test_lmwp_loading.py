import sys
sys.path.insert(0, '/workspace')
import os
import torch
from ultralytics import YOLO

def test_model_configs():
    """Test different LMWP-YOLO configurations."""
    print("Testing LMWP-YOLO Model Configurations")
    print("="*60)
    
    # List of configurations to test
    configs = [
        ('yolov11s-lcnet-mafrneck.yaml', 'YOLOv11s with LCNet-MAFR (scale=s)'),
        ('yolov11l-lcnet-mafrneck.yaml', 'YOLOv11l with LCNet-MAFR (scale=l)'),
        ('yolov11-lcnet-mafrneck-working.yaml', 'YOLOv11 with explicit scale=l'),
        ('LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml', 'Original LMWP-YOLO'),
    ]
    
    working_configs = []
    
    for config_path, description in configs:
        if os.path.exists(config_path):
            print(f"\n{description}")
            print(f"Config: {config_path}")
            print("-" * 40)
            
            try:
                # Try to load the model
                print("Loading model...")
                model = YOLO(config_path)
                print("✓ Model loaded successfully!")
                
                # Test forward pass with dummy input
                print("Testing forward pass...")
                dummy_input = torch.randn(1, 3, 640, 640)
                with torch.no_grad():
                    output = model.model(dummy_input)
                
                if isinstance(output, (list, tuple)):
                    print(f"✓ Forward pass successful! Output: {len(output)} tensors")
                    for i, o in enumerate(output):
                        print(f"  Output {i}: shape {o.shape}")
                else:
                    print(f"✓ Forward pass successful! Output shape: {output.shape}")
                
                # Count parameters
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                
                working_configs.append((config_path, description))
                
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"Error type: {type(e).__name__}")
                
                # Try to get more specific error info
                if "RuntimeError" in str(type(e)):
                    import traceback
                    traceback.print_exc()
        else:
            print(f"\n✗ Config not found: {config_path}")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"Working configurations: {len(working_configs)}/{len(configs)}")
    if working_configs:
        print("\nWorking configs:")
        for config, desc in working_configs:
            print(f"  - {config}: {desc}")
    else:
        print("\nNo working configurations found!")
    
    return working_configs

if __name__ == "__main__":
    working_configs = test_model_configs()
    
    if working_configs:
        print("\n" + "="*60)
        print("Testing training capability with first working config...")
        config_path, description = working_configs[0]
        
        print(f"\nUsing: {config_path}")
        model = YOLO(config_path)
        
        # Try a minimal training test
        print("\nAttempting minimal training test (1 epoch)...")
        try:
            results = model.train(
                data='coco8.yaml',
                epochs=1,
                imgsz=640,
                batch=1,
                device='cpu',
                verbose=False,
                save=False,
            )
            print("✓ Training test successful!")
        except Exception as e:
            print(f"✗ Training test failed: {e}")