"""
Training script for LWMP-YOLO with PP-LCNet multi-output support.
This script patches YOLO to handle the author's multi-output lcnet_075 module.
"""

import sys
import torch
from pathlib import Path

# Apply patches before importing YOLO
print("Applying YOLO patches for multi-output support...")

# Import the patcher
from ultralytics.nn.modules import MultiOutputHandler

# Apply patches
original_forward = MultiOutputHandler.patch_yolo_forward()
original_parse = MultiOutputHandler.patch_parse_model()

print("✓ YOLO core patched successfully!")

# Now import YOLO with patches applied
from ultralytics import YOLO


def train_lwmp_yolo():
    """Train LWMP-YOLO with author's exact YAML configuration."""
    
    print("\n" + "=" * 80)
    print("LWMP-YOLO Training with Multi-Output PP-LCNet")
    print("=" * 80)
    
    # Model configuration
    model_yaml = 'ultralytics/cfg/models/11/yolo11-lwmp-author.yaml'
    data_yaml = 'ultralytics/cfg/datasets/coco8-grayscale.yaml'
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_yaml}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Load model with author's YAML
        print("\nLoading model with author's YAML structure...")
        model = YOLO(model_yaml)
        print("✓ Model loaded successfully!")
        
        # Print model structure
        print("\nModel structure:")
        for i, m in enumerate(model.model.model):
            print(f"  [{i:>2}] {m.__class__.__name__}", end='')
            if hasattr(m, 'multi_output') and m.multi_output:
                print(" [MULTI-OUTPUT]", end='')
            if hasattr(m, 'out_channels'):
                if isinstance(m.out_channels, list):
                    print(f" -> {m.out_channels}", end='')
                else:
                    print(f" -> {m.out_channels}ch", end='')
            print()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Check PP-LCNet backbone parameters
        backbone_params = 0
        for name, param in model.model.named_parameters():
            if 'model.0' in name:  # lcnet_075 is first module
                backbone_params += param.numel()
        
        print(f"PP-LCNet backbone: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        
        # Test forward pass
        print("\nTesting forward pass...")
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(test_input)
        print("✓ Forward pass successful!")
        
        # Train the model
        print("\n" + "-" * 80)
        print("Starting training...")
        print("-" * 80)
        
        results = model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project='runs/lwmp-patched',
            name='train',
            exist_ok=True,
            patience=50,
            save=True,
            plots=True,
            verbose=True
        )
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
        # Validate the model
        print("\nRunning validation...")
        val_results = model.val()
        
        print(f"\nValidation Results:")
        print(f"  mAP@0.5: {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        
        # Model info
        save_dir = Path(model.trainer.save_dir)
        print(f"\nModel saved to: {save_dir}")
        
        # Check final model size
        best_model = save_dir / 'weights' / 'best.pt'
        if best_model.exists():
            model_size_mb = best_model.stat().st_size / (1024 * 1024)
            print(f"Model size: {model_size_mb:.2f} MB")
            
            # Verify parameters match paper target
            checkpoint = torch.load(best_model, map_location='cpu')
            if 'model' in checkpoint:
                model_state = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
                param_count = sum(p.numel() for p in model_state.values() if p.dtype in [torch.float32, torch.float16])
                print(f"Final parameters: {param_count:,} ({param_count/1e6:.2f}M)")
                
                if param_count > 1.5e6:
                    print("\n⚠️  Warning: Parameter count exceeds paper's 1.23M target")
                    print("   Pruning will be needed to achieve the target size")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multi_output():
    """Test that multi-output indexing works correctly."""
    print("\n" + "=" * 80)
    print("Testing Multi-Output Feature Access")
    print("=" * 80)
    
    try:
        # Create a simple test
        from ultralytics.nn.modules import lcnet_075
        
        # Create module
        module = lcnet_075(ch=3)
        print("✓ Multi-output lcnet_075 created")
        
        # Test forward
        x = torch.randn(1, 3, 224, 224)
        output = module(x)
        
        if isinstance(output, list):
            print(f"✓ Returns list of {len(output)} features:")
            for i, feat in enumerate(output):
                print(f"  [{i}] shape: {feat.shape}")
        else:
            print(f"✗ Returns single tensor: {output.shape}")
            
        # Check multi_output flag
        if hasattr(module, 'multi_output'):
            print(f"✓ multi_output flag: {module.multi_output}")
        else:
            print("✗ No multi_output flag")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # First test multi-output functionality
    test_multi_output()
    
    # Then train the model
    train_lwmp_yolo()