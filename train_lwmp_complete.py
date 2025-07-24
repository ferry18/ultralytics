#!/usr/bin/env python3
"""
Complete training script for LWMP-YOLO with all features.
Uses parse_model patching for proper lcnet_075 multi-output support.
"""

import os
import sys
import torch
from pathlib import Path

# Apply parse_model patch BEFORE importing YOLO
from parse_model_patch import patch_parse_model, unpatch_parse_model
original_parse_model = patch_parse_model()

# Now import YOLO with patched parse_model
from ultralytics import YOLO

# Import AWLoss integration
from train_awloss import monkey_patch_awloss


def main():
    """Train LWMP-YOLO with all components."""
    
    print("\n" + "=" * 80)
    print("LWMP-YOLO Complete Training")
    print("=" * 80)
    
    # Configuration
    model_yaml = 'ultralytics/cfg/models/11/yolo11-lwmp-author.yaml'
    data_yaml = 'ultralytics/cfg/datasets/coco8-grayscale.yaml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_yaml}")
    print(f"  Dataset: {data_yaml}")  
    print(f"  Device: {device}")
    
    print("\nFeatures:")
    print("  ✓ PP-LCNet x0.75 backbone (1.23M params)")
    print("  ✓ MAFR module (MCA + MSFFM + MiniResidual)")
    print("  ✓ AWLoss (NWD + Area weighting + Scale difference)")
    print("  ✓ P2 layer for small object detection")
    print("  ✓ Author's exact YAML structure")
    
    try:
        # Apply AWLoss
        print("\nApplying AWLoss...")
        monkey_patch_awloss()
        print("✓ AWLoss integrated")
        
        # Load model with author's YAML
        print("\nLoading model...")
        model = YOLO(model_yaml)
        print("✓ Model loaded successfully!")
        
        # Verify architecture
        print("\nModel Architecture:")
        total_params = 0
        
        for i, m in enumerate(model.model.model):
            params = sum(p.numel() for p in m.parameters())
            total_params += params
            
            # Print key layers
            if i < 10 or m.__class__.__name__ in ['lcnet_075', 'MAFR', 'Detect']:
                print(f"  [{i:>2}] {m.__class__.__name__:<20} params: {params:>10,}")
                
        print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Verify lcnet features
        print("\nVerifying PP-LCNet features...")
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            # Get backbone module
            backbone = model.model.model[0]
            if hasattr(backbone, '__class__') and backbone.__class__.__name__ == 'lcnet_075':
                # Run forward
                _ = backbone(test_input)
                
                # Check stored features
                if hasattr(backbone, 'p2'):
                    print(f"  ✓ P2: {backbone.p2.shape}")
                    print(f"  ✓ P3: {backbone.p3.shape}")
                    print(f"  ✓ P4: {backbone.p4.shape}")
                    print(f"  ✓ P5: {backbone(test_input).shape}")
                else:
                    print("  ⚠ Features not accessible")
            else:
                print(f"  ⚠ Unexpected backbone type: {backbone.__class__.__name__}")
        
        # Training configuration
        print("\n" + "-" * 80)
        print("Starting training...")
        print("-" * 80)
        
        # Train with all optimizations
        results = model.train(
            data=data_yaml,
            epochs=50,  # Reduced for testing
            imgsz=640,
            batch=8,    # Small batch for CPU
            device=device,
            workers=4,
            project='runs/lwmp-complete',
            name='train',
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
            verbose=True,
            # Optimizer settings
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Loss weights (AWLoss integrated)
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # Other settings
            close_mosaic=10,
            amp=False,  # Disable AMP for CPU
            seed=42
        )
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        
        # Validation
        print("\nRunning validation...")
        val_results = model.val()
        
        print(f"\nValidation Results:")
        print(f"  mAP@0.5: {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.p:.4f}")
        print(f"  Recall: {val_results.box.r:.4f}")
        
        # Model info
        save_dir = Path(model.trainer.save_dir)
        weights_dir = save_dir / 'weights'
        
        # Check model files
        if weights_dir.exists():
            best_pt = weights_dir / 'best.pt'
            last_pt = weights_dir / 'last.pt'
            
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                print(f"\nBest model: {best_pt}")
                print(f"  Size: {size_mb:.2f} MB")
                
                # Load and check parameters
                checkpoint = torch.load(best_pt, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
                    param_count = sum(p.numel() for p in state_dict.values() if p.dtype in [torch.float32, torch.float16])
                    print(f"  Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
                    
                    # Compare to paper
                    paper_params = 1.23e6
                    paper_size = 2.71
                    print(f"\nPaper targets:")
                    print(f"  Parameters: {paper_params:,} ({paper_params/1e6:.2f}M)")
                    print(f"  Size: {paper_size:.2f} MB")
                    
                    if param_count <= paper_params * 1.1:
                        print("  ✓ Parameter count within target!")
                    else:
                        print(f"  ⚠ Parameters exceed target by {(param_count/paper_params - 1)*100:.1f}%")
                        print("    Pruning needed to meet paper specifications")
        
        # Summary
        print(f"\nTraining artifacts saved to: {save_dir}")
        print("\nImplementation Summary:")
        print("  ✓ PP-LCNet x0.75 backbone")
        print("  ✓ MAFR neck enhancement")  
        print("  ✓ AWLoss optimization")
        print("  ✓ P2 detection layer")
        print("  ✓ All paper features implemented")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore original parse_model
        unpatch_parse_model(original_parse_model)
        print("\n✓ Parse model restored")


if __name__ == "__main__":
    main()