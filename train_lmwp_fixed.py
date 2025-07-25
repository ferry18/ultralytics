import sys
sys.path.insert(0, '/workspace')
import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.utils.prune as prune

def prepare_grayscale_coco8():
    """Download COCO8 and convert to grayscale."""
    print("Preparing grayscale COCO8 dataset...")
    
    # First, trigger COCO8 download using a standard model
    print("Downloading COCO8 dataset...")
    temp_model = YOLO('yolov8n.yaml')
    
    # Try to download by attempting a quick train
    try:
        temp_model.train(data='coco8.yaml', epochs=1, imgsz=640, device='cpu', batch=1)
    except Exception as e:
        print(f"Download triggered: {e}")
    
    # Find the downloaded dataset
    coco8_paths = [
        Path.home() / 'datasets/coco8',
        Path('/root/datasets/coco8'),
        Path('datasets/coco8'),
    ]
    
    coco8_path = None
    for path in coco8_paths:
        if path.exists():
            coco8_path = path
            break
    
    if not coco8_path:
        print("COCO8 not found in expected locations. Using standard coco8.yaml")
        return 'coco8.yaml'
    
    print(f"Found COCO8 dataset at: {coco8_path}")
    
    # Create grayscale version
    grayscale_path = Path('datasets/coco8-grayscale')
    
    # Copy structure
    if grayscale_path.exists():
        shutil.rmtree(grayscale_path)
    shutil.copytree(coco8_path, grayscale_path)
    
    # Convert all images to grayscale
    for split in ['train', 'val']:
        img_dir = grayscale_path / 'images' / split
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist, skipping...")
            continue
            
        print(f"Converting {split} images to grayscale...")
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        for img_path in tqdm(img_files):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Convert back to 3 channels for YOLO compatibility
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Save
                cv2.imwrite(str(img_path), gray_3ch)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Copy the original yaml and update paths
    original_yaml = coco8_path / 'coco8.yaml'
    if original_yaml.exists():
        with open(original_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update path
        data['path'] = str(grayscale_path.absolute())
        
        # Save new yaml
        grayscale_yaml = Path('datasets/coco8-grayscale.yaml')
        with open(grayscale_yaml, 'w') as f:
            yaml.dump(data, f)
    else:
        # Create yaml from scratch
        grayscale_yaml = Path('datasets/coco8-grayscale.yaml')
        yaml_content = f"""# Grayscale COCO8 dataset
path: {grayscale_path.absolute()}
train: images/train
val: images/val
test: images/val  # Use val as test

# Classes (COCO8 subset)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
"""
        with open(grayscale_yaml, 'w') as f:
            f.write(yaml_content)
    
    print(f"Grayscale COCO8 dataset prepared at: {grayscale_path}")
    return str(grayscale_yaml)

def test_model_loading():
    """Test if the fixed LMWP-YOLO model loads without errors."""
    print("\n" + "="*60)
    print("Testing LMWP-YOLO Model Loading")
    print("="*60)
    
    # Try different model configurations we created
    model_configs = [
        'yolov11s-lcnet-mafrneck.yaml',
        'yolov11l-lcnet-mafrneck.yaml',
        'yolov11-lcnet-mafrneck-working.yaml',
        'LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml'
    ]
    
    working_config = None
    for config in model_configs:
        if os.path.exists(config):
            print(f"\nTesting configuration: {config}")
            try:
                model = YOLO(config)
                print(f"✓ Successfully loaded: {config}")
                working_config = config
                
                # Test forward pass
                print("Testing forward pass...")
                dummy_input = torch.randn(1, 3, 640, 640)
                with torch.no_grad():
                    output = model.model(dummy_input)
                print(f"✓ Forward pass successful! Output shapes: {[o.shape for o in output] if isinstance(output, (list, tuple)) else output.shape}")
                break
                
            except Exception as e:
                print(f"✗ Failed to load {config}: {e}")
    
    return working_config

def train_lmwp_yolo(model_config, data_yaml, epochs=30):
    """Train the LMWP-YOLO model."""
    print("\n" + "="*60)
    print("Training LMWP-YOLO Model")
    print("="*60)
    
    # Initialize model
    print(f"Loading model from: {model_config}")
    model = YOLO(model_config)
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=4,  # Small batch size for CPU
        device='cpu',
        project='runs/lmwp-train',
        name='initial',
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        workers=0,
        amp=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
        close_mosaic=10,
    )
    
    # Return best model path
    best_model = Path('runs/lmwp-train/initial/weights/best.pt')
    if not best_model.exists():
        best_model = Path('runs/lmwp-train/initial/weights/last.pt')
    
    print(f"\nTraining completed. Best model saved at: {best_model}")
    return str(best_model)

def prune_lmwp_model(model_path, prune_ratio=0.1):
    """Apply structured pruning to the model."""
    print("\n" + "="*60)
    print(f"Pruning Model (ratio: {prune_ratio})")
    print("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Count prunable layers
    prunable_layers = []
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels > 16:  # Don't prune very small layers
            prunable_layers.append((name, module))
    
    print(f"Found {len(prunable_layers)} prunable Conv2d layers")
    
    # Apply structured pruning
    for name, module in prunable_layers:
        # Apply L2 norm structured pruning on output channels
        prune.ln_structured(
            module,
            name='weight',
            amount=prune_ratio,
            n=2,
            dim=0  # Prune output channels
        )
    
    # Make pruning permanent
    for name, module in prunable_layers:
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
    
    # Save pruned model
    pruned_path = model_path.replace('.pt', '_pruned.pt')
    torch.save(model.model.state_dict(), pruned_path)
    print(f"Pruned model saved at: {pruned_path}")
    
    return pruned_path

def finetune_lmwp_model(model_path, data_yaml, epochs=30):
    """Finetune the pruned model."""
    print("\n" + "="*60)
    print("Finetuning Pruned Model")
    print("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Finetune with lower learning rate
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=4,
        device='cpu',
        project='runs/lmwp-finetune',
        name='pruned',
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        workers=0,
        amp=False,
        lr0=0.001,  # Lower learning rate
        lrf=0.001,
        resume=True,  # Resume from weights
    )
    
    # Return best finetuned model
    best_model = Path('runs/lmwp-finetune/pruned/weights/best.pt')
    if not best_model.exists():
        best_model = Path('runs/lmwp-finetune/pruned/weights/last.pt')
    
    print(f"\nFinetuning completed. Best model saved at: {best_model}")
    return str(best_model)

def main():
    """Main training pipeline for LMWP-YOLO."""
    print("LMWP-YOLO Training Pipeline with Pruning")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Step 1: Test model loading
        working_config = test_model_loading()
        if not working_config:
            print("\nERROR: No working LMWP-YOLO configuration found!")
            return
        
        # Step 2: Prepare grayscale dataset
        data_yaml = prepare_grayscale_coco8()
        
        # Step 3: Train initial model
        print("\n" + "="*60)
        print("PHASE 1: Initial Training (30 epochs)")
        print("="*60)
        trained_model = train_lmwp_yolo(working_config, data_yaml, epochs=30)
        
        # Step 4: Prune the model
        print("\n" + "="*60)
        print("PHASE 2: Model Pruning")
        print("="*60)
        pruned_model = prune_lmwp_model(trained_model, prune_ratio=0.1)
        
        # Step 5: Finetune pruned model
        print("\n" + "="*60)
        print("PHASE 3: Finetuning (30 epochs)")
        print("="*60)
        final_model = finetune_lmwp_model(pruned_model, data_yaml, epochs=30)
        
        # Step 6: Evaluate final model
        print("\n" + "="*60)
        print("Final Model Evaluation")
        print("="*60)
        
        model = YOLO(final_model)
        metrics = model.val(data=data_yaml, imgsz=640, device='cpu')
        
        print("\nFinal Model Metrics:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # Test on sample image
        test_imgs = list(Path('datasets/coco8-grayscale/images/val').glob('*.jpg'))
        if test_imgs:
            test_img = test_imgs[0]
            results = model(test_img)
            
            # Save prediction
            for r in results:
                im_array = r.plot()
                im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite('lmwp_test_prediction.jpg', im)
            
            print("\nTest prediction saved as: lmwp_test_prediction.jpg")
        
        print("\n" + "="*60)
        print("LMWP-YOLO Training Pipeline Completed Successfully!")
        print(f"Final model: {final_model}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()