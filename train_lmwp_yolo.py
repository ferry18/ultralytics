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

def prepare_grayscale_coco8():
    """Download COCO8 and convert to grayscale."""
    print("Preparing grayscale COCO8 dataset...")
    
    # Download COCO8 if not exists
    from ultralytics.utils.downloads import download
    
    # COCO8 dataset URL
    url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/coco8.zip'
    download_path = Path('datasets/coco8.zip')
    
    # Download if not exists
    if not download_path.exists():
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download([url], dir=download_path.parent)
    
    # Extract
    import zipfile
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall('datasets/')
    
    # Convert images to grayscale
    coco8_path = Path('datasets/coco8')
    grayscale_path = Path('datasets/coco8-grayscale')
    
    # Copy structure
    if grayscale_path.exists():
        shutil.rmtree(grayscale_path)
    shutil.copytree(coco8_path, grayscale_path)
    
    # Convert all images to grayscale
    for split in ['train', 'val']:
        img_dir = grayscale_path / 'images' / split
        print(f"Converting {split} images to grayscale...")
        
        for img_path in tqdm(list(img_dir.glob('*.jpg'))):
            # Read image
            img = cv2.imread(str(img_path))
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert back to 3 channels for YOLO compatibility
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Save
            cv2.imwrite(str(img_path), gray_3ch)
    
    # Update yaml to point to grayscale dataset
    yaml_path = grayscale_path / 'coco8.yaml'
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths
    data['path'] = str(grayscale_path.absolute())
    data['train'] = 'images/train'
    data['val'] = 'images/val'
    
    # Save updated yaml
    grayscale_yaml = 'datasets/coco8-grayscale.yaml'
    with open(grayscale_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Grayscale COCO8 dataset prepared at: {grayscale_path}")
    return grayscale_yaml

def train_initial_model(model_yaml, data_yaml, epochs=30):
    """Train the initial model."""
    print("\n" + "="*60)
    print("Training initial LMWP-YOLO model")
    print("="*60)
    
    # Initialize model
    model = YOLO(model_yaml)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,  # Adjust based on available memory
        device='cpu',  # Use 'cuda' if GPU available
        project='runs/train',
        name='lmwp-yolo-initial',
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        plots=True,
    )
    
    # Return best model path
    best_model = Path('runs/train/lmwp-yolo-initial/weights/best.pt')
    print(f"\nInitial training completed. Best model saved at: {best_model}")
    return str(best_model)

def prune_model(model_path, prune_ratio=0.1):
    """Prune the trained model using structured pruning."""
    print("\n" + "="*60)
    print(f"Pruning model with ratio: {prune_ratio}")
    print("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Get model architecture
    import torch
    import torch.nn.utils.prune as prune
    
    # Apply structured pruning to Conv layers
    pruned_count = 0
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Apply structured pruning (channel pruning)
            prune.ln_structured(
                module, 
                name='weight', 
                amount=prune_ratio, 
                n=2, 
                dim=0  # Prune output channels
            )
            pruned_count += 1
    
    print(f"Pruned {pruned_count} Conv2d layers")
    
    # Remove pruning reparametrization to make permanent
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
    
    # Save pruned model
    pruned_path = model_path.replace('.pt', '_pruned.pt')
    model.save(pruned_path)
    print(f"Pruned model saved at: {pruned_path}")
    
    return pruned_path

def finetune_pruned_model(model_path, data_yaml, epochs=30):
    """Finetune the pruned model."""
    print("\n" + "="*60)
    print("Finetuning pruned model")
    print("="*60)
    
    # Load pruned model
    model = YOLO(model_path)
    
    # Finetune with lower learning rate
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device='cpu',  # Use 'cuda' if GPU available
        project='runs/finetune',
        name='lmwp-yolo-pruned',
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        plots=True,
        lr0=0.001,  # Lower learning rate for finetuning
        lrf=0.01,   # Lower final learning rate
    )
    
    # Return best finetuned model
    best_model = Path('runs/finetune/lmwp-yolo-pruned/weights/best.pt')
    print(f"\nFinetuning completed. Best model saved at: {best_model}")
    return str(best_model)

def main():
    """Main training pipeline."""
    print("LMWP-YOLO Training Pipeline")
    print("="*60)
    
    # Check if we have a working YAML configuration
    model_yamls = [
        'yolov11s-lcnet-mafrneck.yaml',
        'yolov11l-lcnet-mafrneck.yaml',
        'LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml'
    ]
    
    model_yaml = None
    for yaml_file in model_yamls:
        if os.path.exists(yaml_file):
            model_yaml = yaml_file
            print(f"Using model configuration: {model_yaml}")
            break
    
    if not model_yaml:
        print("ERROR: No valid model YAML found!")
        return
    
    try:
        # Step 1: Prepare grayscale COCO8 dataset
        data_yaml = prepare_grayscale_coco8()
        
        # Step 2: Train initial model
        trained_model = train_initial_model(model_yaml, data_yaml, epochs=30)
        
        # Step 3: Prune the model (10% pruning ratio)
        pruned_model = prune_model(trained_model, prune_ratio=0.1)
        
        # Step 4: Finetune pruned model
        final_model = finetune_pruned_model(pruned_model, data_yaml, epochs=30)
        
        print("\n" + "="*60)
        print("Training pipeline completed successfully!")
        print(f"Final model: {final_model}")
        print("="*60)
        
        # Test the final model
        print("\nTesting final model on a sample image...")
        model = YOLO(final_model)
        
        # Get a test image
        test_img = list(Path('datasets/coco8-grayscale/images/val').glob('*.jpg'))[0]
        results = model(test_img)
        
        # Save prediction
        for r in results:
            im_array = r.plot()
            im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test_prediction.jpg', im)
        
        print("Test prediction saved as: test_prediction.jpg")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()