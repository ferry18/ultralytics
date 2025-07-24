"""
Training script for LWMP-YOLO with all modifications:
1. LCBackbone (simplified as standard backbone for initial testing)
2. MAFR modules in neck
3. AWLoss (to be integrated)
4. Pruning after training
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os


class GrayscaleTransform:
    """Transform to convert images to grayscale during training."""
    def __init__(self):
        pass
    
    def __call__(self, im):
        """Convert image to grayscale."""
        if len(im.shape) == 3 and im.shape[2] == 3:
            # Convert BGR to grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Add channel dimension
            gray = np.expand_dims(gray, axis=2)
            return gray
        return im


def download_dataset():
    """Download COCO8 dataset if not present."""
    from ultralytics.utils.downloads import download
    
    dataset_path = Path("coco8")
    if not dataset_path.exists():
        print("Downloading COCO8 dataset...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
        download(url, dir=Path("."))
        
        # Extract zip file
        import zipfile
        with zipfile.ZipFile("coco8.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("coco8.zip")
        print("Dataset downloaded and extracted.")


def convert_dataset_to_grayscale():
    """Convert COCO8 dataset images to grayscale."""
    dataset_path = Path("coco8")
    
    for split in ['train', 'val']:
        img_dir = dataset_path / 'images' / split
        if img_dir.exists():
            print(f"Converting {split} images to grayscale...")
            for img_path in img_dir.glob('*.jpg'):
                # Read image
                img = cv2.imread(str(img_path))
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Save back
                cv2.imwrite(str(img_path), gray)
            print(f"Converted {split} images.")


def train_lwmp_yolo():
    """Train LWMP-YOLO model."""
    
    # Download dataset if needed
    download_dataset()
    
    # Note: For actual grayscale training, we would convert images here
    # convert_dataset_to_grayscale()  # Uncomment to actually convert
    
    # Initialize model with LWMP configuration
    print("Initializing LWMP-YOLO model...")
    print("Configuration: PP-LCNet x0.75 backbone + MAFR + P2/P3 detection")
    print("Target: 1.23M parameters, 2.71MB model size")
    model = YOLO('ultralytics/cfg/models/11/yolo11-lwmp-author-corrected.yaml')
    
    # Training arguments
    args = {
        'data': 'ultralytics/cfg/datasets/coco8.yaml',  # Use regular coco8 for testing
        'epochs': 10,  # Reduced for testing
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'project': 'lwmp-yolo-results',
        'name': 'train',
        'exist_ok': True,
        'pretrained': False,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    # Train model
    print("Starting training...")
    results = model.train(**args)
    
    # Save trained model
    print("Training completed. Model saved.")
    
    # Apply pruning (simplified version)
    print("\nApplying pruning...")
    from ultralytics.utils.pruning import prune_model, compute_pruning_stats
    
    # Get the trained model
    trained_model = model.model
    
    # Apply pruning with 30% reduction
    # Note: This is a simplified version. Full implementation would require
    # rebuilding the model architecture after pruning
    try:
        pruned_model = prune_model(trained_model, prune_ratio=0.3)
        stats = compute_pruning_stats(trained_model, pruned_model)
        print(f"Pruning stats: {stats}")
    except Exception as e:
        print(f"Pruning not fully implemented yet: {e}")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.val()
    
    print(f"\nValidation metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results, metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Train model
    model, results, metrics = train_lwmp_yolo()
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: lwmp-yolo-results/train/")