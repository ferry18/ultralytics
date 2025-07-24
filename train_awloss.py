"""
Training script for LWMP-YOLO with AWLoss integration.
This script trains a YOLO model with the Area-weighted Wasserstein Loss.
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.awloss_integration import v8DetectionAWLoss
from ultralytics.nn.tasks import DetectionModel
import torch

# Monkey patch to replace the loss computation
original_loss_fn = DetectionModel.loss

def awloss_wrapper(self, batch, preds=None):
    """Compute loss using AWLoss instead of standard v8DetectionLoss."""
    if not hasattr(self, 'awloss_criterion'):
        # Initialize AWLoss criterion
        self.awloss_criterion = v8DetectionAWLoss(self)
    
    if preds is None:
        preds = self.forward(batch["img"])
    
    # Use AWLoss
    return self.awloss_criterion(preds, batch)

# Apply the patch
DetectionModel.loss = awloss_wrapper


def monkey_patch_awloss():
    """Apply AWLoss monkey patch to DetectionModel."""
    # The patch is already applied at module import
    # This function exists for explicit control
    if not hasattr(DetectionModel, '_original_loss'):
        DetectionModel._original_loss = original_loss_fn
    DetectionModel.loss = awloss_wrapper
    return original_loss_fn


def train_lwmp_yolo_awloss(
    model_yaml='ultralytics/cfg/models/11/yolo11-lwmp-author.yaml',
    data_yaml='ultralytics/cfg/datasets/coco8-grayscale.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/lwmp-awloss',
    name='train',
    exist_ok=False,
    pretrained=False,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,  # Box loss gain
    cls=0.5,  # Class loss gain  
    dfl=1.5,  # DFL loss gain
    nwd_C=10.0,  # NWD normalization constant
    area_alpha=10.0,  # Area weighting sigmoid scale
    area_beta=0.5,  # Area weighting sigmoid offset
    scale_weight=0.05,  # Scale difference weight
    workers=8,
    verbose=True,
    save=True,
    plots=True
):
    """
    Train LWMP-YOLO with AWLoss.
    
    Args:
        Various training parameters with defaults
    
    Returns:
        results: Training results
    """
    
    print("=" * 80)
    print("LWMP-YOLO Training with AWLoss")
    print("=" * 80)
    print(f"Model: {model_yaml}")
    print(f"Dataset: {data_yaml}")
    print(f"Device: {device}")
    print(f"AWLoss Parameters:")
    print(f"  - NWD C: {nwd_C}")
    print(f"  - Area α: {area_alpha}")
    print(f"  - Area β: {area_beta}")
    print(f"  - Scale weight: {scale_weight}")
    print("=" * 80)
    
    # Initialize model
    model = YOLO(model_yaml)
    
    # Prepare training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': exist_ok,
        'pretrained': pretrained,
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,
        'warmup_bias_lr': warmup_bias_lr,
        'box': box,
        'cls': cls,
        'dfl': dfl,
        'workers': workers,
        'verbose': verbose,
        'save': save,
        'plots': plots,
        # AWLoss specific parameters
        'nwd_C': nwd_C,
        'area_alpha': area_alpha,
        'area_beta': area_beta,
        'scale_weight': scale_weight,
    }
    
    # Train the model
    try:
        results = model.train(**train_args)
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
        # Validate the model
        print("\nRunning validation...")
        val_results = model.val()
        
        print(f"\nValidation Results:")
        print(f"  - mAP@0.5: {val_results.box.map50:.4f}")
        print(f"  - mAP@0.5:0.95: {val_results.box.map:.4f}")
        
        # Save model info
        print(f"\nModel saved to: {model.trainer.save_dir}")
        
        # Calculate model size
        model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"Model size: {model_size_mb:.2f} MB")
            
            # Count parameters
            checkpoint = torch.load(model_path, map_location='cpu')
            model_dict = checkpoint['model']
            total_params = sum(p.numel() for p in model_dict.parameters() if p.requires_grad)
            print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check if custom YAML path is provided
    import argparse
    parser = argparse.ArgumentParser(description='Train LWMP-YOLO with AWLoss')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/yolo11-lwmp-author.yaml',
                        help='Path to model YAML file')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/coco8-grayscale.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--nwd-c', type=float, default=10.0,
                        help='NWD normalization constant C')
    parser.add_argument('--area-alpha', type=float, default=10.0,
                        help='Area weighting sigmoid scale')
    parser.add_argument('--area-beta', type=float, default=0.5,
                        help='Area weighting sigmoid offset')
    parser.add_argument('--scale-weight', type=float, default=0.05,
                        help='Scale difference weight')
    
    args = parser.parse_args()
    
    # Train with AWLoss
    train_lwmp_yolo_awloss(
        model_yaml=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        nwd_C=args.nwd_c,
        area_alpha=args.area_alpha,
        area_beta=args.area_beta,
        scale_weight=args.scale_weight
    )