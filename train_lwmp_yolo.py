"""
Train LWMP-YOLO for small drone detection.

This script demonstrates how to use the LWMP-YOLO model with:
- PP-LCNet backbone
- MAFR and C3TR modules
- Area-weighted Wasserstein Loss
- L1-norm based pruning
"""

from ultralytics import YOLO
from ultralytics.utils.loss import AWDetectionLoss
from ultralytics.nn.modules import prune_model
import torch
from pathlib import Path


def train_lwmp_yolo():
    """Train LWMP-YOLO model with custom configuration."""
    
    # Load LWMP-YOLO configuration
    model = YOLO('ultralytics/cfg/models/11/yolo11-lwmp.yaml')
    
    # Override the default loss with AWDetectionLoss
    # This is done by modifying the model's loss function after initialization
    def setup_awloss(trainer):
        """Replace default loss with AWDetectionLoss."""
        if hasattr(trainer, 'model'):
            # Store original loss class
            trainer.loss_class = AWDetectionLoss
            # Create AWDetectionLoss instance
            trainer.loss = AWDetectionLoss(trainer.model)
            print("Successfully replaced loss with AWDetectionLoss")
    
    # Train the model
    results = model.train(
        data='path/to/drone-dataset.yaml',  # Replace with your dataset config
        epochs=100,
        batch=48,
        imgsz=640,
        device=0,  # GPU device
        workers=8,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Cls loss gain  
        dfl=1.5,  # DFL loss gain
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
        copy_paste=0.0,  # Copy-paste augmentation
        auto_augment='randaugment',
        callbacks={
            'on_pretrain_routine_start': setup_awloss  # Setup AWLoss
        }
    )
    
    # After training, apply pruning
    print("\nApplying L1-norm based pruning...")
    
    # Load the trained model
    trained_model = YOLO(results.save_dir / 'weights' / 'best.pt')
    
    # Apply pruning with 30% filter reduction
    pruned_model = prune_model(trained_model.model, pruning_ratio=0.3)
    
    # Fine-tune the pruned model
    print("\nFine-tuning pruned model...")
    pruned_yolo = YOLO(model=pruned_model)
    
    pruned_results = pruned_yolo.train(
        data='path/to/drone-dataset.yaml',
        epochs=20,  # Fewer epochs for fine-tuning
        batch=48,
        imgsz=640,
        device=0,
        resume=False,  # Don't resume from checkpoint
        lr0=0.001,  # Lower learning rate for fine-tuning
        callbacks={
            'on_pretrain_routine_start': setup_awloss
        }
    )
    
    print(f"\nTraining complete!")
    print(f"Original model saved at: {results.save_dir}")
    print(f"Pruned model saved at: {pruned_results.save_dir}")
    
    return results, pruned_results


def validate_lwmp_yolo(model_path, data_path):
    """Validate LWMP-YOLO model performance."""
    
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=data_path,
        batch=48,
        imgsz=640,
        device=0,
        workers=8,
        save_json=True,  # Save results in COCO format
        save_hybrid=True,  # Save hybrid labels
        conf=0.001,  # Confidence threshold
        iou=0.6,  # NMS IoU threshold
        max_det=300  # Maximum detections per image
    )
    
    print(f"\nValidation Results:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    
    return metrics


def export_lwmp_yolo(model_path, format='onnx'):
    """Export LWMP-YOLO model for deployment."""
    
    model = YOLO(model_path)
    
    # Export model
    export_path = model.export(
        format=format,  # 'onnx', 'torchscript', 'coreml', 'tflite', etc.
        imgsz=640,
        batch=1,
        device=0,
        simplify=True,  # Simplify ONNX model
        opset=12,  # ONNX opset version
        half=False,  # FP16 quantization
        int8=False  # INT8 quantization
    )
    
    print(f"\nModel exported to: {export_path}")
    
    # Get model info
    if format == 'onnx':
        import onnx
        model_onnx = onnx.load(export_path)
        
        # Calculate model size
        import os
        model_size = os.path.getsize(export_path) / (1024 * 1024)  # MB
        
        print(f"Model size: {model_size:.2f} MB")
        print(f"Input shape: {model_onnx.graph.input[0].type.tensor_type.shape}")
        print(f"Output shape: {model_onnx.graph.output[0].type.tensor_type.shape}")
    
    return export_path


if __name__ == '__main__':
    # Example usage
    
    # 1. Train LWMP-YOLO
    print("Starting LWMP-YOLO training...")
    results, pruned_results = train_lwmp_yolo()
    
    # 2. Validate the models
    print("\nValidating original model...")
    validate_lwmp_yolo(
        model_path=results.save_dir / 'weights' / 'best.pt',
        data_path='path/to/drone-dataset.yaml'
    )
    
    print("\nValidating pruned model...")
    validate_lwmp_yolo(
        model_path=pruned_results.save_dir / 'weights' / 'best.pt',
        data_path='path/to/drone-dataset.yaml'
    )
    
    # 3. Export models for deployment
    print("\nExporting models...")
    export_lwmp_yolo(
        model_path=pruned_results.save_dir / 'weights' / 'best.pt',
        format='onnx'
    )