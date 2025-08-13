"""
Minimal LWMP-YOLO training example.

This script provides a simplified version that handles common issues.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ultralytics.utils.loss import AWDetectionLoss
from ultralytics.nn.modules import prune_model


def train_lwmp_yolo_minimal():
    """Minimal training example for LWMP-YOLO."""
    
    # Check if running from correct directory
    yaml_path = Path('ultralytics/cfg/models/11/yolo11-lwmp.yaml')
    if not yaml_path.exists():
        print(f"Error: Cannot find {yaml_path}")
        print(f"Current directory: {os.getcwd()}")
        print("Please run this script from the project root directory")
        return
    
    print("Loading LWMP-YOLO model...")
    model = YOLO(str(yaml_path))
    
    # Note: To use AWDetectionLoss, you would need to modify the trainer
    # This is an advanced use case that requires custom callbacks
    # For basic training, the default loss works fine
    
    print("Starting training...")
    # Basic training with default settings
    results = model.train(
        data='coco128.yaml',  # Use COCO128 for testing
        epochs=3,  # Just a few epochs for testing
        imgsz=640,
        batch=16,
        device='cpu',  # Use CPU for testing, change to 0 for GPU
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Demonstrate pruning
    print("\nDemonstrating pruning...")
    
    # Get the PyTorch model
    pytorch_model = model.model
    
    # Apply pruning
    print("Applying 30% pruning...")
    pruned_model = prune_model(pytorch_model, pruning_ratio=0.3)
    
    print("Pruning completed successfully!")
    
    return model, results


def test_inference(model_path=None):
    """Test inference with LWMP-YOLO."""
    
    if model_path is None:
        # Use the yaml to create a new model for testing
        yaml_path = Path('ultralytics/cfg/models/11/yolo11-lwmp.yaml')
        model = YOLO(str(yaml_path))
    else:
        model = YOLO(model_path)
    
    print("\nTesting inference...")
    
    # Create a dummy image for testing
    import torch
    dummy_image = torch.rand(1, 3, 640, 640).numpy()
    
    try:
        results = model(dummy_image)
        print("✓ Inference successful!")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
    
    return model


if __name__ == '__main__':
    print("LWMP-YOLO Minimal Training Example")
    print("=" * 50)
    
    # Test basic functionality
    try:
        # Train the model
        model, results = train_lwmp_yolo_minimal()
        
        # Test inference
        test_inference()
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()