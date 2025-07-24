#!/usr/bin/env python3
"""
Simple test to verify COCO8 grayscale dataset works with YOLO.
"""

from ultralytics import YOLO


def main():
    """Test basic YOLO training on COCO8 grayscale."""
    
    print("\n" + "=" * 80)
    print("COCO8 Grayscale Training Test")
    print("=" * 80)
    
    # Use standard YOLO model
    model = YOLO('yolo11n.yaml')
    
    # Train for 3 epochs just to verify it works
    print("\nStarting training on COCO8 grayscale...")
    
    results = model.train(
        data='ultralytics/cfg/datasets/coco8-grayscale.yaml',
        epochs=3,
        imgsz=640,
        batch=4,
        device='cpu',
        workers=2,
        project='runs/test',
        name='coco8-gray',
        exist_ok=True,
        verbose=True
    )
    
    print("\n✓ Training completed!")
    
    # Validate
    val_results = model.val()
    print(f"\nValidation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {val_results.box.map:.4f}")
    
    print("\n✓ COCO8 grayscale dataset is working correctly!")
    
    return True


if __name__ == "__main__":
    main()