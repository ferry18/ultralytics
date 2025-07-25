#!/usr/bin/env python3
"""
Train LMWP-YOLO model.
"""

from ultralytics import YOLO

# Load the model
model = YOLO('LMWP-YOLO-main/yolov11-lcnet-mafrneck-final.yaml')

# Train the model
results = model.train(
    data='coco128.yaml',  # Use COCO128 for testing, change to your dataset
    epochs=10,            # Adjust based on your needs
    batch=16,             # Adjust based on your GPU memory
    imgsz=640,
    device=0,             # Use GPU 0, or 'cpu' for CPU training
    workers=8,
    patience=50,
    save=True,
    project='lmwp-yolo-training',
    name='exp'
)

# Validate the model
metrics = model.val()

print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")
