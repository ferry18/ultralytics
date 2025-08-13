# LWMP-YOLO Usage Guide

## Installation and Setup

1. **Directory Structure**: Ensure you're in the project root directory (where the `ultralytics` folder is located).

2. **Dependencies**: Install required packages:
```bash
pip install ultralytics torch torchvision
```

## Quick Start

### 1. Test Imports
First, verify all components are properly imported:

```bash
python test_imports.py
```

This will check all imports and report any issues.

### 2. Basic Training
Use the minimal training script for a quick test:

```bash
python train_lwmp_yolo_minimal.py
```

This script:
- Uses COCO128 dataset (automatically downloaded)
- Runs for 3 epochs
- Tests pruning functionality
- Validates inference

### 3. Full Training Example
For production training with all features:

```python
from ultralytics import YOLO
from ultralytics.nn.modules import prune_model

# Load model
model = YOLO('ultralytics/cfg/models/11/yolo11-lwmp.yaml')

# Train
results = model.train(
    data='path/to/your/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0  # GPU
)

# Apply pruning
pruned_model = prune_model(model.model, pruning_ratio=0.3)
```

## Common Issues and Solutions

### Import Errors

**Issue**: `ImportError: cannot import name 'prune_model'`

**Solution**: Ensure you're running from the project root directory. The imports are fixed in the current implementation.

### Path Issues

**Issue**: Cannot find YAML configuration file

**Solution**: 
- Run scripts from the project root directory
- Use absolute paths or Path objects:
```python
from pathlib import Path
yaml_path = Path(__file__).parent / 'ultralytics/cfg/models/11/yolo11-lwmp.yaml'
```

### AWLoss Integration

To use AWDetectionLoss, you need custom training logic:

```python
from ultralytics import YOLO
from ultralytics.utils.loss import AWDetectionLoss
from ultralytics.engine.trainer import BaseTrainer

class LWMPTrainer(BaseTrainer):
    def get_loss(self):
        """Override to use AWDetectionLoss."""
        return AWDetectionLoss(self.model)

# Use custom trainer
model = YOLO('ultralytics/cfg/models/11/yolo11-lwmp.yaml')
model._trainer = LWMPTrainer
```

## Model Architecture

The LWMP-YOLO model includes:

1. **PP-LCNet Backbone**: Lightweight feature extractor with depthwise separable convolutions
2. **MAFR Module**: Multidimensional attention for enhanced feature representation
3. **C3TR Module**: Transformer-based global context modeling
4. **P2 Detection Head**: For improved small object detection
5. **AWLoss**: Area-weighted Wasserstein loss for small objects

## Performance Tips

1. **GPU Usage**: Always use GPU for training (`device=0`)
2. **Batch Size**: Adjust based on GPU memory (typically 16-48)
3. **Image Size**: 640x640 is recommended for small object detection
4. **Pruning**: Apply after initial training, then fine-tune

## Dataset Format

Use standard YOLO format:
```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 1  # number of classes
names: ['drone']  # class names
```

## Validation

```python
# Validate model
metrics = model.val(
    data='path/to/dataset.yaml',
    batch=32,
    imgsz=640
)

print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
```

## Export for Deployment

```python
# Export to ONNX
model.export(format='onnx', imgsz=640, simplify=True)

# Export to TensorRT
model.export(format='engine', imgsz=640, half=True)
```

## Troubleshooting

1. **Memory Issues**: Reduce batch size or image size
2. **Slow Training**: Ensure CUDA is available with `torch.cuda.is_available()`
3. **Poor Performance**: Check dataset quality and increase training epochs
4. **Module Not Found**: Verify you're in the correct directory with proper Python path

For additional help, check the implementation files:
- `ultralytics/nn/modules/lwmp_modules.py` - Core modules
- `ultralytics/utils/loss.py` - AWLoss implementation
- `ultralytics/cfg/models/11/yolo11-lwmp.yaml` - Model configuration