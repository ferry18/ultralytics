import pytest
import torch
from ultralytics import YOLO

CONFIG_PATH = "LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml"

@pytest.mark.parametrize("img_size", [256, 320])
def test_model_forward(img_size):
    """Build the model and ensure a dummy forward pass succeeds without runtime errors."""
    model = YOLO(CONFIG_PATH).model  # DetectionModel instance
    x = torch.randn(1, 3, img_size, img_size)
    preds = model(x)
    # Detect head returns a tuple of predictions and None (train outputs)
    assert isinstance(preds, (list, tuple)), "Forward should return a list/tuple"
    assert len(preds[0]) == model.model[-1].nl, "Number of detection layers should match"