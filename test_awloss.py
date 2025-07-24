"""
Test script for AWLoss implementation.
Verifies that the loss computation works correctly.
"""

import torch
import torch.nn as nn
from ultralytics.nn.losses.awloss import (
    NormalizedWassersteinDistance, 
    AreaWeighting, 
    ScaleDifference,
    AWLoss
)


def test_nwd():
    """Test Normalized Wasserstein Distance computation."""
    print("Testing NWD...")
    
    # Create test boxes (batch_size=2, n_boxes=3)
    pred_boxes = torch.tensor([
        [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]],
        [[15, 15, 25, 25], [35, 35, 45, 45], [55, 55, 65, 65]]
    ], dtype=torch.float32)
    
    target_boxes = torch.tensor([
        [[12, 12, 22, 22], [32, 32, 42, 42], [52, 52, 62, 62]],
        [[14, 14, 24, 24], [34, 34, 44, 44], [54, 54, 64, 64]]
    ], dtype=torch.float32)
    
    nwd = NormalizedWassersteinDistance(C=10.0)
    scores = nwd(pred_boxes, target_boxes)
    
    print(f"  Input shapes: pred {pred_boxes.shape}, target {target_boxes.shape}")
    print(f"  NWD scores shape: {scores.shape}")
    print(f"  NWD scores: {scores}")
    print(f"  Mean NWD: {scores.mean():.4f}")
    
    # Verify scores are in (0, 1)
    assert torch.all(scores >= 0) and torch.all(scores <= 1), "NWD scores should be in [0, 1]"
    print("  ✓ NWD scores are in valid range [0, 1]")
    
    # Test identical boxes should give score close to 1
    identical_scores = nwd(pred_boxes, pred_boxes)
    assert torch.allclose(identical_scores, torch.ones_like(identical_scores), atol=1e-6), \
        "Identical boxes should have NWD ≈ 1"
    print("  ✓ Identical boxes give NWD ≈ 1")
    
    print("  ✓ NWD test passed!\n")


def test_area_weighting():
    """Test area-based weighting."""
    print("Testing Area Weighting...")
    
    # Create boxes with different sizes
    # Small box: 10x10, Medium box: 30x30, Large box: 50x50
    target_boxes = torch.tensor([
        [[100, 100, 10, 10], [200, 200, 30, 30], [300, 300, 50, 50]],
        [[150, 150, 5, 5], [250, 250, 40, 40], [350, 350, 80, 80]]
    ], dtype=torch.float32)
    
    area_weight = AreaWeighting(alpha=10.0, beta=0.5)
    weights = area_weight(target_boxes, (640, 640))
    
    print(f"  Box sizes: {target_boxes[..., 2:].tolist()}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights: {weights}")
    
    # Verify smaller boxes get higher weights
    assert weights[0, 0] > weights[0, 1] > weights[0, 2], "Smaller boxes should get higher weights"
    assert weights[1, 0] > weights[1, 1] > weights[1, 2], "Smaller boxes should get higher weights"
    print("  ✓ Smaller boxes get higher weights")
    
    # Verify weights are in reasonable range
    assert torch.all(weights >= 1.0) and torch.all(weights <= 2.0), "Weights should be in [1, 2]"
    print("  ✓ Weights are in valid range [1, 2]")
    
    print("  ✓ Area weighting test passed!\n")


def test_scale_difference():
    """Test scale difference computation."""
    print("Testing Scale Difference...")
    
    # Create boxes with different scale differences
    pred_boxes = torch.tensor([
        [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]],
        [[15, 15, 25, 25], [35, 35, 45, 45], [55, 55, 65, 65]]
    ], dtype=torch.float32)
    
    # Target boxes with varying differences
    target_boxes = torch.tensor([
        [[10, 10, 20, 20], [30, 30, 50, 50], [50, 50, 80, 80]],  # Same, wider, much wider
        [[15, 15, 30, 30], [35, 35, 45, 55], [55, 55, 65, 65]]   # Wider, taller, same
    ], dtype=torch.float32)
    
    scale_diff = ScaleDifference()
    losses = scale_diff(pred_boxes, target_boxes)
    
    print(f"  Scale difference shape: {losses.shape}")
    print(f"  Scale differences: {losses}")
    
    # Verify same size boxes have near-zero loss
    assert losses[0, 0] < 0.01, "Same size boxes should have near-zero scale loss"
    assert losses[1, 2] < 0.01, "Same size boxes should have near-zero scale loss"
    print("  ✓ Same size boxes have near-zero scale loss")
    
    # Verify larger differences have higher loss
    assert losses[0, 2] > losses[0, 1] > losses[0, 0], "Larger scale differences should have higher loss"
    print("  ✓ Larger scale differences have higher loss")
    
    print("  ✓ Scale difference test passed!\n")


def test_awloss_integration():
    """Test complete AWLoss integration."""
    print("Testing Complete AWLoss...")
    
    batch_size = 2
    num_anchors = 100
    num_classes = 80
    
    # Create mock predictions
    predictions = {
        'box': torch.randn(batch_size, num_anchors, 4) * 50 + 320,  # Random boxes around center
        'cls': torch.randn(batch_size, num_anchors, num_classes),    # Random class logits
        'obj': torch.randn(batch_size, num_anchors)                  # Random objectness
    }
    
    # Create mock targets
    targets = {
        'box': torch.randn(batch_size, num_anchors, 4) * 50 + 320,
        'cls': torch.zeros(batch_size, num_anchors, num_classes),
        'obj': torch.zeros(batch_size, num_anchors),
        'mask': torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    }
    
    # Set some targets as valid
    targets['mask'][:, :10] = True  # First 10 anchors are valid
    targets['cls'][:, :10, 0] = 1.0  # Class 0
    targets['obj'][:, :10] = 1.0
    
    # Create loss
    awloss = AWLoss()
    
    # Compute loss
    total_loss, loss_items = awloss(predictions, targets, image_size=(640, 640))
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss components:")
    for k, v in loss_items.items():
        print(f"    - {k}: {v.item():.4f}")
    
    # Verify loss is positive
    assert total_loss > 0, "Loss should be positive"
    print("  ✓ Loss is positive")
    
    # Verify all components are computed
    expected_keys = ['box_loss', 'cls_loss', 'obj_loss', 'nwd_mean', 'area_weight_mean', 'scale_loss']
    assert all(k in loss_items for k in expected_keys), "All loss components should be present"
    print("  ✓ All loss components computed")
    
    # Test gradient flow
    total_loss.backward()
    print("  ✓ Gradients computed successfully")
    
    print("  ✓ AWLoss integration test passed!\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("AWLoss Implementation Tests")
    print("=" * 80)
    
    test_nwd()
    test_area_weighting()
    test_scale_difference()
    test_awloss_integration()
    
    print("=" * 80)
    print("All tests passed! AWLoss implementation is working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    main()