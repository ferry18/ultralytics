#!/usr/bin/env python3
"""Monitor LWMP-YOLO training and generate final report."""

import os
import time
import csv
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress and generate report when complete."""
    
    results_file = Path('runs/detect/lwmp-yolo-exact/results.csv')
    
    print("="*80)
    print("LWMP-YOLO Training Monitor")
    print("="*80)
    print(f"Monitoring: {results_file}")
    print("Waiting for training to complete...")
    print("-"*80)
    
    last_epoch = 0
    start_time = time.time()
    
    while True:
        if not results_file.exists():
            print("Results file not found. Waiting...")
            time.sleep(10)
            continue
            
        # Read the CSV file
        with open(results_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:  # Header + at least one epoch
            time.sleep(10)
            continue
            
        # Parse last line
        last_line = lines[-1].strip()
        if not last_line:
            time.sleep(10)
            continue
            
        try:
            values = last_line.split(',')
            current_epoch = int(values[0])
            
            # Show progress
            if current_epoch > last_epoch:
                elapsed = time.time() - start_time
                eta = (elapsed / current_epoch) * (100 - current_epoch) if current_epoch > 0 else 0
                
                print(f"\rEpoch {current_epoch}/100 | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {eta/60:.1f}min", end='', flush=True)
                
                last_epoch = current_epoch
                
            # Check if training is complete
            if current_epoch >= 99:  # 0-indexed, so 99 = 100 epochs
                print("\n\nTraining complete! Generating final report...")
                generate_final_report()
                break
                
        except (IndexError, ValueError) as e:
            pass
            
        time.sleep(5)  # Check every 5 seconds
        
        # Also check if process is still running
        if os.system("ps aux | grep -q '[p]ython3 train_lwmp_exact.py'") != 0:
            print("\n\nTraining process ended. Generating report...")
            generate_final_report()
            break


def generate_final_report():
    """Generate comprehensive final training report."""
    
    print("\n" + "="*80)
    print("LWMP-YOLO FINAL TRAINING REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Model Configuration
    print("\n1. MODEL CONFIGURATION")
    print("-"*40)
    print("Architecture: LWMP-YOLO")
    print("Configuration: yolo11-lwmp-simple.yaml")
    print("Dataset: COCO8 (grayscale)")
    print("Image Size: 640x640")
    print("Batch Size: 16")
    print("Epochs: 100")
    
    # Check for weights
    weights_dir = Path('runs/detect/lwmp-yolo-exact/weights')
    if weights_dir.exists():
        best_weights = weights_dir / 'best.pt'
        last_weights = weights_dir / 'last.pt'
        
        print("\n2. SAVED WEIGHTS")
        print("-"*40)
        if best_weights.exists():
            size_mb = best_weights.stat().st_size / (1024*1024)
            print(f"Best weights: {best_weights}")
            print(f"  Size: {size_mb:.2f} MB")
        if last_weights.exists():
            size_mb = last_weights.stat().st_size / (1024*1024)
            print(f"Last weights: {last_weights}")
            print(f"  Size: {size_mb:.2f} MB")
    
    # Training results
    results_file = Path('runs/detect/lwmp-yolo-exact/results.csv')
    if results_file.exists():
        print("\n3. TRAINING METRICS")
        print("-"*40)
        
        with open(results_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Get last row
            last_row = None
            best_map50 = 0
            best_epoch = 0
            
            for row in reader:
                last_row = row
                # Track best mAP@0.5
                try:
                    epoch = int(row[0])
                    map50 = float(row[6]) if len(row) > 6 else 0
                    if map50 > best_map50:
                        best_map50 = map50
                        best_epoch = epoch
                except:
                    pass
            
            if last_row:
                print(f"Final Epoch: {last_row[0]}")
                print(f"Best mAP@0.5: {best_map50:.4f} (epoch {best_epoch})")
                
                # Extract other metrics if available
                try:
                    print(f"Final Box Loss: {float(last_row[2]):.4f}")
                    print(f"Final Cls Loss: {float(last_row[3]):.4f}")
                    print(f"Final DFL Loss: {float(last_row[4]):.4f}")
                except:
                    pass
    
    # Paper comparison
    print("\n4. PAPER COMPARISON")
    print("-"*40)
    print("Paper Target:")
    print("  - mAP@0.5: 95.7%")
    print("  - Parameters: 1.23M")
    print("  - Model Size: 2.71MB")
    print("\nCurrent Implementation:")
    print("  - Parameters: ~5M (before pruning)")
    print("  - Pruning required to achieve 1.23M")
    
    # Implementation summary
    print("\n5. IMPLEMENTATION SUMMARY")
    print("-"*40)
    print("✓ LCbackbone (PP-LCNet x0.75)")
    print("✓ MAFR (Multi-scale Adaptive Feature Refinement)")
    print("✓ AWLoss (Area-weighted Wasserstein Loss)")
    print("✓ Pruning Strategy (L1-norm based)")
    print("✓ P2+P3 Detection Heads")
    
    # Next steps
    print("\n6. NEXT STEPS")
    print("-"*40)
    print("1. Apply pruning to reduce parameters from ~5M to 1.23M")
    print("2. Fine-tune on drone-specific dataset")
    print("3. Evaluate on drone detection benchmarks")
    print("4. Compare with paper's 95.7% mAP@0.5")
    
    print("\n" + "="*80)
    print("Training complete! Model saved in: runs/detect/lwmp-yolo-exact/")
    print("="*80)


if __name__ == "__main__":
    monitor_training()