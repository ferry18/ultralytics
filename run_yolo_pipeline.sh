#!/usr/bin/env bash

# Exit on first error, unset var, or pipe-fail
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
# Override with e.g. BASE_EPOCHS=2 FT_EPOCHS=1 DEVICE=0 bash run_yolo_pipeline.sh
DEVICE="${DEVICE:-auto}"
RUNS_DIR="${RUNS_DIR:-runs}"
DATA_CFG="ultralytics/cfg/datasets/coco8-grayscale.yaml"
BASE_MODEL_CFG="LMWP-YOLO-main/yolov11-lcnet-mafrneck.yaml"
PRUNE_PERCENT=0.20  # 20 % channel sparsity
BASE_EPOCHS="${BASE_EPOCHS:-1}"   # baseline epochs
FT_EPOCHS="${FT_EPOCHS:-1}"       # fine-tune epochs
BATCH=16
IMGSZ=640

# Output paths
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_DIR="$RUNS_DIR/pipeline_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Helper for pretty prints
log(){ echo -e "[\e[1;34mINFO\e[0m] $*"; }

# -------------------------------------
# 1. Baseline training with AW-Loss
# -------------------------------------
log "Starting baseline training …"
~/.local/bin/yolo train \
  model="$BASE_MODEL_CFG" \
  data="$DATA_CFG" \
  epochs=$BASE_EPOCHS \
  batch=$BATCH \
  imgsz=$IMGSZ \
  device=$DEVICE \
  | tee "$LOG_DIR/baseline_train.log"

# Grab weights path of the best checkpoint just created
BASE_RUN_DIR=$(ls -td $RUNS_DIR/detect/train* | head -1)
BASE_BEST_WEIGHTS="$BASE_RUN_DIR/weights/best.pt"
# Make variables available to the inline Python script
export BASE_BEST_WEIGHTS BASE_MODEL_CFG LOG_DIR PRUNE_PERCENT

# -------------------------------------
# 2. Channel pruning (20 %) and YAML export
# -------------------------------------
log "Pruning $PRUNE_PERCENT of channels …"
# Ultralytics ≥8.2.0 exposes model.prune() programmatically but not via CLI yet.
python - <<'PY'
import sys, torch, yaml, os
percent=float(os.environ.get('PRUNE_PERCENT','0.2'))
weights=os.environ['BASE_BEST_WEIGHTS']
export_yaml=os.path.join(os.environ['LOG_DIR'], 'yolov11-lcnet-pruned.yaml')
export_weights=os.path.join(os.environ['LOG_DIR'], 'yolov11-lcnet-pruned.pt')

from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight
from torch.nn.utils import prune

def global_channel_prune(model, amount):
    """Prune convolutional layer output channels globally by L1-norm."""
    parameters_to_prune=[]
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels>1:
            parameters_to_prune.append((m,'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    # Remove pruning re-parameterisation (make permanent)
    for m,_ in parameters_to_prune:
        prune.remove(m,'weight')

autoshape=False
model=YOLO(weights, task='detect').model
model.eval()
global_channel_prune(model, amount=percent)

# Save pruned weights
ckpt={'model':model.state_dict()}
torch.save(ckpt, export_weights)

# Export a new YAML with width mult scaled by (1-percent)
base_yaml_path=os.environ['BASE_MODEL_CFG']
with open(base_yaml_path) as f:
    cfg=yaml.safe_load(f)
width_mult_key='scales'
for k in cfg.get(width_mult_key,{}).values():
    if isinstance(k,list) and len(k)>=2:
        k[1]=round(k[1]*(1-percent),3)
with open(export_yaml,'w') as f:
    yaml.dump(cfg,f, sort_keys=False)
print(f"Pruned model saved to {export_weights}\nNew YAML saved to {export_yaml}")
PY

# -------------------------------------
# 3. Fine-tune pruned model
# -------------------------------------
PRUNED_YAML="$LOG_DIR/yolov11-lcnet-pruned.yaml"
log "Fine-tuning pruned model …"
~/.local/bin/yolo train \
  model="$PRUNED_YAML" \
  data="$DATA_CFG" \
  epochs=$FT_EPOCHS \
  batch=$BATCH \
  imgsz=$IMGSZ \
  resume \
  device=$DEVICE \
  | tee "$LOG_DIR/finetune_train.log"

# Weight path after fine-tune
PRUNED_RUN_DIR=$(ls -td $RUNS_DIR/detect/train* | head -1)
PRUNED_BEST_WEIGHTS="$PRUNED_RUN_DIR/weights/best.pt"
# Export for downstream commands
export PRUNED_BEST_WEIGHTS LOG_DIR

# -------------------------------------
# 4. Final evaluation (mAP, params, FPS)
# -------------------------------------
log "Running final evaluation …"
~/.local/bin/yolo val model="$PRUNED_BEST_WEIGHTS" data="$DATA_CFG" batch=$BATCH imgsz=$IMGSZ device=$DEVICE | tee "$LOG_DIR/final_val.log"

echo "\nParameter count:" | tee -a "$LOG_DIR/summary.txt"
python - <<'PY'
import torch, os
weights=os.environ['PRUNED_BEST_WEIGHTS']
print(weights)
ckpt=torch.load(weights, map_location='cpu')
params=sum(p.numel() for p in ckpt['model'].values())
print(params, file=open(os.path.join(os.environ['LOG_DIR'],'summary.txt'),'a'))
print(params)
PY

log "Benchmarking speed …"
~/.local/bin/yolo benchmark model="$PRUNED_BEST_WEIGHTS" imgsz=$IMGSZ batch=$BATCH | tee "$LOG_DIR/fps.log"

log "Pipeline complete! Logs and artifacts are in $LOG_DIR"