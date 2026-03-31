#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=caltech_finetune USE_WANDB=False bash scripts/eval_checkpoint_steps.sh <checkpoint_dir> <num_steps...>

Example:
  CONFIG_MODE=caltech_finetune USE_WANDB=False bash scripts/eval_checkpoint_steps.sh \
    files/logs/debug/debug_caltech_SiT_train_only_trainonly_20260328_183515_bfndsf/checkpoint_50 2 4

This script evaluates a single trained checkpoint with the same config mode used for training.
EOF
  exit 1
fi

CHECKPOINT_DIR="$1"
shift
STEPS=()
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(2 4)
fi
CONFIG_MODE="${CONFIG_MODE:-caltech_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-False}"
LOG_DIR="${LOG_DIR:-files/logs}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 2
fi

for NUM_STEPS in "${STEPS[@]}"; do
  if ! [[ "$NUM_STEPS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid num_steps value: $NUM_STEPS" >&2
    exit 3
  fi

  EVAL_WORKDIR="$LOG_DIR/eval_$(basename "$CHECKPOINT_DIR")_${NUM_STEPS}steps"
  mkdir -p "$EVAL_WORKDIR"

  echo "=== Evaluating checkpoint with num_steps=$NUM_STEPS ==="
  echo "checkpoint: $CHECKPOINT_DIR"
  echo "workdir: $EVAL_WORKDIR"
  echo "config mode: $CONFIG_MODE"

  TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
    XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
    PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
    $PYTHON \
      main.py \
      --workdir="$EVAL_WORKDIR" \
      --config=configs/load_config.py:${CONFIG_MODE} \
      --config.eval_only=True \
      --config.training.use_ema=False \
      --config.model.use_dogfit=False \
      --config.model.target_use_null_class=True \
      --config.model.class_dropout_prob=0.0 \
      --config.load_from="$CHECKPOINT_DIR" \
      --config.sampling.num_steps=${NUM_STEPS} \
      --config.logging.use_wandb=${USE_WANDB} \
      2>&1 | tee -a "$EVAL_WORKDIR/output.log"

  echo "=== finished num_steps=$NUM_STEPS ==="
  echo
done
