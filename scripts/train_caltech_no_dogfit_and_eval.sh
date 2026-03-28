#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=caltech_finetune USE_WANDB=True bash scripts/train_caltech_no_dogfit_and_eval.sh <run_label> [num_steps...]

Example:
  CONFIG_MODE=caltech_finetune USE_WANDB=True bash scripts/train_caltech_no_dogfit_and_eval.sh nodogfit 2 4

This script:
  1) runs fine-tuning on Caltech with dogfit disabled and null-class embedding enabled
  2) after training finishes, evaluates the best checkpoint with the specified num_steps values
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(2 4)
fi
CONFIG_MODE="${CONFIG_MODE:-caltech_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_nodogfit_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"

mkdir -p "$WORKDIR"

cat <<EOF
Training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
Sampling steps: ${STEPS[*]}
EOF

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  $PYTHON \
    main.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:${CONFIG_MODE} \
    --config.model.use_dogfit=False \
    --config.model.target_use_null_class=True \
    --config.model.class_dropout_prob=0.0 \
    --config.training.capture_source_from_load=False \
    --config.logging.use_wandb=${USE_WANDB} \
    2>&1 | tee -a "$WORKDIR/output.log"

# Evaluate the best checkpoint saved under best_fid
CONFIG_MODE=caltech_eval \
  USE_WANDB=${USE_WANDB} \
  MODEL_USE_DOGFIT=False \
  TARGET_USE_NULL_CLASS=True \
  CLASS_DROPOUT_PROB=0.0 \
  bash scripts/eval_best_fid_steps.sh "$WORKDIR" "${STEPS[@]}"
