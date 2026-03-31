#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=caltech_finetune USE_WANDB=False bash scripts/debug_sit_caltech_quick.sh <run_label> [num_steps...]

Example:
  CONFIG_MODE=caltech_finetune USE_WANDB=False bash scripts/debug_sit_caltech_quick.sh quicktest 2 4

This script:
  1) runs a short Caltech SiT finetuning run with dogfit disabled
  2) uses frequent sampling, FID, and checkpointing for rapid validation
  3) evaluates the best checkpoint at the specified num_steps values
  4) defaults that post-train evaluation to CPU to avoid GPU checkpoint-restore OOM
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
USE_WANDB="${USE_WANDB:-False}"
LOG_DIR="${LOG_DIR:-files/logs}"
EVAL_PLATFORM="${EVAL_PLATFORM:-cpu}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="debug_caltech_SiT_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/debug/$JOBNAME"

mkdir -p "$WORKDIR"

cat <<EOF
Debug training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
Sampling steps: ${STEPS[*]}
Eval platform: $EVAL_PLATFORM
EOF

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  $PYTHON \
    main.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:${CONFIG_MODE} \
    --config.model.use_dogfit=False \
    --config.model.target_use_null_class=True \
    --config.model.class_dropout_prob=0.0 \
    --config.training.capture_source_from_load=False \
    --config.training.use_ema=False \
    --config.training.num_epochs=1 \
    --config.training.max_train_steps=50 \
    --config.training.batch_size=16 \
    --config.training.sample_per_step=999999 \
    --config.training.fid_per_step=999999 \
    --config.training.checkpoint_per_epoch=1 \
    --config.training.debug_log_during_train=False \
    --config.training.debug_num_images=0 \
    --config.training.save_best_fid_only=True \
    --config.logging.use_wandb=${USE_WANDB} \
    2>&1 | tee -a "$WORKDIR/output.log"

# Evaluate the best checkpoint saved under best_fid
CONFIG_MODE=caltech_finetune \
  USE_WANDB=${USE_WANDB} \
  EVAL_PLATFORM=${EVAL_PLATFORM} \
  MODEL_USE_DOGFIT=False \
  TARGET_USE_NULL_CLASS=True \
  CLASS_DROPOUT_PROB=0.0 \
  bash scripts/eval_best_fid_steps.sh "$WORKDIR" "${STEPS[@]}"
