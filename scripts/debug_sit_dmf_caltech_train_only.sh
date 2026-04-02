#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=caltech_sit_dmf_finetune USE_WANDB=False bash scripts/debug_sit_dmf_caltech_train_only.sh <run_label>

Example:
  CONFIG_MODE=caltech_sit_dmf_finetune USE_WANDB=False bash scripts/debug_sit_dmf_caltech_train_only.sh trainonly

This script:
  1) runs a short SiT-DMF Caltech finetuning run without in-training sample/FID generation
  2) disables EMA and debug-image logging
  3) keeps frequent scalar training metrics via log_per_step
EOF
  exit 1
fi

RUN_LABEL="$1"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-False}"
LOG_DIR="${LOG_DIR:-files/logs}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="debug_caltech_SiT_DMF_train_only_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/debug/$JOBNAME"

mkdir -p "$WORKDIR"

cat <<EOF
Training-only debug workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
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
    --config.model.class_dropout_prob=0.1 \
    --config.model.use_auxiliary_v_head=False \
    --config.model.guidance_scale_strategy=fixed \
    --config.training.use_ema=False \
    --config.training.capture_source_from_load=False \
    --config.training.num_epochs=1 \
    --config.training.max_train_steps=50 \
    --config.training.batch_size=16 \
    --config.training.sample_per_step=999999 \
    --config.training.fid_per_step=999999 \
    --config.training.checkpoint_per_epoch=1 \
    --config.training.log_per_step=5 \
    --config.training.debug_log_during_train=False \
    --config.training.debug_num_images=0 \
    --config.training.save_best_fid_only=False \
    --config.logging.use_wandb=${USE_WANDB} \
    2>&1 | tee -a "$WORKDIR/output.log"
