#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_sit_finetune.sh <run_label> [extra main.py args...]

Example:
  USE_WANDB=True bash scripts/train_caltech_sit_finetune.sh baseline

This script:
  1) launches the default Caltech SiT finetuning config
  2) keeps the config-defined DogFit and null-class settings
  3) lets you append extra `main.py` config overrides if needed
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_SiT_finetune_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"

mkdir -p "$WORKDIR"

cat <<EOF
Training workdir: $WORKDIR
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
    --config.logging.use_wandb=${USE_WANDB} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"
