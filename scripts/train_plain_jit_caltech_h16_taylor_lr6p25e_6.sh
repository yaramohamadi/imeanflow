#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_LABEL="${RUN_LABEL:-caltech_h16_taylor_lr6p25e_6}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}}"

bash scripts/train_plain_jit_caltech_h16_taylor.sh \
  --config.training.learning_rate=6.25e-6 \
  "$@"
