#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_LABEL="${RUN_LABEL:-caltech_h16_taylor_lr5e_5}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}}"

bash scripts/train_plain_jit_caltech_h16_taylor.sh \
  --config.training.learning_rate=5e-5 \
  "$@"
