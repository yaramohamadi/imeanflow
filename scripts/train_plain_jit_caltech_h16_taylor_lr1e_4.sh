#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_LABEL="${RUN_LABEL:-caltech_h16_taylor_lr1e_4}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}}"

bash scripts/train_plain_jit_caltech_h16_taylor.sh \
  --config.training.learning_rate=1e-4 \
  "$@"
