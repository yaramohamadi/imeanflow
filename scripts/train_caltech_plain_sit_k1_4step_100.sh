#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_plain_sit_k1_4step_100.sh <run_label> [extra main_sit.py args...]

Example:
  USE_WANDB=True bash scripts/train_caltech_plain_sit_k1_4step_100.sh caltech_k1

This launches standard Caltech plain SiT fine-tuning with:
  - standard SiT objective
  - 4-step preview sampling every 100 optimizer steps
  - 4-step FID every 100 optimizer steps
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

bash scripts/train_caltech_plain_sit_finetune.sh "${RUN_LABEL}" \
  --config.transport.objective=sit \
  --config.transport.path_power_k=1.0 \
  --config.training.sample_per_step=100 \
  --config.training.fid_per_step=100 \
  --config.training.force_fid_per_step=100 \
  --config.training.force_metric_num_steps=4 \
  --config.training.preview_num_steps='(4,)' \
  "$@"
