#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_plain_sit_4step_2500_common.sh <run_label> <objective> <path_power_k> [extra main_sit.py args...]

Example:
  USE_WANDB=True bash scripts/train_caltech_plain_sit_4step_2500_common.sh caltech_k2_plain sit 2.0

This launches a Caltech plain-SiT run with:
  - preview sampling every 2500 optimizer steps
  - 4-step FID every 2500 optimizer steps
  - 30k total optimizer steps
  - final best-checkpoint evaluation at 1, 2, and 250 steps
EOF
  exit 1
fi

RUN_LABEL="$1"
OBJECTIVE="$2"
PATH_POWER_K="$3"
shift 3

bash scripts/train_caltech_plain_sit_finetune.sh "${RUN_LABEL}" \
  --config.transport.objective="${OBJECTIVE}" \
  --config.transport.path_power_k="${PATH_POWER_K}" \
  --config.training.max_train_steps=30000 \
  --config.training.sample_per_step=2500 \
  --config.training.fid_per_step=2500 \
  --config.training.force_fid_per_step=2500 \
  --config.training.force_metric_num_steps=4 \
  --config.training.preview_num_steps='(4,)' \
  "$@"
