#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_power_meanflow_k3_4step_2500.sh <run_label> [extra main_sit.py args...]

Example:
  USE_WANDB=True bash scripts/train_caltech_power_meanflow_k3_4step_2500.sh caltech_k3_meanflow

This launches Caltech SiT fine-tuning with:
  - experimental power-geometry mean-flow objective
  - k = 3.0
  - 4-step preview sampling every 2500 optimizer steps
  - 4-step FID every 2500 optimizer steps
  - 30k total optimizer steps
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

bash scripts/train_caltech_plain_sit_4step_2500_common.sh "${RUN_LABEL}" power_meanflow 3.0 "$@"
