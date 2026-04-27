#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_plain_sit_power_meanflow_suite_4step_2500.sh <suite_label> [extra main_sit.py args...]

Example:
  USE_WANDB=True bash scripts/train_caltech_plain_sit_power_meanflow_suite_4step_2500.sh apr26

This runs the full Caltech experiment suite:
  - k=1 plain
  - k=2 plain
  - k=3 plain
  - k=2 meanflow
  - k=3 meanflow

Each run uses:
  - 30k total optimizer steps
  - 4-step preview sampling every 2500 steps
  - 4-step FID every 2500 steps
  - final best-checkpoint evaluation at 1, 2, and 250 steps
EOF
  exit 1
fi

SUITE_LABEL="$1"
shift

bash scripts/train_caltech_plain_sit_k1_4step_2500.sh "${SUITE_LABEL}_k1_plain" "$@"
bash scripts/train_caltech_plain_sit_k2_4step_2500.sh "${SUITE_LABEL}_k2_plain" "$@"
bash scripts/train_caltech_plain_sit_k3_4step_2500.sh "${SUITE_LABEL}_k3_plain" "$@"
bash scripts/train_caltech_power_meanflow_k2_4step_2500.sh "${SUITE_LABEL}_k2_meanflow" "$@"
bash scripts/train_caltech_power_meanflow_k3_4step_2500.sh "${SUITE_LABEL}_k3_meanflow" "$@"
