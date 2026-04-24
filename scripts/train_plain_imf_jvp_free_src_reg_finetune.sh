#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_plain_imf_jvp_free_src_reg_finetune.sh <run_label> [extra main.py args...]

Runs plain iMF fine-tuning with the JVP-free source-regularization objective:
  - trains/selects best checkpoint using 4-step FID by default
  - after training, evaluates best_fid/ with 1, 2, and 250 sampling steps

This accepts the same environment variables and extra config overrides as:
  scripts/train_plain_imf_finetune.sh

Common overrides:
  DATASET_NAME=caltech101
  LOAD_FROM=/path/to/iMF-XL-2-full
  METRIC_NUM_STEPS="4"
  FINAL_EVAL_STEPS="1 2 250"
  WANDB_PROJECT=plain_imf_jvp_free_src_reg_finetune
EOF
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export METRIC_NUM_STEPS="${METRIC_NUM_STEPS:-4}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_jvp_free_src_reg_finetune}"
export JOB_PREFIX="${JOB_PREFIX:-plain_iMF_jvp_free_src_reg_finetune}"
export WANDB_NAME="${WANDB_NAME:-${DATASET_NAME:-caltech101}_plain_imf_jvp_free_src_reg_$1}"

bash scripts/train_plain_imf_finetune.sh "$@" \
  --config.model.training_mode=imf_jvp_free_src_reg \
  --config.sampling.num_steps=4
