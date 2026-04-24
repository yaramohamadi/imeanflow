#!/usr/bin/env bash
#SBATCH --job-name=plain_jit_best_eval
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/plain_jit_best_eval_%j.out
#SBATCH --error=files/logs/plain_jit_best_eval_%j.err
#SBATCH --mail-user=yara.mohammadi-bahram@livia.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /home/ymbahram/projects/def-hadi87/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"
export CONFIG_MODE="${CONFIG_MODE:-plain_jit_finetune}"
export USE_WANDB="${USE_WANDB:-True}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-plain_jit_best_fid_eval}"

RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: provide a run dir as the first argument or set RUN_DIR." >&2
  exit 2
fi

read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"

bash scripts/eval_best_fid_steps_plain_jit.sh "$RUN_DIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
  --config.model.model_str=flaxJiT_H_16 \
  --config.dataset.root="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_images}" \
  --config.fid.cache_ref="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}" \
  --config.fd_dino.cache_ref="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}" \
  "$@"
