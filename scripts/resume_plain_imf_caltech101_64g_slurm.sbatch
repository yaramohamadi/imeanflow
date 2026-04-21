#!/usr/bin/env bash
#SBATCH --job-name=resume_imf_caltech101_30k
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923/slurm/resume_imf_caltech101_30k_%j.out
#SBATCH --error=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923/slurm/resume_imf_caltech101_30k_%j.err
#SBATCH --mail-user=yara.mohammadi-bahram@livia.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /project/6086105/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON=.venv/bin/python
export BACKBONE=imf
export DATASET_NAME=caltech101
export DATASET_ROOT=/scratch/ymbahram/datasets/caltech-101_processed_latents
export FID_CACHE_REF=/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz
export FD_DINO_CACHE_REF=/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz
export LOAD_FROM=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923/finetuning/plain_iMF_finetune_caltech101_plain_4step_best_imf_caltech101_imf_20260420_180555_vf6kzl/latest_eval
export LOG_DIR=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923

# Keep W&B on for the resumed run; override with USE_WANDB=False only for debugging.
export USE_WANDB="${USE_WANDB:-True}"
export FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-${USE_WANDB}}"
export WANDB_PROJECT=plain_imf_finetune
export WANDB_NAME=caltech101_plain_imf_plain_4step_best_imf_resume_15000_to_30000

if [[ ! -d "$DATASET_ROOT" && -f "${DATASET_ROOT}.zip" ]]; then
  EXTRACT_LOCK="${DATASET_ROOT}.extract.lock"
  if mkdir "$EXTRACT_LOCK" 2>/dev/null; then
    echo "Extracting ${DATASET_ROOT}.zip into $DATASET_ROOT"
    EXTRACT_TMP="${DATASET_ROOT}.extracting.$$"
    mkdir -p "$EXTRACT_TMP"
    unzip -q "${DATASET_ROOT}.zip" -d "$EXTRACT_TMP"
    EXTRACTED_ROOT="$(find "$EXTRACT_TMP" -type d -name "$(basename "$DATASET_ROOT")" | head -n 1)"
    if [[ -z "$EXTRACTED_ROOT" ]]; then
      echo "ERROR: could not find $(basename "$DATASET_ROOT") inside ${DATASET_ROOT}.zip" >&2
      rmdir "$EXTRACT_LOCK" 2>/dev/null || true
      exit 7
    fi
    mv "$EXTRACTED_ROOT" "$DATASET_ROOT"
    rmdir "$EXTRACT_LOCK" 2>/dev/null || true
  else
    echo "Waiting for another job to finish extracting $DATASET_ROOT"
    while [[ ! -d "$DATASET_ROOT" && -d "$EXTRACT_LOCK" ]]; do sleep 30; done
  fi
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: DATASET_ROOT missing after extraction attempt: $DATASET_ROOT" >&2
  exit 7
fi

bash scripts/train_plain_imf_finetune.sh plain_4step_best_imf_caltech101_imf_resume_15000_to_30000 \
  --config.partial_load=False \
  --config.training.max_train_steps=30000
