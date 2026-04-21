#!/usr/bin/env bash
#SBATCH --job-name=resume_sit_cub200_30k
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/sweeps/plain_sit_dit_plain_4step_best_all_20260420_104955/slurm/resume_sit_cub200_30k_%j.out
#SBATCH --error=files/logs/sweeps/plain_sit_dit_plain_4step_best_all_20260420_104955/slurm/resume_sit_cub200_30k_%j.err
#SBATCH --mail-user=yara.mohammadi-bahram@livia.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /project/6086105/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON=.venv/bin/python
export BACKBONE=sit
export DATASET_NAME=cub200
export DATASET_ROOT=/scratch/ymbahram/datasets/cub-200-2011_processed_latents
export FID_CACHE_REF=/scratch/ymbahram/fid_stats/cub-200-2011_processed-fid_stats.npz
export FD_DINO_CACHE_REF=/scratch/ymbahram/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz
export LOAD_FROM=files/logs/sweeps/plain_sit_dit_plain_4step_best_all_20260420_104955/finetuning/plain_SiT_finetune_cub200_plain_4step_best_all_cub200_sit_20260420_140137_093104/best_fid
export LOG_DIR=files/logs/sweeps/plain_sit_dit_plain_4step_best_all_20260420_104955
export USE_WANDB=True
export WANDB_PROJECT=plain_sit_finetune
export WANDB_NAME=cub200_plain_sit_plain_4step_best_all_resume_27500_to_30000

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

bash scripts/train_plain_sit_finetune.sh plain_4step_best_all_cub200_sit_resume_27500_to_30000 \
  --config.partial_load=False \
  --config.training.max_train_steps=30000
