#!/usr/bin/env bash
#SBATCH --job-name=plain_jit_caltech_h16
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/plain_jit_caltech_h16_%j.out
#SBATCH --error=files/logs/plain_jit_caltech_h16_%j.err
#SBATCH --mail-user=yara.mohammadi-bahram@livia.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /project/6086105/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/home/ymbahram/scratch/datasets/caltech-101_images}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ymbahram/scratch/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ymbahram/scratch/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/home/ymbahram/scratch/weights/JiT-H-16-256.pth}"

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
export RUN_LABEL="${RUN_LABEL:-caltech_h16}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"

export HALF_PRECISION="${HALF_PRECISION:-True}"
export HALF_PRECISION_DTYPE="${HALF_PRECISION_DTYPE:-float16}"
export SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-True}"
export SAMPLING_HALF_PRECISION_DTYPE="${SAMPLING_HALF_PRECISION_DTYPE:-float16}"
export OPTIMIZER="${OPTIMIZER:-adamw}"
export OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"

bash scripts/train_plain_jit_finetune.sh "$RUN_LABEL" \
  --config.model.model_str=flaxJiT_H_16
