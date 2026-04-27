#!/usr/bin/env bash
#SBATCH --job-name=cal_sit_k3_mf
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/cal_sit_k3_mf_%j.out
#SBATCH --error=files/logs/cal_sit_k3_mf_%j.err
#SBATCH --mail-user=yara.mohammadi-bahram@livia.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /project/6086105/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export BACKBONE="${BACKBONE:-sit}"
export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_processed_latents}"
export FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/scratch/ymbahram/weights/SiT-XL-2-256.pt}"

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_sit_finetune}"
export RUN_LABEL="${RUN_LABEL:-caltech_k3_meanflow_4step_fid2500_30k}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_sit_${RUN_LABEL}}"
export LOG_DIR="${LOG_DIR:-files/logs/caltech_plain_sit_power_geometry_4step_fid2500_30k}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"

bash scripts/train_caltech_power_meanflow_k3_4step_2500.sh "$RUN_LABEL" "$@"
