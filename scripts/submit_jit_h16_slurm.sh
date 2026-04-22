#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}"
EXTRA_ARGS=("$@")
SLURM_SCRIPT="${SLURM_SCRIPT:-scripts/train_plain_jit_caltech_h16_slurm.sbatch}"
WEIGHTS="${WEIGHTS:-/home/ymbahram/scratch/weights/JiT-H-16-256.pth}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
RUN_LABEL_SUFFIX="${RUN_LABEL_SUFFIX:-h16}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
DRY_RUN="${DRY_RUN:-False}"

dataset_assets() {
  local dataset="$1"
  case "$dataset" in
    caltech101|caltech-101)
      DATASET_NAME_OUT="caltech101"
      DATASET_ROOT_OUT="/home/ymbahram/scratch/datasets/caltech-101_images"
      FID_CACHE_REF_OUT="/home/ymbahram/scratch/fid_stats/caltech-101-fid_stats.npz"
      FD_DINO_CACHE_REF_OUT="/home/ymbahram/scratch/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"
      ;;
    artbench10|artbench-10)
      DATASET_NAME_OUT="artbench10"
      DATASET_ROOT_OUT="/home/ymbahram/scratch/datasets/artbench-10_images"
      FID_CACHE_REF_OUT="/home/ymbahram/scratch/fid_stats/artbench-10_processed-fid_stats.npz"
      FD_DINO_CACHE_REF_OUT="/home/ymbahram/scratch/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz"
      ;;
    cub200|cub-200|cub-200-2011)
      DATASET_NAME_OUT="cub200"
      DATASET_ROOT_OUT="/home/ymbahram/scratch/datasets/cub-200-2011_images"
      FID_CACHE_REF_OUT="/home/ymbahram/scratch/fid_stats/cub-200-2011_processed-fid_stats.npz"
      FD_DINO_CACHE_REF_OUT="/home/ymbahram/scratch/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz"
      ;;
    food101|food-101)
      DATASET_NAME_OUT="food101"
      DATASET_ROOT_OUT="/home/ymbahram/scratch/datasets/food-101_images"
      FID_CACHE_REF_OUT="/home/ymbahram/scratch/fid_stats/food-101_processed-fid_stats.npz"
      FD_DINO_CACHE_REF_OUT="/home/ymbahram/scratch/fdd_stats/food-101-fd_dino-vitb14_stats.npz"
      ;;
    stanfordcars|stanford-cars|cars)
      DATASET_NAME_OUT="stanfordcars"
      DATASET_ROOT_OUT="/home/ymbahram/scratch/datasets/stanford-cars_images"
      FID_CACHE_REF_OUT="/home/ymbahram/scratch/fid_stats/stanford_cars_processed-fid_stats.npz"
      FD_DINO_CACHE_REF_OUT="/home/ymbahram/scratch/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz"
      ;;
    *)
      echo "ERROR: unknown dataset '$dataset'." >&2
      exit 2
      ;;
  esac
}

check_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 3
  fi
}

check_path "JiT weights" "$WEIGHTS"

for dataset in $DATASETS; do
  dataset_assets "$dataset"
  check_path "$DATASET_NAME_OUT dataset" "$DATASET_ROOT_OUT"
  check_path "$DATASET_NAME_OUT FID stats" "$FID_CACHE_REF_OUT"
  check_path "$DATASET_NAME_OUT FD-DINO stats" "$FD_DINO_CACHE_REF_OUT"

  RUN_LABEL="${DATASET_NAME_OUT}_${RUN_LABEL_SUFFIX}"
  WANDB_NAME="${DATASET_NAME_OUT}_plain_jit_${RUN_LABEL_SUFFIX}"
  JOB_NAME="plain_jit_${DATASET_NAME_OUT}_h16"
  OUT_LOG="files/logs/${JOB_NAME}_%j.out"
  ERR_LOG="files/logs/${JOB_NAME}_%j.err"

  export_args=(
    "ALL"
    "DATASET_NAME=$DATASET_NAME_OUT"
    "DATASET_ROOT=$DATASET_ROOT_OUT"
    "FID_CACHE_REF=$FID_CACHE_REF_OUT"
    "FD_DINO_CACHE_REF=$FD_DINO_CACHE_REF_OUT"
    "LOAD_FROM=$WEIGHTS"
    "USE_WANDB=$USE_WANDB"
    "WANDB_PROJECT=$WANDB_PROJECT"
    "RUN_LABEL=$RUN_LABEL"
    "WANDB_NAME=$WANDB_NAME"
    "RUN_FINAL_BEST_FID_EVAL=$RUN_FINAL_BEST_FID_EVAL"
    "FINAL_EVAL_STEPS=$FINAL_EVAL_STEPS"
  )
  for optional_var in \
    FINAL_EVAL_USE_WANDB \
    HALF_PRECISION \
    HALF_PRECISION_DTYPE \
    SAMPLING_HALF_PRECISION \
    SAMPLING_HALF_PRECISION_DTYPE \
    SAMPLE_DEVICE_BATCH_SIZE \
    SAMPLE_FIRST_DEVICE_ONLY \
    TRAIN_BATCH_SIZE \
    OPTIMIZER \
    OPTIMIZER_MU_DTYPE
  do
    if [[ -v "$optional_var" ]]; then
      export_args+=("${optional_var}=${!optional_var}")
    fi
  done
  export_arg=$(IFS=,; echo "${export_args[*]}")

  cmd=(
    sbatch
    --job-name="$JOB_NAME"
    --output="$OUT_LOG"
    --error="$ERR_LOG"
    --export="$export_arg"
    "$SLURM_SCRIPT"
    "${EXTRA_ARGS[@]}"
  )

  if [[ "${DRY_RUN,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done
