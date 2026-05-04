#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  cat <<'EOF'
Usage: DATASET_NAME=cub200 bash scripts/eval_dogfit_best_fid_taylor.sh [run_dir_or_best_fid_dir]

Examples:
  DATASET_NAME=cub200 PYTHON=/tmp/imeanflow-venv/bin/python \
    CUDA_VISIBLE_DEVICES=0 bash scripts/eval_dogfit_best_fid_taylor.sh

  DATASET_NAME=food101 PYTHON=/tmp/imeanflow-venv/bin/python \
    CUDA_VISIBLE_DEVICES=1 bash scripts/eval_dogfit_best_fid_taylor.sh \
    files/logs/finetuning/food101_SiT_DMF_DogFit_meanflow_taylor_ema_myrun_20260503_162758_adz7gd

This script:
  1) picks the latest Taylor-local EMA DogFit meanflow run for DATASET_NAME if no run path is given
  2) evaluates the saved best_fid checkpoint at 1, 2, and 250 steps
  3) uses the online model only for evaluation
  4) uses /home/ens Taylor dataset and stats paths by default
EOF
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-}"
if [[ -z "${DATASET_NAME}" ]]; then
  echo "ERROR: DATASET_NAME must be set." >&2
  exit 2
fi

RUN_PATH="${1:-}"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_dogfit_meanflow}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-False}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0}}"
EVAL_STEPS="${EVAL_STEPS:-1 2 250}"

DATASET_LABEL=""
DATASET_ROOT=""
FID_CACHE_REF=""
FD_DINO_CACHE_REF=""
DATASET_NUM_CLASSES=""

case "${DATASET_NAME}" in
  caltech|caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    RUN_NAME_PATTERNS=(
      "caltech101_SiT_DMF_DogFit_meanflow_taylor_ema_*"
      "caltech_SiT_DMF_DogFit_meanflow_taylor_ema_*"
    )
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-10}"
    RUN_NAME_PATTERNS=("artbench10_SiT_DMF_DogFit_meanflow_taylor_ema_*")
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-200}"
    RUN_NAME_PATTERNS=("cub200_SiT_DMF_DogFit_meanflow_taylor_ema_*")
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    RUN_NAME_PATTERNS=("food101_SiT_DMF_DogFit_meanflow_taylor_ema_*")
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-196}"
    RUN_NAME_PATTERNS=("stanfordcars_SiT_DMF_DogFit_meanflow_taylor_ema_*")
    ;;
  *)
    echo "ERROR: unknown DATASET_NAME='${DATASET_NAME}'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
    exit 3
    ;;
esac

if [[ -z "${RUN_PATH}" ]]; then
  RUN_PATH="$(
    for pattern in "${RUN_NAME_PATTERNS[@]}"; do
      find files/logs/finetuning -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n'
    done | sort -n | awk '{print $2}' | while IFS= read -r path; do
      if find "$path/best_fid" -maxdepth 1 -type d -name 'checkpoint_*' -print -quit >/dev/null 2>&1; then
        printf '%s\n' "$path"
      fi
    done | tail -n 1
  )"
fi

if [[ -z "${RUN_PATH}" ]]; then
  echo "ERROR: could not find a Taylor EMA DogFit run with a best_fid checkpoint for dataset '${DATASET_LABEL}'." >&2
  exit 4
fi

if [[ ! -d "${RUN_PATH}" ]]; then
  echo "ERROR: run path does not exist: ${RUN_PATH}" >&2
  exit 5
fi

read -r -a EVAL_STEP_ARRAY <<< "${EVAL_STEPS}"

cat <<EOF
DATASET_NAME: ${DATASET_NAME}
DATASET_LABEL: ${DATASET_LABEL}
RUN_PATH: ${RUN_PATH}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
PYTHON: ${PYTHON}
USE_WANDB: ${USE_WANDB}
EVAL_STEPS: ${EVAL_STEPS}
DATASET_ROOT: ${DATASET_ROOT}
DATASET_NUM_CLASSES: ${DATASET_NUM_CLASSES}
FID_CACHE_REF: ${FID_CACHE_REF}
FD_DINO_CACHE_REF: ${FD_DINO_CACHE_REF}
EOF

CONFIG_MODE="${CONFIG_MODE}" \
  PYTHON="${PYTHON}" \
  USE_WANDB="${USE_WANDB}" \
  MODEL_STR="imfSiT_DMF_XL_2" \
  MODEL_USE_DOGFIT="True" \
  TARGET_USE_NULL_CLASS="False" \
  CLASS_DROPOUT_PROB="0.0" \
  DATASET_ROOT="${DATASET_ROOT}" \
  DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES}" \
  FID_CACHE_REF="${FID_CACHE_REF}" \
  FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/eval_best_fid_steps.sh "${RUN_PATH}" "${EVAL_STEP_ARRAY[@]}"
