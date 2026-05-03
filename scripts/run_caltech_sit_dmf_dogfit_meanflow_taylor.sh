#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh <run_label> [extra main.py args...]

Examples:
  DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh dogfit_caltech
  DATASET_NAME=cub200 VC_TARGET_SOURCE=ema bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh dogfit_cub
  DATASET_NAME=cub200 VC_TARGET_SOURCE=online bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh dogfit_cub_online

This script:
  1) runs SiT-DMF DogFit meanflow locally on Taylor
  2) lets you choose the latent dataset via DATASET_NAME
  3) ablates whether DogFit's conditioned target v_c comes from the EMA or online model
  4) always trains with stop-gradient on both v_c and v_u in the DogFit path
  5) runs final best_fid evaluation at 1, 2, and 250 steps on the online model only
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_dogfit_meanflow}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0,1}}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
VC_TARGET_SOURCE="${VC_TARGET_SOURCE:-ema}"
PYTHONUNBUFFERED_VALUE="${PYTHONUNBUFFERED:-1}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

DATASET_LABEL=""
DATASET_ROOT=""
FID_CACHE_REF=""
FD_DINO_CACHE_REF=""
DATASET_NUM_CLASSES=""

case "${DATASET_NAME}" in
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-10}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-200}"
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-196}"
    ;;
  *)
    echo "ERROR: unknown DATASET_NAME='${DATASET_NAME}'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
    exit 2
    ;;
esac

case "${VC_TARGET_SOURCE}" in
  ema)
    USE_EMA_VC="True"
    ;;
  online)
    USE_EMA_VC="False"
    ;;
  *)
    echo "ERROR: VC_TARGET_SOURCE must be 'ema' or 'online', got: ${VC_TARGET_SOURCE}" >&2
    exit 3
    ;;
esac

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="${DATASET_LABEL}_SiT_DMF_DogFit_meanflow_taylor_${VC_TARGET_SOURCE}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
USE_WANDB: ${USE_WANDB}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT: ${DATASET_ROOT}
DATASET_NUM_CLASSES: ${DATASET_NUM_CLASSES}
FID_CACHE_REF: ${FID_CACHE_REF}
FD_DINO_CACHE_REF: ${FD_DINO_CACHE_REF}
VC_TARGET_SOURCE: ${VC_TARGET_SOURCE}
USE_EMA_VC: ${USE_EMA_VC}
PYTHONUNBUFFERED: ${PYTHONUNBUFFERED_VALUE}
TRAINING_GUIDANCE_START_STEP: 0
RUN_FINAL_BEST_FID_EVAL: ${RUN_FINAL_BEST_FID_EVAL}
FINAL_EVAL_STEPS: ${FINAL_EVAL_STEPS}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED_VALUE}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" main.py \
    --workdir="${WORKDIR}" \
    --config="configs/load_config.py:${CONFIG_MODE}" \
    --config.dataset.root="${DATASET_ROOT}" \
    --config.dataset.num_classes="${DATASET_NUM_CLASSES}" \
    --config.model.num_classes="${DATASET_NUM_CLASSES}" \
    --config.sampling.num_classes="${DATASET_NUM_CLASSES}" \
    --config.fid.cache_ref="${FID_CACHE_REF}" \
    --config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}" \
    --config.model.training_guidance_start_step="0" \
    --config.model.use_ema_vc="${USE_EMA_VC}" \
    --config.training.fid_use_online_only="True" \
    --config.logging.use_wandb="${USE_WANDB}" \
    --config.logging.wandb_name="${JOBNAME}" \
    --config.logging.wandb_notes="${DATASET_LABEL} SiT-DMF DogFit meanflow fine-tuning (${VC_TARGET_SOURCE} v_c, stopgrad v_c/v_u)" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  echo "=== STARTING FINAL BEST_FID EVAL (${FINAL_EVAL_STEPS}) ON ONLINE MODEL ===" | tee -a "${WORKDIR}/output.log"
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    MODEL_STR="imfSiT_DMF_XL_2" \
    MODEL_USE_DOGFIT="True" \
    TARGET_USE_NULL_CLASS="True" \
    CLASS_DROPOUT_PROB="0.0" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES}" \
    FID_CACHE_REF="${FID_CACHE_REF}" \
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    bash scripts/eval_best_fid_steps.sh "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
  echo "=== FINAL BEST_FID EVAL DONE ===" | tee -a "${WORKDIR}/output.log"
fi
