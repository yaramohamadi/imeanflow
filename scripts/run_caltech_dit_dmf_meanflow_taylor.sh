#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: DATASET_NAME=caltech101 ENABLE_DOGFIT=True VC_TARGET_SOURCE=ema bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh <run_label> [extra main.py args...]

Examples:
  DATASET_NAME=caltech101 ENABLE_DOGFIT=True VC_TARGET_SOURCE=ema bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh dogfit_caltech
  DATASET_NAME=caltech101 ENABLE_DOGFIT=True VC_TARGET_SOURCE=online bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh dogfit_caltech_online
  DATASET_NAME=caltech101 ENABLE_DOGFIT=False bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh plain_caltech

This script:
  1) runs the DiT_DMF meanflow path locally on Taylor
  2) keeps the SiT_DMF-style single-head encoder/decoder conditioning structure
  3) optionally enables DogFit, or runs a no-DogFit target-side DiT adaptation config
  4) lets DogFit choose whether v_c comes from the EMA or online target model
  5) runs final best_fid evaluation on the online model only
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_dit_dmf_dogfit_meanflow}"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/files/logs}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0,1}}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
ENABLE_DOGFIT="${ENABLE_DOGFIT:-True}"
VC_TARGET_SOURCE="${VC_TARGET_SOURCE:-ema}"
SOURCE_VELOCITY_MAP_MODE="${SOURCE_VELOCITY_MAP_MODE:-transport}"
SOURCE_NATIVE_VELOCITY_DERIVATIVE_MODE="${SOURCE_NATIVE_VELOCITY_DERIVATIVE_MODE:-finite_difference}"
PYTHONUNBUFFERED_VALUE="${PYTHONUNBUFFERED:-1}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 16 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

DATASET_LABEL=""
DATASET_ROOT="${DATASET_ROOT:-}"
FID_CACHE_REF="${FID_CACHE_REF:-}"
FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-}"

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

case "${ENABLE_DOGFIT,,}" in
  1|true|yes|y)
    MODEL_USE_DOGFIT="True"
    TRAIN_USE_EMA="True"
    CAPTURE_SOURCE_FROM_LOAD="True"
    case "${VC_TARGET_SOURCE}" in
      ema)
        USE_EMA_VC="True"
        VC_LABEL="ema"
        ;;
      online)
        USE_EMA_VC="False"
        VC_LABEL="online"
        ;;
      *)
        echo "ERROR: VC_TARGET_SOURCE must be 'ema' or 'online', got: ${VC_TARGET_SOURCE}" >&2
        exit 3
        ;;
    esac
    RUN_FLAVOR="dogfit"
    ;;
  0|false|no|n)
    MODEL_USE_DOGFIT="False"
    TRAIN_USE_EMA="False"
    CAPTURE_SOURCE_FROM_LOAD="False"
    USE_EMA_VC="False"
    VC_LABEL="online"
    RUN_FLAVOR="plain"
    ;;
  *)
    echo "ERROR: ENABLE_DOGFIT must be a boolean-like value, got: ${ENABLE_DOGFIT}" >&2
    exit 4
    ;;
esac

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="${DATASET_LABEL}_DiT_DMF_meanflow_taylor_${RUN_FLAVOR}_${VC_LABEL}_${RUN_LABEL}_${NOW}_${SALT}"
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
ENABLE_DOGFIT: ${MODEL_USE_DOGFIT}
VC_TARGET_SOURCE: ${VC_LABEL}
USE_EMA_VC: ${USE_EMA_VC}
TRAIN_USE_EMA: ${TRAIN_USE_EMA}
SOURCE_VELOCITY_MAP_MODE: ${SOURCE_VELOCITY_MAP_MODE}
SOURCE_NATIVE_VELOCITY_DERIVATIVE_MODE: ${SOURCE_NATIVE_VELOCITY_DERIVATIVE_MODE}
PYTHONUNBUFFERED: ${PYTHONUNBUFFERED_VALUE}
RUN_FINAL_BEST_FID_EVAL: ${RUN_FINAL_BEST_FID_EVAL}
FINAL_EVAL_STEPS: ${FINAL_EVAL_STEPS}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED_VALUE}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" "${REPO_ROOT}/main.py" \
    --workdir="${WORKDIR}" \
    --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
    --config.dataset.root="${DATASET_ROOT}" \
    --config.dataset.num_classes="${DATASET_NUM_CLASSES}" \
    --config.dataset.num_classes_from_data="False" \
    --config.model.num_classes="${DATASET_NUM_CLASSES}" \
    --config.sampling.num_classes="${DATASET_NUM_CLASSES}" \
    --config.fid.cache_ref="${FID_CACHE_REF}" \
    --config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}" \
    --config.model.use_dogfit="${MODEL_USE_DOGFIT}" \
    --config.model.use_ema_vc="${USE_EMA_VC}" \
    --config.model.source_velocity_map_mode="${SOURCE_VELOCITY_MAP_MODE}" \
    --config.model.source_native_velocity_derivative_mode="${SOURCE_NATIVE_VELOCITY_DERIVATIVE_MODE}" \
    --config.training.use_ema="${TRAIN_USE_EMA}" \
    --config.training.capture_source_from_load="${CAPTURE_SOURCE_FROM_LOAD}" \
    --config.training.fid_use_online_only="True" \
    --config.logging.use_wandb="${USE_WANDB}" \
    --config.logging.wandb_name="${JOBNAME}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  echo "=== STARTING FINAL BEST_FID EVAL (${FINAL_EVAL_STEPS}) ON ONLINE MODEL ===" | tee -a "${WORKDIR}/output.log"
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    MODEL_STR="imfDiT_DMF_XL_2" \
    MODEL_USE_DOGFIT="${MODEL_USE_DOGFIT}" \
    TARGET_USE_NULL_CLASS="False" \
    CLASS_DROPOUT_PROB="0.0" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES}" \
    FID_CACHE_REF="${FID_CACHE_REF}" \
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    bash "${SCRIPT_DIR}/eval_best_fid_steps.sh" "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
  echo "=== FINAL BEST_FID EVAL DONE ===" | tee -a "${WORKDIR}/output.log"
fi
