#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: BACKBONE=sit|dit DATASET_NAME=<name> bash scripts/run_dmf_no_dogfit_taylor.sh <run_label> [extra main.py args...]

Examples:
  BACKBONE=sit DATASET_NAME=caltech101 bash scripts/run_dmf_no_dogfit_taylor.sh sit_plain
  BACKBONE=dit DATASET_NAME=cub200 bash scripts/run_dmf_no_dogfit_taylor.sh dit_plain

This script:
  1) runs the no-DogFit DMF path locally on Taylor
  2) supports either SiT_DMF or DiT_DMF through BACKBONE=sit|dit
  3) initializes from the corresponding pretrained checkpoint with partial_load=True
  4) trains the DMF target directly on the meanflow objective
  5) for BACKBONE=dit, does not use any DogFit source, epsilon-to-velocity wrapper, or source mapping
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

BACKBONE="${BACKBONE:-sit}"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-dmf_no_dogfit}"
LOG_DIR="${LOG_DIR:-files/logs}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0,1}}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
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

case "${BACKBONE,,}" in
  sit)
    MODEL_STR="imfSiT_DMF_XL_2"
    LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/SiT-XL-2-256.pt}"
    BACKBONE_LABEL="SiT_DMF"
    BACKBONE_TAG="sit-dmf"
    ;;
  dit)
    MODEL_STR="imfDiT_DMF_XL_2"
    LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/DiT-XL-2-256x256.pt}"
    BACKBONE_LABEL="DiT_DMF"
    BACKBONE_TAG="dit-dmf"
    ;;
  *)
    echo "ERROR: BACKBONE must be 'sit' or 'dit', got: ${BACKBONE}" >&2
    exit 3
    ;;
esac

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="${DATASET_LABEL}_${BACKBONE_LABEL}_plain_meanflow_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
BACKBONE: ${BACKBONE}
MODEL_STR: ${MODEL_STR}
LOAD_FROM: ${LOAD_FROM}
USE_WANDB: ${USE_WANDB}
WANDB_PROJECT: ${WANDB_PROJECT}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT: ${DATASET_ROOT}
DATASET_NUM_CLASSES: ${DATASET_NUM_CLASSES}
FID_CACHE_REF: ${FID_CACHE_REF}
FD_DINO_CACHE_REF: ${FD_DINO_CACHE_REF}
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
  "${PYTHON}" main.py \
    --workdir="${WORKDIR}" \
    --config="configs/load_config.py:${CONFIG_MODE}" \
    --config.dataset.root="${DATASET_ROOT}" \
    --config.dataset.num_classes="${DATASET_NUM_CLASSES}" \
    --config.dataset.num_classes_from_data="False" \
    --config.model.model_str="${MODEL_STR}" \
    --config.model.num_classes="${DATASET_NUM_CLASSES}" \
    --config.model.use_dogfit="False" \
    --config.model.source_model_str="" \
    --config.model.source_num_classes="${DATASET_NUM_CLASSES}" \
    --config.model.source_prediction_space="v" \
    --config.model.use_auxiliary_v_head="False" \
    --config.model.target_use_null_class="True" \
    --config.model.class_dropout_prob="0.1" \
    --config.model.use_ema_vc="False" \
    --config.sampling.num_classes="${DATASET_NUM_CLASSES}" \
    --config.fid.cache_ref="${FID_CACHE_REF}" \
    --config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}" \
    --config.training.use_ema="False" \
    --config.training.capture_source_from_load="False" \
    --config.training.fid_use_online_only="True" \
    --config.load_from="${LOAD_FROM}" \
    --config.partial_load="True" \
    --config.logging.use_wandb="${USE_WANDB}" \
    --config.logging.wandb_project="${WANDB_PROJECT}" \
    --config.logging.wandb_name="${JOBNAME}" \
    --config.logging.wandb_notes="${DATASET_LABEL} ${BACKBONE_LABEL} plain meanflow without DogFit; direct DMF target training from pretrained ${BACKBONE} initialization" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  echo "=== STARTING FINAL BEST_FID EVAL (${FINAL_EVAL_STEPS}) ON ONLINE MODEL ===" | tee -a "${WORKDIR}/output.log"
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    MODEL_STR="${MODEL_STR}" \
    MODEL_USE_DOGFIT="False" \
    TARGET_USE_NULL_CLASS="True" \
    CLASS_DROPOUT_PROB="0.1" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES}" \
    FID_CACHE_REF="${FID_CACHE_REF}" \
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    bash scripts/eval_best_fid_steps.sh "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
  echo "=== FINAL BEST_FID EVAL DONE ===" | tee -a "${WORKDIR}/output.log"
fi
