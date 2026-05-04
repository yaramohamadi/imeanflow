#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/train_plain_imf_dogfit_finetune.sh <run_label> [extra main.py args...]

Examples:
  DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/train_plain_imf_dogfit_finetune.sh dogfit_caltech
  DATASET_NAME=cub200 VC_TARGET_SOURCE=online bash scripts/train_plain_imf_dogfit_finetune.sh dogfit_cub_online

This script:
  1) runs native iMF -> iMF DogFit meanflow fine-tuning
  2) uses the same DogFit stop-gradient treatment as the SiT -> iMF DogFit path
  3) ablates whether conditioned target v_c comes from the EMA or online model
  4) uses EMA decay 0.9998 and trains for 40k steps by default
  5) runs final best_fid evaluation at 1, 2, and 250 steps on the online model only
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-plain_imf_dogfit_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
LOAD_FROM="${LOAD_FROM:-/projets/Ymohammadi/imeanflow/files/weights/iMF-XL-2-full}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
VC_TARGET_SOURCE="${VC_TARGET_SOURCE:-ema}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_dogfit_finetune}"

DATASET_LABEL=""
DATASET_ROOT=""
FID_CACHE_REF=""
FD_DINO_CACHE_REF=""
DATASET_NUM_CLASSES=""

case "${DATASET_NAME}" in
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-10}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-200}"
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-101}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/projets/Ymohammadi/imeanflow/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
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

ensure_dataset_root() {
  if [[ -d "$DATASET_ROOT" ]]; then
    return 0
  fi
  if [[ ! -f "${DATASET_ROOT}.zip" ]]; then
    echo "ERROR: DATASET_ROOT missing and no zip fallback found: $DATASET_ROOT" >&2
    exit 7
  fi

  local extract_lock="${DATASET_ROOT}.extract.lock"
  if mkdir "$extract_lock" 2>/dev/null; then
    echo "Extracting ${DATASET_ROOT}.zip into $DATASET_ROOT"
    local extract_tmp="${DATASET_ROOT}.extracting.$$"
    mkdir -p "$extract_tmp"
    unzip -q "${DATASET_ROOT}.zip" -d "$extract_tmp"
    local extracted_root
    extracted_root="$(find "$extract_tmp" -type d -name "$(basename "$DATASET_ROOT")" | head -n 1)"
    if [[ -z "$extracted_root" ]]; then
      echo "ERROR: could not find $(basename "$DATASET_ROOT") inside ${DATASET_ROOT}.zip" >&2
      rmdir "$extract_lock" 2>/dev/null || true
      exit 7
    fi
    mv "$extracted_root" "$DATASET_ROOT"
    rmdir "$extract_lock" 2>/dev/null || true
  else
    echo "Waiting for another job to finish extracting $DATASET_ROOT"
    while [[ ! -d "$DATASET_ROOT" && -d "$extract_lock" ]]; do
      sleep 30
    done
  fi

  if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: DATASET_ROOT missing after extraction attempt: $DATASET_ROOT" >&2
    exit 7
  fi
}

ensure_cuda_ptxas() {
  if command -v ptxas >/dev/null 2>&1; then
    return 0
  fi

  if command -v module >/dev/null 2>&1; then
    local cuda_module="${CUDA_MODULE:-cuda/12.6}"
    module load "$cuda_module" 2>/dev/null || module load cuda 2>/dev/null || true
  fi

  if ! command -v ptxas >/dev/null 2>&1; then
    echo "WARNING: ptxas was not found. On Compute Canada, run: module load ${CUDA_MODULE:-cuda/12.6}" >&2
  fi
}

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOB_PREFIX="${JOB_PREFIX:-plain_iMF_DogFit_finetune}"
JOBNAME="${JOB_PREFIX}_${DATASET_LABEL}_${VC_TARGET_SOURCE}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"
WANDB_NAME="${WANDB_NAME:-${DATASET_LABEL}_plain_imf_dogfit_${VC_TARGET_SOURCE}_${RUN_LABEL}}"
mkdir -p "$WORKDIR"

cat <<EOF
iMF DogFit training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
RUN_FINAL_BEST_FID_EVAL: $RUN_FINAL_BEST_FID_EVAL
FINAL_EVAL_STEPS: $FINAL_EVAL_STEPS
FINAL_EVAL_USE_WANDB: $FINAL_EVAL_USE_WANDB
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT: ${DATASET_ROOT}
DATASET_NUM_CLASSES: ${DATASET_NUM_CLASSES}
FID_CACHE_REF: ${FID_CACHE_REF}
FD_DINO_CACHE_REF: ${FD_DINO_CACHE_REF}
LOAD_FROM: ${LOAD_FROM}
VC_TARGET_SOURCE: ${VC_TARGET_SOURCE}
USE_EMA_VC: ${USE_EMA_VC}
WANDB_NAME: ${WANDB_NAME}
WANDB_PROJECT: ${WANDB_PROJECT}
EOF

ensure_dataset_root
ensure_cuda_ptxas

CONFIG_OVERRIDE_ARGS=(
  --config.dataset.root="${DATASET_ROOT}"
  --config.dataset.num_classes="${DATASET_NUM_CLASSES}"
  --config.dataset.num_classes_from_data="False"
  --config.model.num_classes="${DATASET_NUM_CLASSES}"
  --config.sampling.num_classes="${DATASET_NUM_CLASSES}"
  --config.fid.cache_ref="${FID_CACHE_REF}"
  --config.load_from="${LOAD_FROM}"
  --config.model.use_ema_vc="${USE_EMA_VC}"
  --config.training.fid_use_online_only="True"
  --config.logging.wandb_name="${WANDB_NAME}"
  --config.logging.wandb_project="${WANDB_PROJECT}"
  --config.logging.wandb_notes="${DATASET_LABEL} native iMF DogFit meanflow fine-tuning (${VC_TARGET_SOURCE} v_c, stopgrad v_c/v_u, ema=0.9998, single-head boundary v)"
)

if [[ -n "${FD_DINO_CACHE_REF:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}")
fi

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  "$PYTHON" \
    main.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:"${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${CONFIG_OVERRIDE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"

case "${RUN_FINAL_BEST_FID_EVAL,,}" in
  1|true|yes|y)
    read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
    echo "Training finished. Evaluating best-FID checkpoint at steps: ${FINAL_EVAL_STEP_ARRAY[*]}"
    CONFIG_MODE="$CONFIG_MODE" \
      PYTHON="$PYTHON" \
      USE_WANDB="$FINAL_EVAL_USE_WANDB" \
      WANDB_NAME_PREFIX="$WANDB_NAME" \
      bash scripts/eval_best_fid_steps_plain_imf.sh "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- "${CONFIG_OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  0|false|no|n)
    ;;
  *)
    echo "ERROR: RUN_FINAL_BEST_FID_EVAL must be a boolean-like value, got '$RUN_FINAL_BEST_FID_EVAL'." >&2
    exit 2
    ;;
esac
