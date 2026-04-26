#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_plain_imf_finetune.sh <run_label> [extra main.py args...]

Example:
  DATASET_NAME=caltech101 CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True bash scripts/train_plain_imf_finetune.sh baseline

Optional env vars:
  CONFIG_MODE=plain_imf_finetune
  DATASET_NAME=caltech101             # caltech101, artbench10, cub200, food101, stanfordcars
  DATASET_ROOT=/path/to/root
  FID_CACHE_REF=/path/to/ref.npz
  FD_DINO_CACHE_REF=/path/to/ref.npz  # empty disables FD-DINO
  LOAD_FROM=/path/to/iMF-XL-2-full
  SAMPLE_DEVICE_BATCH_SIZE=32
  SAMPLE_LOG_EVERY=1
  SAMPLE_FIRST_DEVICE_ONLY=False
  RUN_FINAL_BEST_FID_EVAL=True       # run post-training best-checkpoint eval
  FINAL_EVAL_STEPS="1 2 250"         # final best-checkpoint FID/FD-DINO/IS evals
  FINAL_EVAL_USE_WANDB=False         # final eval defaults to local CSV only
  FORCE_FID_STEPS="5"                # optional: exact space-separated train steps for FID
  FORCE_FID_PER_STEP=5               # optional: ignore config fid_schedule and run FID every N steps
  METRIC_NUM_STEPS="4"               # optional: space-separated sampling step counts for FID
  GUIDANCE_SCALE=7.5
  SAMPLING_T_MIN=0.4
  SAMPLING_T_MAX=0.65
  TRAINING_MODE=imf_jvp              # set imf_split_consistency for SplitMeanFlow
  SPLIT_MIDPOINT_STRATEGY=uniform    # uniform or midpoint
  SPLIT_MIDPOINT_EPS=0.001
  SPLIT_SOURCE_FIRST_PROB=0.0
  SPLIT_SOURCE_SECOND_PROB=0.0
  FID_NUM_SAMPLES=50000
  TRAIN_BATCH_SIZE=4
  GRAD_ACCUM_STEPS=8
  WANDB_PROJECT=plain_imf_finetune
  WANDB_NAME=caltech101_plain_imf_baseline
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-plain_imf_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
LOAD_FROM="${LOAD_FROM:-/scratch/ymbahram/weights/iMF-XL-2-full}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
SAMPLE_FIRST_DEVICE_ONLY="${SAMPLE_FIRST_DEVICE_ONLY:-False}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
SAMPLING_T_MIN="${SAMPLING_T_MIN:-0.4}"
SAMPLING_T_MAX="${SAMPLING_T_MAX:-0.65}"
TRAINING_MODE="${TRAINING_MODE:-imf_jvp}"
SPLIT_MIDPOINT_STRATEGY="${SPLIT_MIDPOINT_STRATEGY:-uniform}"
SPLIT_MIDPOINT_EPS="${SPLIT_MIDPOINT_EPS:-0.001}"
SPLIT_SOURCE_FIRST_PROB="${SPLIT_SOURCE_FIRST_PROB:-0.0}"
SPLIT_SOURCE_SECOND_PROB="${SPLIT_SOURCE_SECOND_PROB:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_finetune}"
CONFIG_OVERRIDE_ARGS=()

DATASET_LABEL=""
case "${DATASET_NAME}" in
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
    ;;
  *)
    echo "ERROR: unknown DATASET_NAME='$DATASET_NAME'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
    exit 2
    ;;
esac

WANDB_NAME="${WANDB_NAME:-${DATASET_LABEL}_plain_imf_${RUN_LABEL}}"
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_name="${WANDB_NAME}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_project="${WANDB_PROJECT}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_notes="Dataset ${DATASET_LABEL} iMF fine-tuning run ${RUN_LABEL}")

CONFIG_OVERRIDE_ARGS+=(--config.dataset.root="${DATASET_ROOT}")
CONFIG_OVERRIDE_ARGS+=(--config.fid.cache_ref="${FID_CACHE_REF}")
CONFIG_OVERRIDE_ARGS+=(--config.load_from="${LOAD_FROM}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.omega="${GUIDANCE_SCALE}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.t_min="${SAMPLING_T_MIN}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.t_max="${SAMPLING_T_MAX}")
CONFIG_OVERRIDE_ARGS+=(--config.model.training_mode="${TRAINING_MODE}")
CONFIG_OVERRIDE_ARGS+=(--config.model.split_consistency_midpoint_strategy="${SPLIT_MIDPOINT_STRATEGY}")
CONFIG_OVERRIDE_ARGS+=(--config.model.split_consistency_midpoint_eps="${SPLIT_MIDPOINT_EPS}")
CONFIG_OVERRIDE_ARGS+=(--config.model.split_consistency_source_first_prob="${SPLIT_SOURCE_FIRST_PROB}")
CONFIG_OVERRIDE_ARGS+=(--config.model.split_consistency_source_second_prob="${SPLIT_SOURCE_SECOND_PROB}")

if [[ -n "${FD_DINO_CACHE_REF+x}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}")
fi
if [[ -n "${SAMPLE_DEVICE_BATCH_SIZE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_device_batch_size="${SAMPLE_DEVICE_BATCH_SIZE}")
fi
if [[ -n "${SAMPLE_LOG_EVERY:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_log_every="${SAMPLE_LOG_EVERY}")
fi
if [[ -n "${FID_NUM_SAMPLES:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.num_samples="${FID_NUM_SAMPLES}")
fi
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.batch_size="${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.grad_accum_steps="${GRAD_ACCUM_STEPS}")
fi
if [[ -n "${FORCE_FID_STEPS:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.force_fid_steps="${FORCE_FID_STEPS}")
fi
if [[ -n "${FORCE_FID_PER_STEP:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.force_fid_per_step="${FORCE_FID_PER_STEP}")
fi
if [[ -n "${METRIC_NUM_STEPS:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.force_metric_num_steps="${METRIC_NUM_STEPS}")
fi
case "${SAMPLE_FIRST_DEVICE_ONLY,,}" in
  1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_first_device_only=True) ;;
  0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_first_device_only=False) ;;
  *) echo "ERROR: SAMPLE_FIRST_DEVICE_ONLY must be boolean-like, got '$SAMPLE_FIRST_DEVICE_ONLY'." >&2; exit 2 ;;
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
JOB_PREFIX="${JOB_PREFIX:-plain_iMF_finetune}"
if [[ -n "$DATASET_LABEL" ]]; then
  JOB_PREFIX="${JOB_PREFIX}_${DATASET_LABEL}"
fi
JOBNAME="${JOB_PREFIX}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"
mkdir -p "$WORKDIR"

cat <<EOF
iMF JAX training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
RUN_FINAL_BEST_FID_EVAL: $RUN_FINAL_BEST_FID_EVAL
FINAL_EVAL_STEPS: $FINAL_EVAL_STEPS
FINAL_EVAL_USE_WANDB: $FINAL_EVAL_USE_WANDB
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT: ${DATASET_ROOT}
LOAD_FROM: ${LOAD_FROM}
GUIDANCE_SCALE: ${GUIDANCE_SCALE}
SAMPLING_T_MIN: ${SAMPLING_T_MIN}
SAMPLING_T_MAX: ${SAMPLING_T_MAX}
SAMPLE_FIRST_DEVICE_ONLY: ${SAMPLE_FIRST_DEVICE_ONLY}
TRAIN_BATCH_SIZE override: ${TRAIN_BATCH_SIZE:-<config default>}
GRAD_ACCUM_STEPS override: ${GRAD_ACCUM_STEPS:-<config default>}
WANDB_NAME: $WANDB_NAME
WANDB_PROJECT: $WANDB_PROJECT
EOF

ensure_dataset_root
ensure_cuda_ptxas

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
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
