#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_plain_sit_finetune.sh <run_label> [extra main_sit.py args...]

Example:
  DATASET_NAME=food101 USE_WANDB=True bash scripts/train_plain_sit_finetune.sh baseline

This script:
  1) launches the dataset-agnostic plain-SiT finetuning config
  2) keeps the config-defined official transport settings
  3) lets you append extra `main_sit.py` config overrides if needed
  4) by default, selects best_fid/ with 4-step FID during training
  5) runs final best-checkpoint eval at 1, 2, and 250 steps after training

Optional env vars:
  RUN_FINAL_BEST_FID_EVAL=True       # run post-training best-checkpoint eval
  FINAL_EVAL_STEPS="1 2 250"         # override the final evaluation step list
  FINAL_EVAL_USE_WANDB=True          # defaults to USE_WANDB; set False to disable final eval wandb
  SAMPLE_DEVICE_BATCH_SIZE=16        # optional override for per-GPU sampling/decode batch size
  SAMPLE_LOG_EVERY=10                # optional override for sampling timing log frequency
  HALF_PRECISION=True                # optional override for training/model half precision
  HALF_PRECISION_DTYPE=float16       # optional training/model dtype; bf16 fails on this environment's PTX path
  SAMPLING_HALF_PRECISION=True       # optional override for preview/FID sampling half precision
  SAMPLING_HALF_PRECISION_DTYPE=float16
  DATASET_NAME=caltech101            # named latent dataset: caltech101, artbench10, cub200, food101, stanfordcars
  DATASET_ROOT=/path/to/root         # optional explicit latent dataset root, overrides DATASET_NAME root
  FID_CACHE_REF=/path/to/ref.npz     # optional explicit FID stats ref
  FD_DINO_CACHE_REF=/path/to/ref.npz # optional explicit FD-DINO stats ref; empty disables FD-DINO
  FORCE_FID_PER_STEP=5               # optional: ignore config fid_schedule and run FID every N steps
  METRIC_NUM_STEPS="1"               # optional: space-separated sampling step counts for FID
  BACKBONE=sit                       # sit or dit; selects model_str and default checkpoint
  MODEL_STR=flaxDiT_XL_2             # optional explicit Flax backbone override
  LOAD_FROM=/path/to/checkpoint      # optional initial checkpoint override
  WANDB_PROJECT=plain_sit_finetune   # optional wandb project override
  WANDB_NAME=food101_plain_sit_base  # optional wandb run name override
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-plain_sit_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-${USE_WANDB}}"
HALF_PRECISION="${HALF_PRECISION:-False}"
SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-False}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_sit_finetune}"
CONFIG_OVERRIDE_ARGS=()

case "${BACKBONE:-}" in
  "")
    ;;
  sit|SiT)
    MODEL_STR="${MODEL_STR:-flaxSiT_XL_2}"
    LOAD_FROM="${LOAD_FROM:-/scratch/ymbahram/weights/SiT-XL-2-256.pt}"
    ;;
  dit|DiT)
    MODEL_STR="${MODEL_STR:-flaxDiT_XL_2}"
    LOAD_FROM="${LOAD_FROM:-/scratch/ymbahram/weights/DiT-XL-2-256x256.pt}"
    ;;
  *)
    echo "ERROR: unknown BACKBONE='$BACKBONE'. Known: sit, dit." >&2
    exit 2
    ;;
esac

if [[ -n "${MODEL_STR:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.model.model_str="${MODEL_STR}")
fi

DATASET_LABEL=""
case "${DATASET_NAME}" in
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}"
    if [[ ! -v FD_DINO_CACHE_REF ]]; then
      FD_DINO_CACHE_REF="/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"
    fi
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/artbench-10_processed-fid_stats.npz}"
    if [[ ! -v FD_DINO_CACHE_REF ]]; then
      FD_DINO_CACHE_REF="/scratch/ymbahram/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz"
    fi
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

WANDB_NAME="${WANDB_NAME:-${DATASET_LABEL}_plain_sit_${RUN_LABEL}}"
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_name="${WANDB_NAME}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_project="${WANDB_PROJECT}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_notes="Dataset ${DATASET_LABEL} plain SiT fine-tuning run ${RUN_LABEL}")

if [[ -n "${DATASET_ROOT:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.dataset.root="${DATASET_ROOT}")
fi
if [[ -n "${FID_CACHE_REF:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.cache_ref="${FID_CACHE_REF}")
fi
if [[ -n "${FD_DINO_CACHE_REF+x}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}")
fi
if [[ -n "${LOAD_FROM:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.load_from="${LOAD_FROM}")
fi

if [[ -n "${SAMPLE_DEVICE_BATCH_SIZE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_device_batch_size="${SAMPLE_DEVICE_BATCH_SIZE}")
fi
if [[ -n "${SAMPLE_LOG_EVERY:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_log_every="${SAMPLE_LOG_EVERY}")
fi

if [[ -n "${FORCE_FID_PER_STEP:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.force_fid_per_step="${FORCE_FID_PER_STEP}")
fi

if [[ -n "${METRIC_NUM_STEPS:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.force_metric_num_steps="${METRIC_NUM_STEPS}")
fi

if [[ -n "${HALF_PRECISION:-}" ]]; then
  case "${HALF_PRECISION,,}" in
    1|true|yes|y|on)
      CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=True)
      ;;
    0|false|no|n|off)
      CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=False)
      ;;
    *)
      echo "ERROR: HALF_PRECISION must be a boolean-like value, got '$HALF_PRECISION'." >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${HALF_PRECISION_DTYPE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision_dtype="${HALF_PRECISION_DTYPE}")
fi

if [[ -n "${SAMPLING_HALF_PRECISION:-}" ]]; then
  case "${SAMPLING_HALF_PRECISION,,}" in
    1|true|yes|y|on)
      CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=True)
      ;;
    0|false|no|n|off)
      CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=False)
      ;;
    *)
      echo "ERROR: SAMPLING_HALF_PRECISION must be a boolean-like value, got '$SAMPLING_HALF_PRECISION'." >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${SAMPLING_HALF_PRECISION_DTYPE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision_dtype="${SAMPLING_HALF_PRECISION_DTYPE}")
fi

case "${HALF_PRECISION_DTYPE:-}" in
  bf16|bfloat16)
    echo "WARNING: bf16 failed on this environment with a PTX ISA error; use float16 unless the CUDA/JAX toolchain is fixed." >&2
    ;;
esac
case "${SAMPLING_HALF_PRECISION_DTYPE:-}" in
  bf16|bfloat16)
    echo "WARNING: sampling bf16 failed on this environment with a PTX ISA error; use float16 unless the CUDA/JAX toolchain is fixed." >&2
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

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOB_PREFIX="${JOB_PREFIX:-plain_SiT_finetune}"
if [[ -n "$DATASET_LABEL" ]]; then
  JOB_PREFIX="${JOB_PREFIX}_${DATASET_LABEL}"
fi
JOBNAME="${JOB_PREFIX}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"

mkdir -p "$WORKDIR"

cat <<EOF
Training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
RUN_FINAL_BEST_FID_EVAL: $RUN_FINAL_BEST_FID_EVAL
FINAL_EVAL_STEPS: $FINAL_EVAL_STEPS
FINAL_EVAL_USE_WANDB: $FINAL_EVAL_USE_WANDB
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT override: ${DATASET_ROOT:-<config default>}
FID_CACHE_REF override: ${FID_CACHE_REF:-<config default>}
FD_DINO_CACHE_REF override: ${FD_DINO_CACHE_REF:-<config default>}
LOAD_FROM override: ${LOAD_FROM:-<config default>}
BACKBONE: ${BACKBONE:-<config default>}
MODEL_STR override: ${MODEL_STR:-<config default>}
SAMPLE_DEVICE_BATCH_SIZE override: ${SAMPLE_DEVICE_BATCH_SIZE:-<config default>}
SAMPLE_LOG_EVERY override: ${SAMPLE_LOG_EVERY:-<config default>}
HALF_PRECISION override: ${HALF_PRECISION:-<config default>}
HALF_PRECISION_DTYPE override: ${HALF_PRECISION_DTYPE:-<config default>}
SAMPLING_HALF_PRECISION override: ${SAMPLING_HALF_PRECISION:-<config default>}
SAMPLING_HALF_PRECISION_DTYPE override: ${SAMPLING_HALF_PRECISION_DTYPE:-<config default>}
WANDB_NAME: $WANDB_NAME
WANDB_PROJECT: $WANDB_PROJECT
EOF

ensure_dataset_root

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  $PYTHON \
    main_sit.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:${CONFIG_MODE} \
    --config.logging.use_wandb=${USE_WANDB} \
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
      bash scripts/eval_best_fid_steps_plain_sit.sh "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- "${CONFIG_OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  0|false|no|n)
    ;;
  *)
    echo "ERROR: RUN_FINAL_BEST_FID_EVAL must be a boolean-like value, got '$RUN_FINAL_BEST_FID_EVAL'." >&2
    exit 2
    ;;
esac
