#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_caltech_dit_finetune.sh <run_label> [extra main_dit.py args...]

Example:
  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True bash scripts/train_caltech_dit_finetune.sh baseline

Optional env vars:
  CONFIG_MODE=caltech_plain_dit_finetune
  DATASET_NAME=caltech101            # caltech101, artbench10, cub200, food101, stanfordcars
  DATASET_ROOT=/path/to/root
  FID_CACHE_REF=/path/to/ref.npz
  FD_DINO_CACHE_REF=/path/to/ref.npz # empty disables FD-DINO
  LOAD_FROM=/path/to/DiT-XL-2-256x256.pt
  SAMPLE_DEVICE_BATCH_SIZE=16
  SAMPLE_LOG_EVERY=1
  HALF_PRECISION=True
  HALF_PRECISION_DTYPE=float16
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_plain_dit_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/DiT-XL-2-256x256.pt}"
HALF_PRECISION="${HALF_PRECISION:-False}"
CONFIG_OVERRIDE_ARGS=()

DATASET_LABEL=""
case "${DATASET_NAME:-}" in
  "")
    ;;
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
    ;;
  *)
    echo "ERROR: unknown DATASET_NAME='$DATASET_NAME'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
    exit 2
    ;;
esac

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
if [[ -n "${HALF_PRECISION:-}" ]]; then
  case "${HALF_PRECISION,,}" in
    1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=True) ;;
    0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=False) ;;
    *) echo "ERROR: HALF_PRECISION must be boolean-like, got '$HALF_PRECISION'." >&2; exit 2 ;;
  esac
fi
if [[ -n "${HALF_PRECISION_DTYPE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision_dtype="${HALF_PRECISION_DTYPE}")
fi

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOB_PREFIX="${JOB_PREFIX:-caltech_plain_DiT_finetune}"
if [[ -n "$DATASET_LABEL" ]]; then
  JOB_PREFIX="${JOB_PREFIX}_${DATASET_LABEL}"
fi
JOBNAME="${JOB_PREFIX}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"
mkdir -p "$WORKDIR"

cat <<EOF
DiT JAX training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
DATASET_NAME: ${DATASET_NAME:-<config default>}
DATASET_ROOT override: ${DATASET_ROOT:-<config default>}
LOAD_FROM override: ${LOAD_FROM:-<config default>}
SAMPLE_DEVICE_BATCH_SIZE override: ${SAMPLE_DEVICE_BATCH_SIZE:-<config default>}
HALF_PRECISION override: ${HALF_PRECISION:-<config default>}
HALF_PRECISION_DTYPE override: ${HALF_PRECISION_DTYPE:-<config default>}
EOF

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  "$PYTHON" \
    main_dit.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:"${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${CONFIG_OVERRIDE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"
