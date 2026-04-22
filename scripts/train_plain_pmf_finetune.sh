#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_plain_pmf_finetune.sh <run_label> [extra main_pmf.py args...]

Optional env vars:
  CONFIG_MODE=plain_pmf_finetune
  DATASET_NAME=caltech101            # caltech101, artbench10, cub200, food101, stanfordcars
  DATASET_ROOT=/path/to/root         # pixel ImageFolder root with train/<class>
  FID_CACHE_REF=/path/to/ref.npz
  FD_DINO_CACHE_REF=/path/to/ref.npz
  LOAD_FROM=/path/to/pMF-H-16-full
  LOAD_ZIP=/path/to/pMF-H-16-full.zip
  TRAIN_BATCH_SIZE=16
  SAMPLE_DEVICE_BATCH_SIZE=4
  SAMPLE_FIRST_DEVICE_ONLY=False
  HALF_PRECISION=False
  SAMPLING_HALF_PRECISION=True
  GUIDANCE_SCALE=7.0
  OPTIMIZER=muon
  OPTIMIZER_MU_DTYPE=float16
  PMF_LPIPS=True
  PMF_CONVNEXT=True
  WANDB_PROJECT=plain_pmf_finetune
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-plain_pmf_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
export HF_HOME="${HF_HOME:-/home/ens/AT74470/imeanflow/files/hf_cache}"
LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full}"
LOAD_ZIP="${LOAD_ZIP:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full.zip}"
HALF_PRECISION="${HALF_PRECISION:-False}"
HALF_PRECISION_DTYPE="${HALF_PRECISION_DTYPE:-float16}"
SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-True}"
SAMPLING_HALF_PRECISION_DTYPE="${SAMPLING_HALF_PRECISION_DTYPE:-float16}"
SAMPLE_FIRST_DEVICE_ONLY="${SAMPLE_FIRST_DEVICE_ONLY:-False}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_pmf_finetune}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
OPTIMIZER="${OPTIMIZER:-muon}"
OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"
PMF_LPIPS="${PMF_LPIPS:-True}"
PMF_CONVNEXT="${PMF_CONVNEXT:-True}"
CONFIG_OVERRIDE_ARGS=()

if [[ ! -d "$LOAD_FROM" ]]; then
  if [[ -f "$LOAD_ZIP" ]]; then
    echo "Extracting pMF checkpoint zip to $(dirname "$LOAD_FROM") ..."
    unzip -q "$LOAD_ZIP" -d "$(dirname "$LOAD_FROM")"
  else
    echo "ERROR: checkpoint directory missing and zip not found: $LOAD_FROM / $LOAD_ZIP" >&2
    exit 2
  fi
fi

DATASET_LABEL=""
case "${DATASET_NAME}" in
  caltech101|caltech-101)
    DATASET_LABEL="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_images}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    ;;
  artbench10|artbench-10)
    DATASET_LABEL="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/artbench-10_images}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_LABEL="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/cub-200-2011_images}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    ;;
  food101|food-101)
    DATASET_LABEL="food101"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/food-101_images}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_LABEL="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/stanford-cars_images}"
    FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
    ;;
  *)
    echo "ERROR: unknown DATASET_NAME='$DATASET_NAME'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
    exit 2
    ;;
esac

WANDB_NAME="${WANDB_NAME:-${DATASET_LABEL}_plain_pmf_${RUN_LABEL}}"
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_name="${WANDB_NAME}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_project="${WANDB_PROJECT}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_notes="Dataset ${DATASET_LABEL} plain pMF fine-tuning run ${RUN_LABEL}")
CONFIG_OVERRIDE_ARGS+=(--config.dataset.root="${DATASET_ROOT}")
CONFIG_OVERRIDE_ARGS+=(--config.fid.cache_ref="${FID_CACHE_REF}")
CONFIG_OVERRIDE_ARGS+=(--config.fd_dino.cache_ref="${FD_DINO_CACHE_REF}")
CONFIG_OVERRIDE_ARGS+=(--config.load_from="${LOAD_FROM}")

if [[ -n "$SAMPLE_DEVICE_BATCH_SIZE" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_device_batch_size="${SAMPLE_DEVICE_BATCH_SIZE}")
fi
if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
  if [[ "$TRAIN_BATCH_SIZE" = "auto" ]]; then
    TRAIN_BATCH_SIZE="$("$PYTHON" - <<'PY'
import jax
print(jax.local_device_count())
PY
)"
  fi
  CONFIG_OVERRIDE_ARGS+=(--config.training.batch_size="${TRAIN_BATCH_SIZE}")
fi

case "${SAMPLE_FIRST_DEVICE_ONLY,,}" in
  1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_first_device_only=True) ;;
  0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.fid.sample_first_device_only=False) ;;
  *) echo "ERROR: SAMPLE_FIRST_DEVICE_ONLY must be boolean-like, got '$SAMPLE_FIRST_DEVICE_ONLY'." >&2; exit 2 ;;
esac
case "${HALF_PRECISION,,}" in
  1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=True) ;;
  0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision=False) ;;
  *) echo "ERROR: HALF_PRECISION must be boolean-like, got '$HALF_PRECISION'." >&2; exit 2 ;;
esac
CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision_dtype="${HALF_PRECISION_DTYPE}")
case "${SAMPLING_HALF_PRECISION,,}" in
  1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=True) ;;
  0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=False) ;;
  *) echo "ERROR: SAMPLING_HALF_PRECISION must be boolean-like, got '$SAMPLING_HALF_PRECISION'." >&2; exit 2 ;;
esac
CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision_dtype="${SAMPLING_HALF_PRECISION_DTYPE}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.omega="${GUIDANCE_SCALE}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.cfg_scale="${GUIDANCE_SCALE}")
CONFIG_OVERRIDE_ARGS+=(--config.training.optimizer="${OPTIMIZER}")
CONFIG_OVERRIDE_ARGS+=(--config.training.optimizer_mu_dtype="${OPTIMIZER_MU_DTYPE}")
CONFIG_OVERRIDE_ARGS+=(--config.model.lpips="${PMF_LPIPS}")
CONFIG_OVERRIDE_ARGS+=(--config.model.convnext="${PMF_CONVNEXT}")

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOB_PREFIX="${JOB_PREFIX:-plain_pMF_finetune}_${DATASET_LABEL}"
JOBNAME="${JOB_PREFIX}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="$LOG_DIR/finetuning/$JOBNAME"
mkdir -p "$WORKDIR"

cat <<EOF
pMF JAX training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT override: ${DATASET_ROOT}
LOAD_FROM override: ${LOAD_FROM}
HF_HOME: ${HF_HOME}
SAMPLE_DEVICE_BATCH_SIZE override: ${SAMPLE_DEVICE_BATCH_SIZE:-<config default>}
SAMPLE_FIRST_DEVICE_ONLY override: ${SAMPLE_FIRST_DEVICE_ONLY}
TRAIN_BATCH_SIZE override: ${TRAIN_BATCH_SIZE:-<config default>}
HALF_PRECISION override: ${HALF_PRECISION}
SAMPLING_HALF_PRECISION override: ${SAMPLING_HALF_PRECISION}
GUIDANCE_SCALE override: ${GUIDANCE_SCALE}
OPTIMIZER override: ${OPTIMIZER}
OPTIMIZER_MU_DTYPE override: ${OPTIMIZER_MU_DTYPE}
PMF_LPIPS override: ${PMF_LPIPS}
PMF_CONVNEXT override: ${PMF_CONVNEXT}
WANDB_NAME: $WANDB_NAME
WANDB_PROJECT: $WANDB_PROJECT
EOF

TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  "$PYTHON" \
    main_pmf.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:"${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${CONFIG_OVERRIDE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"
