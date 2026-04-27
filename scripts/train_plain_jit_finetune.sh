#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_WANDB=True bash scripts/train_plain_jit_finetune.sh <run_label> [extra main_jit.py args...]

Example:
  DATASET_NAME=caltech101 PYTHON=.venv/bin/python USE_WANDB=True bash scripts/train_plain_jit_finetune.sh baseline

Optional env vars:
  CONFIG_MODE=plain_jit_finetune
  DATASET_NAME=caltech101            # caltech101, artbench10, cub200, food101, stanfordcars
  DATASET_ROOT=/path/to/root         # pixel ImageFolder root with train/<class>
  FID_CACHE_REF=/path/to/ref.npz
  FD_DINO_CACHE_REF=/path/to/ref.npz
  LOAD_FROM=/path/to/JiT-H-16-256.pth
  SAMPLE_DEVICE_BATCH_SIZE=8       # optional; unset keeps config.fid.sample_device_batch_size
  SAMPLE_FIRST_DEVICE_ONLY=False
  TRAIN_BATCH_SIZE=16              # optional; unset keeps config.training.batch_size
  HALF_PRECISION=True
  HALF_PRECISION_DTYPE=float16
  SAMPLING_HALF_PRECISION=True
  SAMPLING_HALF_PRECISION_DTYPE=float16
  GUIDANCE_SCALE=2.2
  OPTIMIZER=adamw               # official JiT default; lion is available by override
  OPTIMIZER_MU_DTYPE=float16
  RUN_FINAL_BEST_FID_EVAL=True
  FINAL_EVAL_STEPS="1 2 50"
  FINAL_EVAL_USE_WANDB=True
  WANDB_PROJECT=plain_jit_finetune
  WANDB_NAME=caltech101_plain_jit_base
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-plain_jit_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/JiT-H-16-256.pth}"
HALF_PRECISION="${HALF_PRECISION:-False}"
SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-False}"
SAMPLE_FIRST_DEVICE_ONLY="${SAMPLE_FIRST_DEVICE_ONLY:-False}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
DATASET_NAME="${DATASET_NAME:-caltech101}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.2}"
OPTIMIZER="${OPTIMIZER:-adamw}"
OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-$USE_WANDB}"
FINAL_EVAL_EXTRA_ARGS="${FINAL_EVAL_EXTRA_ARGS:-}"
CONFIG_OVERRIDE_ARGS=()

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

WANDB_NAME="${WANDB_NAME:-${DATASET_LABEL}_plain_jit_${RUN_LABEL}_EMA}"
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_name="${WANDB_NAME}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_project="${WANDB_PROJECT}")
CONFIG_OVERRIDE_ARGS+=(--config.logging.wandb_notes="Dataset ${DATASET_LABEL} plain JiT fine-tuning run ${RUN_LABEL} with EMA")

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
if [[ -n "${HALF_PRECISION_DTYPE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.training.half_precision_dtype="${HALF_PRECISION_DTYPE}")
fi
case "${SAMPLING_HALF_PRECISION,,}" in
  1|true|yes|y|on) CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=True) ;;
  0|false|no|n|off) CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision=False) ;;
  *) echo "ERROR: SAMPLING_HALF_PRECISION must be boolean-like, got '$SAMPLING_HALF_PRECISION'." >&2; exit 2 ;;
esac
if [[ -n "${SAMPLING_HALF_PRECISION_DTYPE:-}" ]]; then
  CONFIG_OVERRIDE_ARGS+=(--config.sampling.half_precision_dtype="${SAMPLING_HALF_PRECISION_DTYPE}")
fi
CONFIG_OVERRIDE_ARGS+=(--config.sampling.omega="${GUIDANCE_SCALE}")
CONFIG_OVERRIDE_ARGS+=(--config.sampling.cfg_scale="${GUIDANCE_SCALE}")
CONFIG_OVERRIDE_ARGS+=(--config.training.optimizer="${OPTIMIZER}")
CONFIG_OVERRIDE_ARGS+=(--config.training.optimizer_mu_dtype="${OPTIMIZER_MU_DTYPE}")

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOB_PREFIX="${JOB_PREFIX:-plain_JiT_finetune}_${DATASET_LABEL}"
JOBNAME="${JOB_PREFIX}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${WORKDIR:-$LOG_DIR/finetuning/$JOBNAME}"
mkdir -p "$WORKDIR"

cat <<EOF
JiT JAX training workdir: $WORKDIR
CONFIG_MODE: $CONFIG_MODE
USE_WANDB: $USE_WANDB
RUN_FINAL_BEST_FID_EVAL: $RUN_FINAL_BEST_FID_EVAL
FINAL_EVAL_STEPS: $FINAL_EVAL_STEPS
FINAL_EVAL_USE_WANDB: $FINAL_EVAL_USE_WANDB
FINAL_EVAL_EXTRA_ARGS: ${FINAL_EVAL_EXTRA_ARGS:-<none>}
DATASET_NAME: ${DATASET_NAME}
DATASET_ROOT override: ${DATASET_ROOT}
LOAD_FROM override: ${LOAD_FROM}
SAMPLE_DEVICE_BATCH_SIZE override: ${SAMPLE_DEVICE_BATCH_SIZE:-<config default>}
SAMPLE_FIRST_DEVICE_ONLY override: ${SAMPLE_FIRST_DEVICE_ONLY}
TRAIN_BATCH_SIZE override: ${TRAIN_BATCH_SIZE:-<config default>}
HALF_PRECISION override: ${HALF_PRECISION}
SAMPLING_HALF_PRECISION override: ${SAMPLING_HALF_PRECISION}
GUIDANCE_SCALE override: ${GUIDANCE_SCALE}
OPTIMIZER override: ${OPTIMIZER}
OPTIMIZER_MU_DTYPE override: ${OPTIMIZER_MU_DTYPE}
WANDB_NAME: $WANDB_NAME
WANDB_PROJECT: $WANDB_PROJECT
EOF

set +e
TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
  XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
  PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
  "$PYTHON" \
    main_jit.py \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:"${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${CONFIG_OVERRIDE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  if grep -q "Reached max_train_steps=.* at step" "$WORKDIR/output.log" \
    && grep -q "Fatal Python error: none_dealloc" "$WORKDIR/output.log"; then
    echo "Training reached max_train_steps, then Python hit none_dealloc during shutdown; continuing to final eval."
  else
    exit "$TRAIN_STATUS"
  fi
fi

case "${RUN_FINAL_BEST_FID_EVAL,,}" in
  1|true|yes|y)
    read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
    read -r -a FINAL_EVAL_EXTRA_ARG_ARRAY <<< "$FINAL_EVAL_EXTRA_ARGS"
    echo "Training finished. Evaluating best-FID checkpoint at steps: ${FINAL_EVAL_STEP_ARRAY[*]}"
    CONFIG_MODE="$CONFIG_MODE" \
      PYTHON="$PYTHON" \
      USE_WANDB="$FINAL_EVAL_USE_WANDB" \
      WANDB_NAME_PREFIX="$WANDB_NAME" \
      bash scripts/eval_best_fid_steps_plain_jit.sh "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- "${CONFIG_OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}" "${FINAL_EVAL_EXTRA_ARG_ARRAY[@]}"
    ;;
  0|false|no|n)
    ;;
  *)
    echo "ERROR: RUN_FINAL_BEST_FID_EVAL must be a boolean-like value, got '$RUN_FINAL_BEST_FID_EVAL'." >&2
    exit 2
    ;;
esac
