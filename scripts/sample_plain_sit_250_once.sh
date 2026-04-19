#!/usr/bin/env bash
set -euo pipefail

CONFIG_MODE="${CONFIG_MODE:-plain_sit_finetune}"
PYTHON="${PYTHON:-python3}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
NUM_STEPS="${NUM_STEPS:-250}"
SAMPLE_LOG_EVERY="${SAMPLE_LOG_EVERY:-1}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-64}"
HALF_PRECISION="${HALF_PRECISION:-False}"
HALF_PRECISION_DTYPE="${HALF_PRECISION_DTYPE:-float16}"
LOG_DIR="${LOG_DIR:-files/logs}"
WORKDIR="${WORKDIR:-files/debug/plain_sit_250_once}"

CHECKPOINT_OR_RUN_DIR="${1:-}"
if [[ -n "$CHECKPOINT_OR_RUN_DIR" ]]; then
  shift
else
  if [[ -d "$LOG_DIR/finetuning" ]]; then
    CHECKPOINT_OR_RUN_DIR="$(find "$LOG_DIR/finetuning" -maxdepth 1 -type d -name 'plain_SiT_finetune_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  fi
fi

if [[ -z "$CHECKPOINT_OR_RUN_DIR" ]]; then
  echo "ERROR: pass a checkpoint/run directory, or set LOG_DIR to a directory containing finetuning runs." >&2
  exit 2
fi

EXTRA_ARGS=()
case "${HALF_PRECISION,,}" in
  1|true|yes|y|on)
    case "${HALF_PRECISION_DTYPE,,}" in
      bf16|bfloat16)
        echo "WARNING: bf16 failed on this environment with a PTX ISA error; use float16 unless your CUDA/JAX toolchain supports bf16." >&2
        ;;
    esac
    EXTRA_ARGS+=(--half-precision --half-precision-dtype "$HALF_PRECISION_DTYPE")
    ;;
  0|false|no|n|off)
    ;;
  *)
    echo "ERROR: HALF_PRECISION must be boolean-like, got '$HALF_PRECISION'." >&2
    exit 3
    ;;
esac

echo "checkpoint/run dir: $CHECKPOINT_OR_RUN_DIR"
echo "config mode: $CONFIG_MODE"
echo "device batch size: $SAMPLE_DEVICE_BATCH_SIZE"
echo "num samples: $NUM_SAMPLES"
echo "num steps: $NUM_STEPS"
echo "sample log every: $SAMPLE_LOG_EVERY"
echo "preview samples: $PREVIEW_SAMPLES"
echo "half precision: $HALF_PRECISION"
echo "half precision dtype: $HALF_PRECISION_DTYPE"
echo "workdir: $WORKDIR"

TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "$PYTHON" scripts/sample_plain_sit_250_once.py \
    "$CHECKPOINT_OR_RUN_DIR" \
    --config-mode "$CONFIG_MODE" \
    --workdir "$WORKDIR" \
    --device-batch-size "$SAMPLE_DEVICE_BATCH_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --num-steps "$NUM_STEPS" \
    --log-every "$SAMPLE_LOG_EVERY" \
    --preview-samples "$PREVIEW_SAMPLES" \
    "${EXTRA_ARGS[@]}" \
    "$@"
