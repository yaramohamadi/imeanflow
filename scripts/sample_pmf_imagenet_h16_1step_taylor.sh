#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export PYTHON="${PYTHON:-.venv/bin/python}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full}"
export LOAD_ZIP="${LOAD_ZIP:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full.zip}"
export OUTPUT_DIR="${OUTPUT_DIR:-files/logs/pmf_imagenet_h16_1step_sample_$(date +%Y%m%d_%H%M%S)}"
export NUM_SAMPLES="${NUM_SAMPLES:-16}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_STEPS="${NUM_STEPS:-1}"
export CFG_SCALE="${CFG_SCALE:-7.0}"
export T_MIN="${T_MIN:-0.2}"
export T_MAX="${T_MAX:-0.6}"
export EMA="${EMA:-1000}"
export SEED="${SEED:-99}"

if [[ ! -e "$PYTHON" ]]; then
  echo "ERROR: required local path does not exist: $PYTHON" >&2
  exit 2
fi

if [[ ! -d "$LOAD_FROM" ]]; then
  if [[ -f "$LOAD_ZIP" ]]; then
    echo "Extracting pMF checkpoint zip to $(dirname "$LOAD_FROM") ..."
    unzip -q "$LOAD_ZIP" -d "$(dirname "$LOAD_FROM")"
  else
    echo "ERROR: checkpoint directory missing and zip not found: $LOAD_FROM / $LOAD_ZIP" >&2
    exit 2
  fi
fi

if [[ "${REQUIRE_GPU:-True}" =~ ^([Tt]rue|1|[Yy]es|[Yy]|[Oo]n)$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    sys.exit("ERROR: no JAX GPU device is visible. Run inside a GPU session or set REQUIRE_GPU=False intentionally.")
PY
fi

"$PYTHON" scripts/sample_pmf_imagenet_h16_grid.py \
  --load-from "$LOAD_FROM" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-steps "$NUM_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --t-min "$T_MIN" \
  --t-max "$T_MAX" \
  --ema "$EMA" \
  --seed "$SEED" \
  "$@"
