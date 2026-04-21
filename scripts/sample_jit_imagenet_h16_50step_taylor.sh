#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export PYTHON="${PYTHON:-.venv/bin/python}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/JiT-H-16-256.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-files/logs/jit_imagenet_h16_sample_$(date +%Y%m%d_%H%M%S)}"
export NUM_SAMPLES="${NUM_SAMPLES:-16}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_STEPS="${NUM_STEPS:-50}"
export CFG_SCALE="${CFG_SCALE:-2.2}"
export SEED="${SEED:-99}"

for path in "$PYTHON" "$LOAD_FROM"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required local path does not exist: $path" >&2
    exit 2
  fi
done

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

"$PYTHON" scripts/sample_jit_imagenet_h16_grid.py \
  --load-from "$LOAD_FROM" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-steps "$NUM_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --seed "$SEED" \
  "$@"
