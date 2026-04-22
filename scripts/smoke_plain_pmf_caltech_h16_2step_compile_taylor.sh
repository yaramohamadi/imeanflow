#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export PYTHON="${PYTHON:-.venv/bin/python}"
export RUN_LABEL="${RUN_LABEL:-caltech_h16_2step_compile_taylor}"
export USE_WANDB="${USE_WANDB:-False}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-auto}"
export SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-1}"
export HALF_PRECISION="${HALF_PRECISION:-False}"
export SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-True}"

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

bash scripts/train_plain_pmf_finetune.sh "$RUN_LABEL" \
  --config.training.max_train_steps=2 \
  --config.training.sample_per_step=0 \
  --config.training.fid_per_step=0 \
  --config.training.force_fid_per_step=0 \
  --config.training.preview_at_step0=False \
  --config.training.save_best_fid_only=False \
  --config.training.checkpoint_per_epoch=1000000 \
  "$@"
