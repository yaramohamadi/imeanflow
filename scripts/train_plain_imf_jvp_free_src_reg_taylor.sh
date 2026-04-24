#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/iMF-XL-2-full}"

export USE_WANDB="${USE_WANDB:-True}"
export RUN_LABEL="${RUN_LABEL:-caltech_xl2_jvp_free_src_reg_taylor}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_jvp_free_src_reg_finetune}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_imf_jvp_free_src_reg_${RUN_LABEL}}"

export METRIC_NUM_STEPS="${METRIC_NUM_STEPS:-4}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
export SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-32}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

for path in "$PYTHON" "$DATASET_ROOT" "$FID_CACHE_REF" "$FD_DINO_CACHE_REF" "$LOAD_FROM"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required Taylor path does not exist: $path" >&2
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

bash scripts/train_plain_imf_jvp_free_src_reg_finetune.sh "$RUN_LABEL" "$@"
