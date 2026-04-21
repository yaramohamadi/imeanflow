#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_images}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/JiT-H-16-256.pth}"

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
export RUN_LABEL="${RUN_LABEL:-caltech_h16_20step_smoke_taylor}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}}"

export HALF_PRECISION="${HALF_PRECISION:-True}"
export HALF_PRECISION_DTYPE="${HALF_PRECISION_DTYPE:-float16}"
export SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-False}"
export SAMPLING_HALF_PRECISION_DTYPE="${SAMPLING_HALF_PRECISION_DTYPE:-float16}"
export OPTIMIZER="${OPTIMIZER:-lion}"
export OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
export FINAL_EVAL_EXTRA_ARGS="${FINAL_EVAL_EXTRA_ARGS:---config.training.force_metric_num_steps=}"

for path in "$PYTHON" "$DATASET_ROOT" "$FID_CACHE_REF" "$FD_DINO_CACHE_REF" "$LOAD_FROM"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required local path does not exist: $path" >&2
    exit 2
  fi
done

if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: DATASET_ROOT must be an ImageFolder-style root containing train/<class>: $DATASET_ROOT" >&2
  exit 2
fi

if [[ "${REQUIRE_GPU:-True}" =~ ^([Tt]rue|1|[Yy]es|[Yy]|[Oo]n)$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    sys.exit("ERROR: no JAX GPU device is visible. Run inside a GPU allocation or set REQUIRE_GPU=False intentionally.")
PY
fi

bash scripts/train_plain_jit_finetune.sh "$RUN_LABEL" \
  --config.model.model_str=flaxJiT_H_16 \
  --config.training.max_train_steps=20 \
  --config.training.sample_per_step=10 \
  --config.training.fid_per_step=10 \
  --config.training.force_fid_per_step=10 \
  --config.training.force_metric_num_steps=4 \
  --config.training.preview_num_steps='(4,)' \
  --config.sampling.num_steps=4 \
  --config.fid.num_samples=100 \
  --config.fid.num_images_to_log=16 \
  --config.training.preview_at_step0=True \
  "$@"
