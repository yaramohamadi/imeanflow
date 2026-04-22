#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"
export CONFIG_MODE="${CONFIG_MODE:-plain_pmf_finetune}"
export USE_WANDB="${USE_WANDB:-False}"

export DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_images}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"

RUN_DIR="${1:-files/logs/finetuning/plain_pMF_finetune_caltech101_caltech_h16_20step_smoke_taylor_20260421_211539_o9pw6t}"
shift || true

if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(1)
fi

for path in "$PYTHON" "$DATASET_ROOT" "$FID_CACHE_REF" "$FD_DINO_CACHE_REF" "$RUN_DIR"; do
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

WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_pmf_smoke_eval_taylor}" \
bash scripts/eval_best_fid_steps_plain_pmf.sh "$RUN_DIR" "${STEPS[@]}" -- \
  --config.dataset.root="$DATASET_ROOT" \
  --config.fid.cache_ref="$FID_CACHE_REF" \
  --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
  --config.fid.num_samples="${FID_NUM_SAMPLES:-100}" \
  --config.fid.num_images_to_log="${FID_NUM_IMAGES_TO_LOG:-16}" \
  --config.fid.sample_device_batch_size="${SAMPLE_DEVICE_BATCH_SIZE:-4}" \
  --config.sampling.half_precision="${SAMPLING_HALF_PRECISION:-True}" \
  --config.sampling.omega="${GUIDANCE_SCALE:-7.0}" \
  --config.sampling.cfg_scale="${GUIDANCE_SCALE:-7.0}" \
  --config.sampling.t_min="${SAMPLING_T_MIN:-0.2}" \
  --config.sampling.t_max="${SAMPLING_T_MAX:-0.6}"
