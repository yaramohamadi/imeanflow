#!/usr/bin/env bash
set -euo pipefail

RUN_LABEL="${1:-abl}"
PYTHON="${PYTHON:-.venv/bin/python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-1,2}"
BASE_WORKDIR="${BASE_WORKDIR:-files/logs/finetuning}"

WORKDIR="${BASE_WORKDIR}/caltech_SiT_DMF_guidance_interval_0p3_1p0_start_6000_${RUN_LABEL}"

echo "=== Running ${WORKDIR} ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" main.py \
    --workdir="${WORKDIR}" \
    --config=configs/load_config.py:caltech_sit_dmf_finetune \
    --config.dataset.num_workers=0 \
    --config.sampling.num_steps=1 \
    --config.sampling.omega=1.0 \
    --config.model.training_guidance_interval_strategy=fixed \
    --config.model.training_guidance_t_min=0.3 \
    --config.model.training_guidance_t_max=1.0 \
    --config.model.training_guidance_start_step=6000 \
    2>&1 | stdbuf -oL grep -a -v -E '(\+ptx[0-9]+|recognized feature for this target|ignoring feature)'
