#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/run_caltech_plain_sit_ditinit_taylor.sh <run_label> [extra main_sit.py args...]

Example:
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_caltech_plain_sit_ditinit_taylor.sh caltech_plain_sit_ditinit

This script:
  1) runs Caltech plain SiT locally on Taylor
  2) initializes the plain SiT backbone from the DiT checkpoint
  3) keeps training and sampling on the native plain SiT path
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_plain_sit_ditinit}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0,1}}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_plain_SiT_DiTinit_taylor_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
USE_WANDB: ${USE_WANDB}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
RUN_FINAL_BEST_FID_EVAL: ${RUN_FINAL_BEST_FID_EVAL}
FINAL_EVAL_STEPS: ${FINAL_EVAL_STEPS}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" main_sit.py \
    --workdir="${WORKDIR}" \
    --config="configs/load_config.py:${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_sit_ditinit_${RUN_LABEL}}" \
    bash scripts/eval_best_fid_steps_plain_sit.sh "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
fi
