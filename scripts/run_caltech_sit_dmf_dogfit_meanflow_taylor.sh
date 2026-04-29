#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: USE_LATE_START=True bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh <run_label> [extra main.py args...]

Examples:
  USE_LATE_START=True bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh caltech_dogfit
  USE_LATE_START=False bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh caltech_dogfit_nols

This script:
  1) runs Caltech SiT-DMF DogFit meanflow locally on Taylor
  2) uses home/ens paths via the dedicated config
  3) lets you toggle late-start guidance with USE_LATE_START=True/False
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_dogfit_meanflow}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-files/logs}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1}"
USE_LATE_START="${USE_LATE_START:-True}"
LATE_START_STEP="${LATE_START_STEP:-6000}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

if [[ "${USE_LATE_START}" == "True" ]]; then
  TRAINING_GUIDANCE_START_STEP="${LATE_START_STEP}"
else
  TRAINING_GUIDANCE_START_STEP="0"
fi

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_SiT_DMF_DogFit_meanflow_taylor_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
USE_WANDB: ${USE_WANDB}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
USE_LATE_START: ${USE_LATE_START}
TRAINING_GUIDANCE_START_STEP: ${TRAINING_GUIDANCE_START_STEP}
RUN_FINAL_BEST_FID_EVAL: ${RUN_FINAL_BEST_FID_EVAL}
FINAL_EVAL_STEPS: ${FINAL_EVAL_STEPS}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" main.py \
    --workdir="${WORKDIR}" \
    --config="configs/load_config.py:${CONFIG_MODE}" \
    --config.model.training_guidance_start_step="${TRAINING_GUIDANCE_START_STEP}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    MODEL_STR="imfSiT_DMF_XL_2" \
    MODEL_USE_DOGFIT="True" \
    TARGET_USE_NULL_CLASS="True" \
    CLASS_DROPOUT_PROB="0.0" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    bash scripts/eval_best_fid_steps.sh "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
fi
