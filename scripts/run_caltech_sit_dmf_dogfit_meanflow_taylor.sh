#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: VC_TARGET_SOURCE=ema bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh <run_label> [extra main.py args...]

Examples:
  VC_TARGET_SOURCE=ema bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh caltech_dogfit
  VC_TARGET_SOURCE=online bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh caltech_dogfit_online

This script:
  1) runs Caltech SiT-DMF DogFit meanflow locally on Taylor
  2) uses home/ens paths via the dedicated config
  3) ablates whether DogFit's conditioned target v_c comes from the EMA or online model
  4) always trains with stop-gradient on both v_c and v_u in the DogFit path
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
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-0,1}}"
VC_TARGET_SOURCE="${VC_TARGET_SOURCE:-ema}"
PYTHONUNBUFFERED_VALUE="${PYTHONUNBUFFERED:-1}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

case "${VC_TARGET_SOURCE}" in
  ema)
    USE_EMA_VC="True"
    ;;
  online)
    USE_EMA_VC="False"
    ;;
  *)
    echo "ERROR: VC_TARGET_SOURCE must be 'ema' or 'online', got: ${VC_TARGET_SOURCE}" >&2
    exit 2
    ;;
esac

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_SiT_DMF_DogFit_meanflow_taylor_${VC_TARGET_SOURCE}_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
USE_WANDB: ${USE_WANDB}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
VC_TARGET_SOURCE: ${VC_TARGET_SOURCE}
USE_EMA_VC: ${USE_EMA_VC}
PYTHONUNBUFFERED: ${PYTHONUNBUFFERED_VALUE}
TRAINING_GUIDANCE_START_STEP: 0
RUN_FINAL_BEST_FID_EVAL: ${RUN_FINAL_BEST_FID_EVAL}
FINAL_EVAL_STEPS: ${FINAL_EVAL_STEPS}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED_VALUE}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" main.py \
    --workdir="${WORKDIR}" \
    --config="configs/load_config.py:${CONFIG_MODE}" \
    --config.model.training_guidance_start_step="0" \
    --config.model.use_ema_vc="${USE_EMA_VC}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    --config.logging.wandb_name="${JOBNAME}" \
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
