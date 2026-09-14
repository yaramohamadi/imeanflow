#!/usr/bin/env bash
# GT-anchored on-policy post-training of the Caltech-101 plain SiT-XL/2 on the
# single-GPU WSL box (RTX 6000 Ada, 48 GB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/run_caltech_plain_sit_gton_wsl.sh <run_label> [extra main_sit.py args...]

Arms:
  A  base checkpoint, no post-training
       bash scripts/run_caltech_plain_sit_gton_wsl.sh armA_base --config.eval_only=True
  B  matched-budget ordinary FM control (lambda=1 => zero corrective gradient)
       bash scripts/run_caltech_plain_sit_gton_wsl.sh armB_fm --config.model.sit_gt_on_lambda=1.0
  C  the method (lambda=0.5, real endpoint)
       bash scripts/run_caltech_plain_sit_gton_wsl.sh armC_gton
  D  self-endpoint contrast
       bash scripts/run_caltech_plain_sit_gton_wsl.sh armD_self --config.model.sit_gt_on_target=self
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

CONFIG_MODE="${CONFIG_MODE:-caltech_plain_sit_gton}"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/files/logs}"
# Single GPU on this box.
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
# Compare against the recorded 16-step baseline; 1 and 2 are cheap extras.
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 16}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"
XLA_FLAGS_VALUE="${XLA_FLAGS_VALUE:---xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="caltech_plain_SiT_GTon_wsl_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"

mkdir -p "${WORKDIR}"

cat <<EOF
Training workdir: ${WORKDIR}
CONFIG_MODE: ${CONFIG_MODE}
USE_WANDB: ${USE_WANDB}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VALUE}
EXTRA_ARGS: ${EXTRA_ARGS[*]:-<none>}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_FLAGS="${XLA_FLAGS_VALUE}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "${PYTHON}" "${REPO_ROOT}/main_sit.py" \
    --workdir="${WORKDIR}" \
    --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
    --config.logging.use_wandb="${USE_WANDB}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

if [[ "${RUN_FINAL_BEST_FID_EVAL}" == "True" ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS}"
  CONFIG_MODE="${CONFIG_MODE}" \
    PYTHON="${PYTHON}" \
    USE_WANDB="${FINAL_EVAL_USE_WANDB}" \
    WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_sit_gton_${RUN_LABEL}}" \
    bash "${SCRIPT_DIR}/eval_best_fid_steps_plain_sit.sh" "${WORKDIR}" "${FINAL_EVAL_STEP_ARRAY[@]}"
fi
