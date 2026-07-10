#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Train DiT→SiT Transport-v on CUB-200-2011
# Override GPUs: CUDA_VISIBLE_DEVICES=2,3 ./train_cub_transportv.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_LABEL="cub_transportv"
CONFIG_MODE="caltech_plain_sit_transportv"
DATASET="cub-200-2011"
NUM_CLASSES=200

DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/files/logs}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="cub_plain_SiT_transportv_taylor_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"
mkdir -p "${WORKDIR}"

echo "=========================================="
echo "Training DiT→SiT Transport-v on CUB-200-2011"
echo "Workdir: ${WORKDIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Train
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TF_CPP_MIN_LOG_LEVEL=3 \
  XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONWARNINGS=ignore \
  "${PYTHON}" "${REPO_ROOT}/main_sit.py" \
    --workdir="${WORKDIR}" \
    --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
    --config.dataset.root="${HOME}/datasets/${DATASET}_processed_latents" \
    --config.dataset.name="${DATASET}" \
    --config.dataset.num_classes=${NUM_CLASSES} \
    --config.model.num_classes=${NUM_CLASSES} \
    --config.fid.cache_ref="${REPO_ROOT}/files/fid_stats/cub-200-2011_processed-fid_stats.npz" \
    --config.fd_dino.cache_ref="${REPO_ROOT}/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz" \
    --config.logging.wandb_name="CUB_SIT_plain_DiTinit_Transport-v" \
    --config.logging.use_wandb="${USE_WANDB}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

# Final evaluation
echo ""
echo "Running final evaluation (1, 2, 4, 250 steps)..."
CONFIG_MODE="${CONFIG_MODE}" PYTHON="${PYTHON}" USE_WANDB=False \
  bash "${SCRIPT_DIR}/eval_best_fid_steps_plain_sit.sh" "${WORKDIR}" 1 2 4 250

echo "=========================================="
echo "Complete! Results: ${WORKDIR}"
echo "=========================================="
