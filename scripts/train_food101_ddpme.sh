#!/usr/bin/env bash
set -euo pipefail
# DiT->SiT DDPM-e plain SiT (DiT-init) on Food-101. dev5 paths. Missing SiT FT gap fill.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_MODE="caltech_plain_sit_ddpme"
DATASET="food-101"
NUM_CLASSES=101
DATA_ROOT="${DATA_ROOT:-/opt/dlami/nvme/meanflow/datasets}"
DIT_WEIGHTS="${DIT_WEIGHTS:-${REPO_ROOT}/files/weights/DiT-XL-2-256x256.pt}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
USE_WANDB="${USE_WANDB:-False}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/files/logs}"
NOW=$(date "+%Y%m%d_%H%M%S"); SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="food_plain_SiT_ddpme_taylor_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"; mkdir -p "${WORKDIR}"
echo "== SiT DDPM-e Food-101 -> ${WORKDIR} (GPU ${CUDA_VISIBLE_DEVICES}) =="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" TF_CPP_MIN_LOG_LEVEL=3 \
  XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONWARNINGS=ignore \
  "${PYTHON}" "${REPO_ROOT}/main_sit.py" \
    --workdir="${WORKDIR}" \
    --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
    --config.training.batch_size=16 \
    --config.training.grad_accum_steps=2 \
    --config.fid.num_samples=10000 \
    --config.load_from="${DIT_WEIGHTS}" \
    --config.dataset.root="${DATA_ROOT}/food-101_processed_latents" \
    --config.dataset.name="${DATASET}" \
    --config.dataset.num_workers=0 \
    --config.dataset.num_classes=${NUM_CLASSES} \
    --config.model.num_classes=${NUM_CLASSES} \
    --config.fid.cache_ref="${REPO_ROOT}/files/fid_stats/food-101_processed-fid_stats.npz" \
    --config.fd_dino.cache_ref="${REPO_ROOT}/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz" \
    --config.logging.wandb_name="FOOD_SIT_plain_DiTinit_DDPM-e" \
    --config.logging.use_wandb="${USE_WANDB}" \
    2>&1 | tee -a "${WORKDIR}/output.log"
echo "== final eval NFE 1 2 4 250 =="
CONFIG_MODE="${CONFIG_MODE}" PYTHON="${PYTHON}" USE_WANDB=False \
  bash "${SCRIPT_DIR}/eval_best_fid_steps_plain_sit.sh" "${WORKDIR}" 1 2 4 250 \
    -- --config.load_from="${DIT_WEIGHTS}" \
       --config.dataset.num_classes=${NUM_CLASSES} \
       --config.model.num_classes=${NUM_CLASSES} \
       --config.dataset.name="${DATASET}" \
       --config.dataset.root="${DATA_ROOT}/food-101_processed_latents" \
       --config.fid.num_samples=10000 \
       --config.fid.cache_ref="${REPO_ROOT}/files/fid_stats/food-101_processed-fid_stats.npz" \
       --config.fd_dino.cache_ref="${REPO_ROOT}/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz"
echo "== Complete: ${WORKDIR} =="
