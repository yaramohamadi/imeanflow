#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Train DiT->SiT DDPM-e (plain SiT/FM, DiT-init, NO velocity mapping, NO Diff2Flow)
# on ArtBench-10. dev5 paths. Override GPUs: CUDA_VISIBLE_DEVICES=2,3 ./train_artbench_ddpme.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_LABEL="artbench_ddpme"
CONFIG_MODE="caltech_plain_sit_ddpme"
DATASET="artbench-10"
NUM_CLASSES=10
DATA_ROOT="${DATA_ROOT:-/opt/dlami/nvme/meanflow/datasets}"
DIT_WEIGHTS="${DIT_WEIGHTS:-${REPO_ROOT}/files/weights/DiT-XL-2-256x256.pt}"

DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
USE_WANDB="${USE_WANDB:-True}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/files/logs}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="artbench_plain_SiT_ddpme_taylor_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"
mkdir -p "${WORKDIR}"

echo "=========================================="
echo "Training DiT->SiT DDPM-e on ArtBench-10"
echo "Workdir: ${WORKDIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}   load_from: ${DIT_WEIGHTS}"
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
    --config.training.batch_size=16 \
    --config.training.grad_accum_steps=2 \
    --config.load_from="${DIT_WEIGHTS}" \
    --config.dataset.root="${DATA_ROOT}/artbench-10_processed_latents" \
    --config.dataset.name="${DATASET}" \
    --config.dataset.num_workers=0 \
    --config.dataset.num_classes=${NUM_CLASSES} \
    --config.model.num_classes=${NUM_CLASSES} \
    --config.fid.cache_ref="${REPO_ROOT}/files/fid_stats/artbench-10_processed-fid_stats.npz" \
    --config.fd_dino.cache_ref="${REPO_ROOT}/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz" \
    --config.logging.wandb_name="ARTBENCH_SIT_plain_DiTinit_DDPM-e" \
    --config.logging.use_wandb="${USE_WANDB}" \
    2>&1 | tee -a "${WORKDIR}/output.log"

# Final evaluation (1, 2, 250 steps; 4 is the during-training metric, skipped)
echo ""
echo "Running final evaluation (1, 2, 250 steps)..."
CONFIG_MODE="${CONFIG_MODE}" PYTHON="${PYTHON}" USE_WANDB=False \
  bash "${SCRIPT_DIR}/eval_best_fid_steps_plain_sit.sh" "${WORKDIR}" 1 2 250 \
    -- --config.load_from="${DIT_WEIGHTS}" \
       --config.dataset.num_classes=${NUM_CLASSES} \
       --config.model.num_classes=${NUM_CLASSES} \
       --config.dataset.name="${DATASET}" \
       --config.dataset.root="${DATA_ROOT}/artbench-10_processed_latents" \
       --config.fid.cache_ref="${REPO_ROOT}/files/fid_stats/artbench-10_processed-fid_stats.npz" \
       --config.fd_dino.cache_ref="${REPO_ROOT}/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz"

echo "=========================================="
echo "Complete! Results: ${WORKDIR}"
echo "=========================================="
