#!/usr/bin/env bash
# =============================================================================
# Smoke-test: single plain-JiT fine-tune on caltech-101 (H100, GPU 0, fp32, bs32)
# Confirms: converted weights load, model fits GPU, dataloader + FID pipeline work.
# Run this FIRST; once it's training and passes step-0 preview, launch the rest.
# =============================================================================
set -euo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

GPU="${GPU:-1}"   # GPUs 1-6 only on this box (never 0 or 7)
DATASET_NAME=caltech101
DATASET_ROOT="$DATA/caltech-101_images"

# fail fast on missing prerequisites
for p in "$REPO/.venv/bin/python" "$DATASET_ROOT/train" \
         "$REPO/files/weights/JiT-H-16-256.pth" \
         "$REPO/files/fid_stats/caltech-101-fid_stats.npz" \
         "$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export DATASET_NAME DATASET_ROOT
export FID_CACHE_REF="$REPO/files/fid_stats/caltech-101-fid_stats.npz"
export FD_DINO_CACHE_REF="$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"
export LOAD_FROM="$REPO/files/weights/JiT-H-16-256.pth"
export PYTHON="$REPO/.venv/bin/python"

# fp32 everywhere (H100 has headroom at bs32); matches SiT baseline precision
export HALF_PRECISION=False
export SAMPLING_HALF_PRECISION=False
export OPTIMIZER=adamw
export OPTIMIZER_MU_DTYPE=float32
export TRAIN_BATCH_SIZE=32

# memory hygiene (same as the SiT runs on this box)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer="
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT=plain_jit_finetune
# final best-fid sweep after training (short: 1/2/4/50 steps like the config's few-step regime)
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
export FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 4 50}"
export FINAL_EVAL_USE_WANDB=False

echo "=== JiT smoke-test: caltech101 on GPU $GPU, fp32, bs32 ==="
# num_workers=0: /dev/shm is only 64MB in this container -> multi-worker DataLoader
# dies with a shm bus error. Main-process loading is plenty for these dataset sizes.
bash scripts/train_plain_jit_finetune.sh smoke_h100 \
  --config.logging.wandb_entity="" \
  --config.dataset.num_workers=0 \
  "$@"
