#!/usr/bin/env bash
# =============================================================================
# Launch ONE plain-iMF fine-tune on a chosen GPU. Mirrors launch_imf_all5.sh env
# exactly (iMF-XL/2, latent-space, init iMF-XL-2-full, fp32, bs16 x ga2, 40k steps,
# use_ema=False, FID every 2500 @ 4-NFE, best_fid only, final sweep 1/2/250).
# Usage: bash scripts/launch_imf_one.sh <dataset> <gpu>
#   dataset in: caltech101 artbench10 cub200 food101 stanfordcars
# Runs in detached screen: imf_gpu<N>_<dataset>.
# =============================================================================
set -euo pipefail

DS="${1:?usage: launch_imf_one.sh <dataset> <gpu>}"
GPU="${2:?usage: launch_imf_one.sh <dataset> <gpu>}"

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
IMF_WEIGHTS="$REPO/files/weights/iMF-XL-2-full"

declare -A LAT=(
  [caltech101]="caltech-101_processed_latents"
  [artbench10]="artbench-10_processed_latents"
  [cub200]="cub-200-2011_processed_latents"
  [food101]="food-101_processed_latents"
  [stanfordcars]="stanford-cars_processed_latents"
)
declare -A FID_REF=(
  [caltech101]="caltech-101-fid_stats.npz"
  [artbench10]="artbench-10_processed-fid_stats.npz"
  [cub200]="cub-200-2011_processed-fid_stats.npz"
  [food101]="food-101_processed-fid_stats.npz"
  [stanfordcars]="stanford_cars_processed-fid_stats.npz"
)
declare -A FDD_REF=(
  [caltech101]="caltech-101-fd_dino-vitb14_stats.npz"
  [artbench10]="artbench-10-fd_dino-vitb14_stats.npz"
  [cub200]="cub-200-2011-fd_dino-vitb14_stats.npz"
  [food101]="food-101-fd_dino-vitb14_stats.npz"
  [stanfordcars]="stanford-cars-fd_dino-vitb14_stats.npz"
)
[[ -n "${LAT[$DS]:-}" ]] || { echo "ERROR: unknown dataset '$DS'"; exit 2; }

LATBASE="${LAT[$DS]}"; ROOT="$DATA/$LATBASE"
command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not installed"; exit 2; }
[[ -d "$IMF_WEIGHTS" ]] || { echo "ERROR: missing iMF weights dir $IMF_WEIGHTS"; exit 2; }

if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || echo "WARN: could not remount /dev/shm (need root); workers may crash" >&2
fi

if [[ ! -d "$ROOT/train" ]]; then
  echo "[$DS] extracting $LATBASE.zip ..."
  "$REPO/.venv/bin/python" -c "import zipfile; zipfile.ZipFile('$DATA/$LATBASE.zip').extractall('$DATA/')"
fi
[[ -d "$ROOT/train" ]] || { echo "ERROR: [$DS] no train/ dir after extract at $ROOT" >&2; exit 2; }

fid="$REPO/files/fid_stats/${FID_REF[$DS]}"
fdd="$REPO/files/fdd_stats/${FDD_REF[$DS]}"
[[ -e "$fid" ]] || { echo "ERROR: [$DS] missing fid ref $fid" >&2; exit 2; }
[[ -e "$fdd" ]] || { echo "ERROR: [$DS] missing fdd ref $fdd" >&2; exit 2; }

# guard: refuse if GPU already busy (>2GB used) to avoid collisions
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
if [[ -n "$used" && "$used" -gt 2000 ]]; then
  echo "ERROR: GPU $GPU already has ${used}MiB used — refusing to launch (collision guard). Override by freeing it first." >&2
  exit 3
fi

SESS="imf_gpu${GPU}_${DS}"
if screen -ls 2>/dev/null | grep -q "\.${SESS}[[:space:]]"; then
  echo "ERROR: screen session $SESS already exists" >&2; exit 4
fi
echo "[$DS] launching on GPU $GPU  (screen $SESS)  -> 40k steps, use_ema=False"

screen -dmS "$SESS" bash -c "
  cd $REPO
  export CUDA_VISIBLE_DEVICES=$GPU
  export DATASET_NAME=$DS DATASET_ROOT='$ROOT'
  export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd'
  export LOAD_FROM='$IMF_WEIGHTS'
  export PYTHON='$REPO/.venv/bin/python'
  export HALF_PRECISION=False SAMPLING_HALF_PRECISION=False
  export OPTIMIZER=adamw OPTIMIZER_MU_DTYPE=float32
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer='
  export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
  export USE_WANDB=True WANDB_PROJECT=plain_imf_finetune
  export TRAIN_BATCH_SIZE=16 GRAD_ACCUM_STEPS=2
  export WANDB_NAME=${DS}_plain_imf_online
  export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False
  export FORCE_FID_PER_STEP=2500
  bash scripts/train_plain_imf_finetune.sh h100 --config.logging.wandb_entity='ea-fc' \
    --config.logging.wandb_group='plain_imf_online_20260722' \
    --config.dataset.name=$DS \
    --config.dataset.num_workers=16 \
    --config.dataset.prefetch_factor=4 \
    --config.dataset.pin_memory=True \
    --config.training.max_train_steps=40000 \
    --config.training.use_ema=False \
    --config.fid.num_samples=5000 \
    2>&1 | tee -a $REPO/files/logs/imf_${DS}_run.log
"
sleep 2
screen -ls | grep "$SESS" || echo "WARN: session not found after launch"
echo "Attach: screen -r $SESS   | Log: files/logs/imf_${DS}_run.log"
