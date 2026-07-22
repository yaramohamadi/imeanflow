#!/usr/bin/env bash
# =============================================================================
# Launch all 5 plain-iMF fine-tunes in parallel, one dataset per GPU (1-5).
# iMF-XL/2, latent-space (like SiT). Init from ImageNet iMF-XL-2-full (step 1M).
# GPUs 1-6 only on this box (never 0 or 7). fp32, effective batch 32 (bs16 x ga2).
# Each runs in a detached screen: imf_gpu<N>_<dataset>.
#
# Protocol matched to the JiT/SiT baselines for comparability:
#   - use_ema=False (online eval)          - num_samples=5000 (not config's 10000)
#   - FID every 2500 steps @ 4-NFE         - keep best_fid only
#   - final sweep 1 / 2 / 250 (latent => 250 is the many-step reference)
#   - 30k train steps
#   - num_workers=16 + big /dev/shm (avoids the DataLoader bottleneck)
#   - bs16 x ga2 (author default; fits on 80GB post jit-fix, fastest option)
# =============================================================================
set -euo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

IMF_WEIGHTS="$REPO/files/weights/iMF-XL-2-full"

# Ensure /dev/shm is big enough for multi-worker PyTorch DataLoaders (default 64MB
# in this container -> worker bus error). Needs root; harmless if already large.
if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || echo "WARN: could not remount /dev/shm (need root); workers may crash" >&2
fi

# dataset_name : gpu : latent_root_basename
DATASETS=(
  "caltech101:1:caltech-101_processed_latents"
  "artbench10:2:artbench-10_processed_latents"
  "cub200:3:cub-200-2011_processed_latents"
  "food101:4:food-101_processed_latents"
  "stanfordcars:5:stanford-cars_processed_latents"
)

# fid/fdd cache basenames (match repo naming)
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

command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not installed (apt-get install -y screen)"; exit 2; }
[[ -d "$IMF_WEIGHTS" ]] || { echo "ERROR: missing iMF weights dir $IMF_WEIGHTS"; exit 2; }

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r DS GPU LATBASE <<< "$entry"
  ROOT="$DATA/$LATBASE"

  # extract latents if needed (zip fallback)
  if [[ ! -d "$ROOT/train" ]]; then
    echo "[$DS] extracting $LATBASE.zip ..."
    "$REPO/.venv/bin/python" -c "import zipfile; zipfile.ZipFile('$DATA/$LATBASE.zip').extractall('$DATA/')"
  fi
  [[ -d "$ROOT/train" ]] || { echo "ERROR: [$DS] no train/ dir after extract at $ROOT" >&2; exit 2; }

  fid="$REPO/files/fid_stats/${FID_REF[$DS]}"
  fdd="$REPO/files/fdd_stats/${FDD_REF[$DS]}"
  [[ -e "$fid" ]] || { echo "ERROR: [$DS] missing fid ref $fid" >&2; exit 2; }
  [[ -e "$fdd" ]] || { echo "ERROR: [$DS] missing fdd ref $fdd" >&2; exit 2; }

  SESS="imf_gpu${GPU}_${DS}"
  echo "[$DS] launching on GPU $GPU  (screen $SESS)"

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
      --config.logging.wandb_group='plain_imf_online_20260715' \
      --config.dataset.name=$DS \
      --config.dataset.num_workers=16 \
      --config.dataset.prefetch_factor=4 \
      --config.dataset.pin_memory=True \
      --config.training.max_train_steps=30000 \
      --config.training.use_ema=False \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/imf_${DS}_launch.log
  "
  sleep 3
done

echo
echo "=== launched. Sessions: ==="
screen -ls | grep imf_gpu || true
echo "Attach: screen -r imf_gpu1_caltech101   | GPUs: watch nvidia-smi"
