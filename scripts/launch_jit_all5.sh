#!/usr/bin/env bash
# =============================================================================
# Launch all 5 plain-JiT fine-tunes in parallel, one dataset per GPU (1-5).
# GPUs 1-6 only on this box (never 0 or 7). fp32, batch 32.
# Each runs in a detached screen: jit_gpu<N>_<dataset>.
# Extracts image zips on demand. Run launch_jit_smoke.sh FIRST to de-risk.
# =============================================================================
set -euo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

# Ensure /dev/shm is big enough for multi-worker PyTorch DataLoaders (default 64MB
# in this container -> worker bus error). Needs root; harmless if already large.
if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || echo "WARN: could not remount /dev/shm (need root); workers may crash" >&2
fi

# dataset_name : gpu : zipbase : imgdir
DATASETS=(
  "caltech101:1:caltech-101_images:caltech-101_images"
  "artbench10:2:artbench-10_images:artbench-10_images"
  "cub200:3:cub-200-2011_images:cub-200-2011_images"
  "food101:4:food-101_images:food-101_images"
  "stanfordcars:5:stanford-cars_images:stanford-cars_images"
)

# fid/fdd cache basenames differ per dataset (match repo naming)
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
[[ -f "$REPO/files/weights/JiT-H-16-256.pth" ]] || { echo "ERROR: missing JiT weights"; exit 2; }

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r DS GPU ZIPBASE IMGDIR <<< "$entry"
  ROOT="$DATA/$IMGDIR"

  # extract images if needed (unzip may be absent -> python fallback)
  if [[ ! -d "$ROOT/train" ]]; then
    echo "[$DS] extracting $ZIPBASE.zip ..."
    "$REPO/.venv/bin/python" -c "import zipfile; zipfile.ZipFile('$DATA/$ZIPBASE.zip').extractall('$DATA/')"
  fi
  [[ -d "$ROOT/train" ]] || { echo "ERROR: [$DS] no train/ dir after extract at $ROOT" >&2; exit 2; }

  fid="$REPO/files/fid_stats/${FID_REF[$DS]}"
  fdd="$REPO/files/fdd_stats/${FDD_REF[$DS]}"
  [[ -e "$fid" ]] || { echo "ERROR: [$DS] missing fid ref $fid" >&2; exit 2; }
  [[ -e "$fdd" ]] || { echo "ERROR: [$DS] missing fdd ref $fdd" >&2; exit 2; }

  SESS="jit_gpu${GPU}_${DS}"
  echo "[$DS] launching on GPU $GPU  (screen $SESS)"

  screen -dmS "$SESS" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$GPU
    export DATASET_NAME=$DS DATASET_ROOT='$ROOT'
    export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd'
    export LOAD_FROM='$REPO/files/weights/JiT-H-16-256.pth'
    export PYTHON='$REPO/.venv/bin/python'
    export HALF_PRECISION=False SAMPLING_HALF_PRECISION=False
    export OPTIMIZER=adamw OPTIMIZER_MU_DTYPE=float32 TRAIN_BATCH_SIZE=32
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer='
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    export USE_WANDB=True WANDB_PROJECT=plain_jit_finetune
    export WANDB_NAME=${DS}_plain_jit_online
    export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 50' FINAL_EVAL_USE_WANDB=False
    bash scripts/train_plain_jit_finetune.sh h100 --config.logging.wandb_entity='' \
      --config.dataset.num_workers=16 \
      --config.dataset.prefetch_factor=4 \
      --config.dataset.pin_memory=True \
      --config.training.max_train_steps=40000 \
      --config.training.use_ema=False \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/jit_${DS}_launch.log
  "
  sleep 3
done

echo
echo "=== launched. Sessions: ==="
screen -ls | grep jit_gpu || true
echo "Attach: screen -r jit_gpu0_caltech101   | GPUs: watch nvidia-smi"
