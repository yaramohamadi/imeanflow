#!/usr/bin/env bash
# =============================================================================
# Launch all 5 DiT-XL DMF MeFT runs WITH OBJECTIVE-SIDE EMA (use_ema_vc=True).
#   - EMA teacher supplies the conditioned velocity target v_c in the meanflow
#     objective (guided, non-dogfit path; requires the imf.py use_ema_vc gate).
#   - use_ema=True maintains state.ema_params (the teacher); ema_val 0.9995.
#   - 40k train steps (vs 30k for the original spiky online runs).
#   - config caltech_dit_dmf_ddpmv_ema byte-matches the DiT training model
#     section (dit_native time remap) + defaults num_samples handling.
# One dataset per GPU: 0,1,4,6,7 (user explicitly authorized 0/7 for this).
# Effective batch 32 (bs8 x ga4). Screens: ditemavc_gpu<N>_<dataset>.
# =============================================================================
set -euo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

DIT_WEIGHTS="$REPO/files/weights/DiT-XL-2-256x256.pt"
[[ -f "$DIT_WEIGHTS" ]] || { echo "ERROR: missing DiT weights $DIT_WEIGHTS"; exit 2; }
command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not installed"; exit 2; }

if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || echo "WARN: could not remount /dev/shm" >&2
fi

# dataset : gpu : latent_basename : num_classes
DATASETS=(
  "caltech101:0:caltech-101_processed_latents:101"
  "artbench10:1:artbench-10_processed_latents:10"
  "cub200:4:cub-200-2011_processed_latents:200"
  "food101:6:food-101_processed_latents:101"
  "stanfordcars:7:stanford-cars_processed_latents:196"
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

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r DS GPU LATBASE NC <<< "$entry"
  ROOT="$DATA/$LATBASE"

  if [[ ! -d "$ROOT/train" ]]; then
    echo "[$DS] extracting $LATBASE.zip ..."
    "$REPO/.venv/bin/python" -c "import zipfile; zipfile.ZipFile('$DATA/$LATBASE.zip').extractall('$DATA/')"
  fi
  [[ -d "$ROOT/train" ]] || { echo "ERROR: [$DS] no train/ at $ROOT" >&2; exit 2; }

  fid="$REPO/files/fid_stats/${FID_REF[$DS]}"
  fdd="$REPO/files/fdd_stats/${FDD_REF[$DS]}"
  [[ -e "$fid" ]] || { echo "ERROR: [$DS] missing fid ref $fid" >&2; exit 2; }
  [[ -e "$fdd" ]] || { echo "ERROR: [$DS] missing fdd ref $fdd" >&2; exit 2; }

  SESS="ditemavc_gpu${GPU}_${DS}"
  echo "[$DS] launching DiT MeFT EMA-vc on GPU $GPU  (screen $SESS)"

  screen -dmS "$SESS" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$GPU
    export CONFIG_MODE=caltech_dit_dmf_ddpmv_ema
    export DATASET_NAME=$DS
    export DATASET_ROOT='$ROOT'
    export DATASET_NUM_CLASSES=$NC
    export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd'
    export PYTHON='$REPO/.venv/bin/python'
    export ENABLE_DOGFIT=False
    export TRAIN_USE_EMA=True USE_EMA_VC=True
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    export USE_WANDB=True
    export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False
    bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh meft_emavc \
      --config.load_from='$DIT_WEIGHTS' \
      --config.logging.wandb_project='dit_meft' \
      --config.logging.wandb_entity='ea-fc' \
      --config.logging.wandb_group='dit_meft_emavc_20260723' \
      --config.dataset.name=$DS \
      --config.dataset.num_workers=16 \
      --config.dataset.prefetch_factor=4 \
      --config.dataset.pin_memory=True \
      --config.training.max_train_steps=40000 \
      --config.fid.num_samples=5000 \
      --config.training.force_fid_per_step=2500 \
      2>&1 | tee -a $REPO/files/logs/ditemavc_${DS}_launch.log
  "
  sleep 3
done

echo
echo "=== launched. Sessions: ==="
screen -ls | grep ditemavc_gpu || true
