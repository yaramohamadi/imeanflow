#!/usr/bin/env bash
# SANITY CHECK for the teacher-map fix: caltech101 ONLY, GPU 1, EMA-vc (use_ema_vc),
# with an EARLY 4-step FID (every 500 steps, 2000 samples) so a number appears well
# under 30 min. If FID descends we fan out to all 5. Uses the patched imf.py
# (teacher_v_cond_fn now maps model time + wraps epsilon like the online v_c) and
# the patched config (sampling.num_steps=4 -> best_fid selected on 4-step FID).
set -euo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

DIT_WEIGHTS="$REPO/files/weights/DiT-XL-2-256x256.pt"
[[ -f "$DIT_WEIGHTS" ]] || { echo "ERROR: missing DiT weights $DIT_WEIGHTS"; exit 2; }

DS=caltech101; GPU=1; LATBASE=caltech-101_processed_latents; NC=101
ROOT="$DATA/$LATBASE"
[[ -d "$ROOT/train" ]] || { echo "ERROR: no train/ at $ROOT" >&2; exit 2; }
fid="$REPO/files/fid_stats/caltech-101-fid_stats.npz"
fdd="$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"
[[ -e "$fid" ]] || { echo "ERROR: missing fid ref $fid" >&2; exit 2; }
[[ -e "$fdd" ]] || { echo "ERROR: missing fdd ref $fdd" >&2; exit 2; }

SESS="ditemavc_sanity_caltech"
echo "[$DS] launching EMA-vc SANITY on GPU $GPU (screen $SESS)"
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
  export RUN_FINAL_BEST_FID_EVAL=False
  bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh meft_emavc_sanity \
    --config.load_from='$DIT_WEIGHTS' \
    --config.logging.wandb_project='dit_meft' \
    --config.logging.wandb_entity='ea-fc' \
    --config.logging.wandb_group='dit_meft_emavc_sanity_20260723' \
    --config.dataset.name=$DS \
    --config.dataset.num_workers=16 \
    --config.dataset.prefetch_factor=4 \
    --config.dataset.pin_memory=True \
    --config.training.max_train_steps=6000 \
    --config.fid.num_samples=2000 \
    --config.training.force_fid_per_step=500 \
    2>&1 | tee -a $REPO/files/logs/ditemavc_sanity_caltech.log
"
sleep 3
echo "=== session ==="; screen -ls | grep "$SESS" || true
