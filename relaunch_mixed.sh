#!/usr/bin/env bash
# Relaunch: 6 eval-only reruns (fixed num_classes + per-dataset stats) + 2 full retrains.
set -uo pipefail
REPO_ROOT="/opt/dlami/nvme/meanflow/imeanflow"
cd "$REPO_ROOT"
LOGS="$REPO_ROOT/files/logs"
FID="$REPO_ROOT/files/fid_stats"
FDD="$REPO_ROOT/files/fdd_stats"

echo "Killing any existing main_sit.py..."
pkill -9 -f main_sit.py 2>/dev/null || true
sleep 3

# --- eval-only helper: gpu, screen, config_mode, dataset, nclasses, fid_ref, fdd_ref, workdir ---
launch_eval() {
  local GPU="$1" NAME="$2" CMODE="$3" DS="$4" NC="$5" FIDF="$6" FDDF="$7" WD="$8"
  local LOG="$LOGS/${NAME}.out"
  screen -S "$NAME" -X quit 2>/dev/null || true
  echo "EVAL  gpu$GPU  $NAME  (ckpt in $WD/best_fid)"
  screen -dmS "$NAME" bash -c "cd $REPO_ROOT && \
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    CONFIG_MODE=$CMODE USE_WANDB=False \
    bash scripts/eval_best_fid_steps_plain_sit.sh $WD 1 2 250 \
      -- --config.dataset.num_classes=$NC \
         --config.model.num_classes=$NC \
         --config.dataset.num_classes_from_data=False \
         --config.dataset.name=$DS \
         --config.dataset.num_workers=0 \
         --config.fid.cache_ref=$FID/$FIDF \
         --config.fd_dino.cache_ref=$FDD/$FDDF \
    2>&1 | tee $LOG; ec=\$?; echo \"=== $NAME EXITED (\$ec) ===\"; exec bash"
  sleep 2
}

# --- full-retrain helper: gpu, screen, train_script ---
launch_train() {
  local GPU="$1" NAME="$2" SCRIPT="$3"
  local LOG="$LOGS/${NAME}.out"
  screen -S "$NAME" -X quit 2>/dev/null || true
  echo "TRAIN gpu$GPU  $NAME  ($SCRIPT)"
  screen -dmS "$NAME" bash -c "cd $REPO_ROOT && \
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_ALLOCATOR=platform USE_WANDB=True \
    bash scripts/$SCRIPT 2>&1 | tee $LOG; ec=\$?; echo \"=== $NAME EXITED (\$ec) ===\"; exec bash"
  sleep 2
}

# ---- 6 eval-only reruns (completed runs, valid best_fid checkpoints) ----
launch_eval 1 sit_gpu1_eval_cub_transportv      caltech_plain_sit_transportv cub-200-2011 200 \
  cub-200-2011_processed-fid_stats.npz cub-200-2011-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/cub_plain_SiT_transportv_taylor_20260711_231954_fdcy5u
launch_eval 2 sit_gpu2_eval_food_ddpmv          caltech_plain_sit_ddpmv food-101 101 \
  food-101_processed-fid_stats.npz food-101-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/food_plain_SiT_ddpmv_taylor_20260711_231956_ypbz3r
launch_eval 3 sit_gpu3_eval_food_transportv     caltech_plain_sit_transportv food-101 101 \
  food-101_processed-fid_stats.npz food-101-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/food_plain_SiT_transportv_taylor_20260711_231958_nso4nu
launch_eval 4 sit_gpu4_eval_stanford_ddpmv      caltech_plain_sit_ddpmv stanford-cars 196 \
  stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/stanford_plain_SiT_ddpmv_taylor_20260711_232001_qrndr0
launch_eval 6 sit_gpu6_eval_artbench_ddpmv      caltech_plain_sit_ddpmv artbench-10 10 \
  artbench-10_processed-fid_stats.npz artbench-10-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/artbench_plain_SiT_ddpmv_taylor_20260711_232005_tfbiz9
launch_eval 7 sit_gpu7_eval_artbench_transportv caltech_plain_sit_transportv artbench-10 10 \
  artbench-10_processed-fid_stats.npz artbench-10-fd_dino-vitb14_stats.npz \
  /opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning/artbench_plain_SiT_transportv_taylor_20260711_232840_vy7l4p

# ---- 2 full retrains (died at step 2500 from shm bus error) ----
launch_train 0 sit_gpu0_cub_ddpmv          train_cub_ddpmv.sh
launch_train 5 sit_gpu5_stanford_transportv train_stanford_transportv.sh

echo ""
echo "=== screens ==="
screen -ls | grep sit_gpu || true
