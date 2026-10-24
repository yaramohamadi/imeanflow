#!/usr/bin/env bash
# Plain SiT-XL/2 fine-tuning (Tab. 2 "FT" baseline for SiT) on all 5 target datasets, one GPU each.
# Protocol = configs/plain_sit_finetune_config.yml: lr 1e-4, batch 32, 30K steps, Adam b2 0.999, no EMA,
# CFG w=1.5, best_fid/ chosen by 4-step 10k-sample FID; then final best-ckpt eval at FINAL_EVAL_STEPS.
# Usage: RUN_LABEL=rebuttal GPUS="0 1 2 3 4" bash scripts/launch_sit_ft_all5.sh
set -uo pipefail
MAIN=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
RUN_LABEL="${RUN_LABEL:-rebuttal}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 4 8 250}"
cd "$MAIN"

# dataset  latent_dir  fid_stats  fdd_stats
SPECS=(
  "artbench10   artbench-10_processed_latents   artbench-10_processed-fid_stats.npz   artbench-10-fd_dino-vitb14_stats.npz"
  "caltech101   caltech-101_processed_latents   caltech-101-fid_stats.npz             caltech-101-fd_dino-vitb14_stats.npz"
  "cub200       cub-200-2011_processed_latents  cub-200-2011_processed-fid_stats.npz  cub-200-2011-fd_dino-vitb14_stats.npz"
  "food101      food-101_processed_latents      food-101_processed-fid_stats.npz      food-101-fd_dino-vitb14_stats.npz"
  "stanfordcars stanford-cars_processed_latents stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz"
)

for i in "${!SPECS[@]}"; do
  read -r ds latent fid fdd <<< "${SPECS[$i]}"
  gpu="${GPUS[$i]}"
  sess="sitft_${ds}"
  if screen -ls | grep -q "\.${sess}\b"; then echo "SKIP $ds: screen $sess already running"; continue; fi
  echo "LAUNCH $ds on GPU $gpu (screen $sess)"
  screen -dmS "$sess" bash -c "
    cd $MAIN
    export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    CONFIG_MODE=plain_sit_finetune PYTHON=$MAIN/.venv/bin/python USE_WANDB=True \
    BACKBONE=sit LOAD_FROM=$MAIN/files/weights/SiT-XL-2-256.pt \
    DATASET_NAME=$ds DATASET_ROOT=$DATA/$latent \
    FID_CACHE_REF=$MAIN/files/fid_stats/$fid FD_DINO_CACHE_REF=$MAIN/files/fdd_stats/$fdd \
    RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS' \
    bash scripts/train_plain_sit_finetune.sh $RUN_LABEL \
      2>&1 | tee -a $MAIN/files/logs/sitft_${RUN_LABEL}_${ds}.log
  "
done
screen -ls | grep sitft_
