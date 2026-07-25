#!/usr/bin/env bash
# Generate a self-contained per-dataset DiT MeFT launch script and start it in a
# detached screen. Usage: gen_and_launch.sh <dataset> <gpu>
# Literal paths, no arrays — avoids the quoting fragility that broke the orchestrator.
set -e
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
ds=$1; gpu=$2

case "$ds" in
  caltech101)   lat=caltech-101_processed_latents;  nc=101; fid=caltech-101-fid_stats.npz;            fdd=caltech-101-fd_dino-vitb14_stats.npz ;;
  artbench10)   lat=artbench-10_processed_latents;   nc=10;  fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz ;;
  cub200)       lat=cub-200-2011_processed_latents;  nc=200; fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      lat=food-101_processed_latents;      nc=101; fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) lat=stanford-cars_processed_latents; nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz ;;
  *) echo "unknown dataset $ds"; exit 1 ;;
esac

root="$DATA/$lat"
[ -d "$root/train" ] || { echo "ERROR: $root/train missing (extract first)"; exit 2; }

runsh="$REPO/files/logs/run_ditmeft_${ds}.sh"
cat > "$runsh" <<EOF
#!/bin/bash
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export DATASET_NAME=$ds
export DATASET_ROOT=$root
export DATASET_NUM_CLASSES=$nc
export FID_CACHE_REF=$REPO/files/fid_stats/$fid
export FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd
export PYTHON=$REPO/.venv/bin/python
export ENABLE_DOGFIT=False
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export USE_WANDB=True
export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2' FINAL_EVAL_USE_WANDB=False
bash scripts/run_caltech_dit_dmf_ddpmv_taylor.sh meft_online \\
  --config.load_from=$REPO/files/weights/DiT-XL-2-256x256.pt \\
  --config.logging.wandb_project=dit_meft \\
  --config.logging.wandb_entity=ea-fc \\
  --config.logging.wandb_group=dit_meft_ddpmv_20260722 \\
  --config.dataset.name=$ds \\
  --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \\
  --config.training.max_train_steps=30000 \\
  --config.fid.num_samples=5000 \\
  --config.training.force_fid_per_step=2500 \\
  --config.sampling.num_steps=4 \\
  --config.training.debug_log_during_train=False
EOF
chmod +x "$runsh"
screen -dmS "ditmeft_${ds}" bash -c "bash $runsh 2>&1 | tee -a $REPO/files/logs/ditmeft_${ds}_run.log"
echo "launched $ds on GPU $gpu (screen ditmeft_${ds}, script $runsh)"
