#!/usr/bin/env bash
# Launch one pixel-space JiT MeFT run on a given GPU. Usage: <dataset> <gpu>
# Mirrors scripts/gen_launch_sit_meft.sh but for the pixel-space JiT-DMF path
# (main_imf_jit.py + configs/caltech_jit_dmf_meft_config.yml). No VAE latents;
# uses the ImageFolder pixel dataset root.
set -e
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
ds=$1; gpu=$2
case "$ds" in
  caltech101)   img=caltech-101_images;   nc=101; fid=caltech-101-fid_stats.npz;          fdd=caltech-101-fd_dino-vitb14_stats.npz ;;
  artbench10)   img=artbench-10_images;   nc=10;  fid=artbench-10_processed-fid_stats.npz; fdd=artbench-10-fd_dino-vitb14_stats.npz ;;
  cub200)       img=cub-200-2011_images;  nc=200; fid=cub-200-2011_processed-fid_stats.npz; fdd=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      img=food-101_images;      nc=101; fid=food-101_processed-fid_stats.npz;    fdd=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) img=stanford-cars_images; nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz ;;
  *) echo "unknown $ds"; exit 1 ;;
esac
root="$DATA/$img"; [ -d "$root/train" ] || { echo "ERROR $root/train missing"; exit 2; }
CONFIG="$REPO/configs/caltech_jit_dmf_meft_config.yml"
WORKDIR="$REPO/files/workdirs/jitmeft_${ds}"
runsh="$REPO/files/logs/run_jitmeft_${ds}.sh"
mkdir -p "$REPO/files/logs" "$WORKDIR"
cat > "$runsh" <<EOF
#!/bin/bash
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export MPLCONFIGDIR=/tmp/\$USER-matplotlib
.venv/bin/python main_imf_jit.py \\
  --workdir=$WORKDIR \\
  --config=$CONFIG \\
  --config.dataset.root=$root \\
  --config.dataset.num_classes=$nc \\
  --config.model.num_classes=$nc \\
  --config.fid.cache_ref=$REPO/files/fid_stats/$fid \\
  --config.fd_dino.cache_ref=$REPO/files/fdd_stats/$fdd \\
  --config.load_from=$REPO/files/weights/JiT-H-16-256.pth \\
  --config.logging.use_wandb=True \\
  --config.logging.wandb_project=jit_meft \\
  --config.logging.wandb_entity=ea-fc \\
  --config.logging.wandb_group=jit_meft_20260722 \\
  --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \\
  --config.training.max_train_steps=30000
EOF
chmod +x "$runsh"
screen -dmS "jitmeft_${ds}" bash -c "bash $runsh 2>&1 | tee -a $REPO/files/logs/jitmeft_${ds}_run.log"
echo "launched JiT MeFT $ds on GPU $gpu (screen jitmeft_${ds})"
