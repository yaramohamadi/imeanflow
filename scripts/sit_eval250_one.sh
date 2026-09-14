#!/usr/bin/env bash
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
ds=$1; gpu=$2
case "$ds" in
  caltech101)   nc=101; fid=caltech-101-fid_stats.npz;            fdd=caltech-101-fd_dino-vitb14_stats.npz;   lat=caltech-101_processed_latents ;;
  artbench10)   nc=10;  fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz;   lat=artbench-10_processed_latents ;;
  cub200)       nc=200; fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz;  lat=cub-200-2011_processed_latents ;;
  food101)      nc=101; fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz;      lat=food-101_processed_latents ;;
  stanfordcars) nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz; lat=stanford-cars_processed_latents ;;
  *) echo "unknown $ds"; exit 1;;
esac
rd=$(ls -dt files/logs/finetuning/*"$ds"*SiT*meft_online*/ 2>/dev/null | grep -v smoke | head -1); rd=${rd%/}
[ -z "$rd" ] && { echo "no run dir $ds"; exit 1; }
[ -f "$rd/eval_best_fid_250steps/eval_metrics.csv" ] && { echo "already done $ds"; exit 0; }
root="$DATA/$lat"
sh="files/logs/run_sit250_${ds}.sh"
cat > "$sh" <<EOF
#!/bin/bash
cd $REPO
CONFIG_MODE=caltech_sit_dmf_finetune PYTHON=$REPO/.venv/bin/python USE_WANDB=False \\
MODEL_STR=imfSiT_DMF_XL_2 MODEL_USE_DOGFIT=False TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 \\
DATASET_ROOT=$root DATASET_NUM_CLASSES=$nc \\
FID_CACHE_REF=$REPO/files/fid_stats/$fid FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd \\
CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \\
bash scripts/eval_best_fid_steps.sh "$rd" 250
EOF
chmod +x "$sh"
screen -dmS "sit250_${ds}" bash -c "bash $sh 2>&1 | tee -a $REPO/files/logs/sit250_${ds}_run.log"
echo "LAUNCHED $ds on GPU $gpu (screen sit250_${ds})"
