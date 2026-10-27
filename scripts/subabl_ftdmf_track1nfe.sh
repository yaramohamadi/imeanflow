#!/usr/bin/env bash
# Rerun of the subset FT->DMF stage (scripts/subabl_ftdmf_stage2.sh) with 1-NFE FID/FDD logged every 2500 steps
# next to 4-NFE (--config.training.force_metric_num_steps="4 1"). Best checkpoint is still chosen by 4-step FID.
# Question: is FT->DMF's weak 1-NFE result a checkpoint-selection effect? Uses the existing converted FT checkpoint.
# Usage: bash scripts/subabl_ftdmf_track1nfe.sh <cub200|stanfordcars> <10|25> <gpu>
set -uo pipefail
ds=$1; p=$2; gpu=$3
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
PY=$REPO/.venv/bin/python
case "$ds" in
  cub200)       lat=cub-200-2011_processed_latents;  nc=200; full_spe=737;  fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  stanfordcars) lat=stanford-cars_processed_latents; nc=196; full_spe=1008; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz ;;
  *) echo "unknown $ds"; exit 1 ;;
esac
cd $REPO
tag=sub${p}pct_s42; sub=$DATA/${lat}_${p}pct_seed42; init=$REPO/files/weights/sit_ft_subset/${ds}_${p}pct.pt
[ -f "$init" ] || { echo "ERROR missing $init"; exit 1; }
log=$REPO/files/logs/subabl_ftdmf1nfe_${tag}_${ds}.log
n=$(ls $sub/train | grep -c '\.pt$'); spe=$((n / 16))
w_mf=$(awk "BEGIN{printf \"%.6f\", (int(5*$full_spe)+0.5)/$spe}")
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$REPO/files/wandb_offline
export DATASET_ROOT=$sub DATASET_NUM_CLASSES=$nc PYTHON=$PY TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export FID_CACHE_REF=$REPO/files/fid_stats/$fid FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd
export LOAD_FROM=$init USE_WANDB=True FINAL_EVAL_USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8'
export WANDB_PROJECT=sit_ft_dmf
bash scripts/train_sit_dmf_no_dogfit_dataset.sh $ds ftdmf1nfe_$tag \
  --config.load_from=$init --config.logging.wandb_project=sit_ft_dmf \
  --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
  --config.training.max_train_steps=30000 --config.training.num_epochs=100000 --config.training.warmup_epochs=$w_mf \
  --config.fid.num_samples=5000 --config.training.force_fid_per_step=2500 \
  --config.training.debug_log_during_train=False \
  "--config.training.force_metric_num_steps=4 1" \
  2>&1 | tee -a $log
