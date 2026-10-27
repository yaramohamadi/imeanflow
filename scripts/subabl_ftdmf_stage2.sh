#!/usr/bin/env bash
# Stage 2 of the SiT subset ablation (scripts/launch_sit_subset_ablation.sh): convert the subset FT best-FID
# checkpoint to torch layout, then run DMF from it. Same DMF arguments as the launcher.
# Needed because the launcher's chained conversion ran JAX without JAX_PLATFORMS=cpu and crashed (no visible GPU).
# Waits until the FT screen subabl_ft_<ds>_<p> has exited (FT + its final evals finished).
# Usage: bash scripts/subabl_ftdmf_stage2.sh <cub200|stanfordcars> <10|25> <gpu>
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
log=$REPO/files/logs/subabl_ftdmf_${tag}_${ds}.log
while screen -ls | grep -q "\.subabl_ft_${ds}_${p}\b"; do sleep 60; done
n=$(ls $sub/train | grep -c '\.pt$'); spe=$((n / 16))
w_mf=$(awk "BEGIN{printf \"%.6f\", (int(5*$full_spe)+0.5)/$spe}")
ftdir=$(ls -dt files/logs/finetuning/plain_SiT_finetune_${ds}_${tag}_* | head -1)
ck=$(ls -d $ftdir/best_fid/checkpoint_* | sort -V | tail -1)
[ -n "$ck" ] || { echo "ERROR no FT best_fid checkpoint" | tee -a $log; exit 1; }
echo "CONVERT $ck -> $init (warmup_epochs $w_mf, $n images)" | tee -a $log
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY scripts/ft_sit_to_torch.py $ck $init --check 2>&1 | tee -a $log
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "ERROR conversion failed" | tee -a $log; exit 1; }
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$REPO/files/wandb_offline
export DATASET_ROOT=$sub DATASET_NUM_CLASSES=$nc PYTHON=$PY TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export FID_CACHE_REF=$REPO/files/fid_stats/$fid FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd
export LOAD_FROM=$init USE_WANDB=True FINAL_EVAL_USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8'
export WANDB_PROJECT=sit_ft_dmf
bash scripts/train_sit_dmf_no_dogfit_dataset.sh $ds ftdmf_$tag \
  --config.load_from=$init --config.logging.wandb_project=sit_ft_dmf \
  --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
  --config.training.max_train_steps=30000 --config.training.num_epochs=100000 --config.training.warmup_epochs=$w_mf \
  --config.fid.num_samples=5000 --config.training.force_fid_per_step=2500 \
  --config.training.debug_log_during_train=False \
  2>&1 | tee -a $log
