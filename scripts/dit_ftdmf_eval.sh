#!/usr/bin/env bash
# Evaluate one FT -> DMF DiT run exactly as Tab. 2's DiT MF-T runs were evaluated (scripts/diteval_clip_watcher.sh):
# eval_best_fid_steps.sh with TARGET_USE_NULL_CLASS=True, CLASS_DROPOUT_PROB=0.1, noise output + dit_native map, 10K samples.
# (The run script's built-in final eval forces TARGET_USE_NULL_CLASS=False and crashes on the 197-row class table.)
# Waits until the run's training screen (ftdmf_dit_<ds>) is gone, then evaluates the best checkpoint at NFE 1 2 4 8.
# Usage: bash scripts/dit_ftdmf_eval.sh <dataset> <gpu>
set -uo pipefail
ds=$1; gpu=$2
REPO=/opt/dlami/nvme/meanflow/imeanflow; PY=$REPO/.venv/bin/python; cd $REPO
declare -A NC=([artbench10]=10 [caltech101]=101 [cub200]=200 [food101]=101 [stanfordcars]=196)
declare -A LAT=([artbench10]=artbench-10_processed_latents [caltech101]=caltech-101_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents)
declare -A FID=([artbench10]=artbench-10_processed-fid_stats.npz [caltech101]=caltech-101-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz)
declare -A FDD=([artbench10]=artbench-10-fd_dino-vitb14_stats.npz [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz)
while screen -ls | grep -qE "\.ftdmf_dit_${ds}[[:space:]]"; do sleep 60; done
rd=$(ls -d files/logs/finetuning/${ds}_DiT_DMF_meanflow_taylor_*ftdmf_${ds}_* | head -1)
[[ -d "$rd/best_fid" ]] || { echo "no best_fid in $rd"; exit 1; }
echo "[$(date '+%F %T')] eval $rd on GPU $gpu"
CONFIG_MODE=caltech_dit_dmf_ddpmv PYTHON=$PY USE_WANDB=False MODEL_STR=imfDiT_DMF_XL_2 MODEL_USE_DOGFIT=False \
TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 TARGET_OUTPUT_PREDICTION_SPACE=noise TARGET_VELOCITY_MAP_MODE=dit_native \
USE_EMA_VC=False DATASET_ROOT=$REPO/../datasets/${LAT[$ds]} DATASET_NUM_CLASSES=${NC[$ds]} \
FID_CACHE_REF=$REPO/files/fid_stats/${FID[$ds]} FD_DINO_CACHE_REF=$REPO/files/fdd_stats/${FDD[$ds]} \
CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
  bash scripts/eval_best_fid_steps.sh "$rd" 1 2 4 8 2>&1 | tee -a $REPO/files/logs/ftdmf_dit_${ds}_eval.log
ls $rd/eval_best_fid_8steps/eval_metrics.csv >/dev/null 2>&1 && touch $REPO/files/logs/ftdmf_dit_${ds}.evaldone
echo "[$(date '+%F %T')] eval finished for $ds"
