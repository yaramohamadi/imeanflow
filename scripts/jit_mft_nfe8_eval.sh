#!/usr/bin/env bash
# JiT MF-T at 8 NFE, 10k samples: same command as scripts/redo10k_watcher.sh (jitdmf family; Tab. 2 JiT MF-T ckpts
# files/workdirs/jitdmfclip_<ds>/best_fid), only force_metric_num_steps changed to 8. Output: files/logs/redo10k/jitdmf_<ds>_nfe8.
# Usage: bash scripts/jit_mft_nfe8_eval.sh <gpu> <dataset> [<dataset> ...]   (runs the datasets one after another)
set -uo pipefail
gpu=$1; shift
REPO=/opt/dlami/nvme/meanflow/imeanflow; DATA=/opt/dlami/nvme/meanflow/datasets; PY=$REPO/.venv/bin/python; NS=10000
declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A IMG=( [caltech101]=caltech-101_images [artbench10]=artbench-10_images [cub200]=cub-200-2011_images [food101]=food-101_images [stanfordcars]=stanford-cars_images )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )
cd $REPO
for ds in "$@"; do
  ck=$(ls -d $REPO/files/workdirs/jitdmfclip_${ds}/best_fid/checkpoint_* 2>/dev/null | tail -1)
  [ -n "$ck" ] || { echo "ERROR no ckpt for $ds"; continue; }
  out=$REPO/files/logs/redo10k/jitdmf_${ds}_nfe8; mkdir -p $out
  [ -f $out/eval_metrics.csv ] && { echo "SKIP $ds (done)"; continue; }
  echo "EVAL $ds NFE 8 ckpt=$(basename $ck)" | tee -a $out/run.log
  CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/mpl-nfe8-$ds \
  $PY main_imf_jit.py --workdir=$out --config=configs/load_config.py:caltech_jit_dmf_meft --config.eval_only=True \
    --config.load_from=$ck --config.dataset.root=$DATA/${IMG[$ds]} --config.dataset.num_classes=${NC[$ds]} \
    --config.model.num_classes=${NC[$ds]} --config.fid.cache_ref=$REPO/files/fid_stats/${FID[$ds]} \
    --config.fd_dino.cache_ref=$REPO/files/fdd_stats/${FDD[$ds]} --config.logging.use_wandb=False \
    --config.training.force_metric_num_steps='8' --config.fid.num_samples=$NS 2>&1 | tee -a $out/run.log
done
echo "=== NFE8 QUEUE DONE ==="
