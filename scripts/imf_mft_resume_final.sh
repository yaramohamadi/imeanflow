#!/usr/bin/env bash
# Final-checkpoint numbers for iMF MF-T (reviewer: "final-checkpoint or held-out-selected numbers for the iMF and SiT blocks").
# The original MF-T runs saved only best_fid (with optimizer state) and trained to 40K steps. This resumes from best_fid
# with the run's own dumped config (configs/load_dump.py), trains to 40K, and saves the final checkpoint. Then it evaluates
# BOTH the best and the final checkpoint at 1 and 4 NFE with 10k samples (same eval as the other iMF evals).
# Caveat: the resumed steps after best_fid are a reconstruction (data order / RNG differ from the lost original tail).
# Usage: bash scripts/imf_mft_resume_final.sh <orig_run_dir_name> <dataset> <gpu>
set -uo pipefail
run=$1; ds=$2; gpu=$3
M=/opt/dlami/nvme/meanflow/imeanflow; D=/opt/dlami/nvme/meanflow/datasets; F=$M/files/logs/finetuning; PY=$M/.venv/bin/python
case "$ds" in
  artbench10)   lat=artbench-10_processed_latents;   fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   lat=caltech-101_processed_latents;   fid=caltech-101-fid_stats.npz;             fdd=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       lat=cub-200-2011_processed_latents;  fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      lat=food-101_processed_latents;      fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) lat=stanford-cars_processed_latents; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz ;;
  *) echo "unknown $ds"; exit 1 ;;
esac
cd $M
src=$F/$run; best=$(ls -d $src/best_fid/checkpoint_* | sort -V | tail -1)
wd=$F/${run}_resume_final; mkdir -p $wd
log=$wd/resume.log
awk '/FLAGS.config:/{f=1;next} f&&/^[IWE][0-9]{4} /{exit} f&&/^wandb:/{next} f' $src/output.log > $wd/orig_config.yml
[ -s $wd/orig_config.yml ] || { echo "ERROR empty config dump" | tee -a $log; exit 1; }
# Default XLA preallocation, as the original runs: with PREALLOCATE=false the first train step OOMs (32.8 GB alloc).
export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
echo "RESUME $best -> 40000 in $wd" | tee -a $log
$PY main.py --workdir=$wd --config=configs/load_dump.py:$wd/orig_config.yml \
  --config.load_from=$best --config.partial_load=False \
  --config.training.save_best_fid_only=False --config.training.checkpoint_per_epoch=1000000 \
  --config.logging.use_wandb=False 2>&1 | tee -a $log
final=$(ls -d $wd/checkpoint_* 2>/dev/null | sort -V | tail -1)
[ -n "$final" ] || { echo "ERROR no final checkpoint saved" | tee -a $log; exit 1; }
echo "FINAL $final" | tee -a $log
# evaluate best and final at 1 and 4 NFE, 10k samples (the eval script needs <dir>/best_fid/<one checkpoint>)
for tag in best final; do
  ck=$([ $tag = best ] && echo $best || echo $final); ed=$F/${run}_eval10k_$tag; mkdir -p $ed/best_fid
  [ -e $ed/best_fid/$(basename $ck) ] || cp -al $ck $ed/best_fid/
  CONFIG_MODE=plain_imf_finetune PYTHON=$PY USE_WANDB=False bash scripts/eval_best_fid_steps_plain_imf.sh $ed 1 4 -- \
    --config.dataset.name=$ds --config.dataset.root=$D/$lat \
    --config.fid.cache_ref=$M/files/fid_stats/$fid --config.fd_dino.cache_ref=$M/files/fdd_stats/$fdd 2>&1 | tee -a $log
done
echo "=== RESUME + EVAL DONE ===" | tee -a $log
