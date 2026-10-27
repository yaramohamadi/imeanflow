#!/usr/bin/env bash
# Final-checkpoint numbers for the SiT block (reviewer 8oSC: "final-checkpoint or held-out-selected
# numbers for at least the iMF and SiT blocks"). Two modes:
#   mft: SiT MF-T runs saved only best_fid (with optimizer state). Resume it with the run's own dumped
#        config (configs/load_dump.py) to the run's max_train_steps (30K), save the final checkpoint,
#        then eval BOTH best and final at NFE 1 4, 10k, with the exact redo10k SiT env
#        (eval_best_fid_steps.sh, CONFIG_MODE=caltech_sit_dmf_finetune).
#        Caveat: resumed steps after best_fid are a reconstruction (data order / RNG differ).
#   ft:  SiT FT runs already saved the final step as latest_eval/checkpoint_30000 (full state).
#        Eval it at NFE 4 250, 10k, with the same call as the EXP-303..307 final evals
#        (eval_best_fid_steps_plain_sit.sh, CONFIG_MODE=plain_sit_finetune_1gpu + dataset/fid overrides).
# Usage: bash scripts/sit_final_ckpt.sh <mft|ft> <orig_run_dir_name> <dataset> <gpu>
set -uo pipefail
mode=$1; run=$2; ds=$3; gpu=$4
M=/opt/dlami/nvme/meanflow/imeanflow; D=/opt/dlami/nvme/meanflow/datasets; F=$M/files/logs/finetuning; PY=$M/.venv/bin/python
declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A LAT=( [caltech101]=caltech-101_processed_latents [artbench10]=artbench-10_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )
fid=$M/files/fid_stats/${FID[$ds]}; fdd=$M/files/fdd_stats/${FDD[$ds]}
cd $M
src=$F/$run
export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

eval_mft(){  # $1=checkpoint dir  $2=tag
  local ck=$1 tag=$2 ed=$F/${run}_eval10k_$2; mkdir -p $ed/best_fid
  [ -e $ed/best_fid/$(basename $ck) ] || cp -al $ck $ed/best_fid/
  CONFIG_MODE=caltech_sit_dmf_finetune PYTHON=$PY USE_WANDB=False \
  MODEL_STR=imfSiT_DMF_XL_2 MODEL_USE_DOGFIT=False TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 \
  DATASET_ROOT=$D/${LAT[$ds]} DATASET_NUM_CLASSES=${NC[$ds]} \
  FID_CACHE_REF=$fid FD_DINO_CACHE_REF=$fdd FID_NUM_SAMPLES=10000 \
  bash scripts/eval_best_fid_steps.sh $ed 1 4
}

if [ "$mode" = mft ]; then
  best=$(ls -d $src/best_fid/checkpoint_* | sort -V | tail -1)
  wd=$F/${run}_resume_final; mkdir -p $wd; log=$wd/resume.log
  awk '/FLAGS.config:/{f=1;next} f&&/^[IWE][0-9]{4} /{exit} f&&/^wandb:/{next} f' $src/output.log > $wd/orig_config.yml
  [ -s $wd/orig_config.yml ] || { echo "ERROR empty config dump" | tee -a $log; exit 1; }
  echo "RESUME $best -> max_train_steps in $wd" | tee -a $log
  $PY main.py --workdir=$wd --config=configs/load_dump.py:$wd/orig_config.yml \
    --config.load_from=$best --config.partial_load=False \
    --config.training.save_best_fid_only=False --config.training.checkpoint_per_epoch=1000000 \
    --config.logging.use_wandb=False 2>&1 | tee -a $log
  final=$(ls -d $wd/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -n "$final" ] || { echo "ERROR no final checkpoint saved" | tee -a $log; exit 1; }
  echo "FINAL $final" | tee -a $log
  eval_mft $best best 2>&1 | tee -a $log
  eval_mft $final final 2>&1 | tee -a $log
  echo "=== RESUME + EVAL DONE ===" | tee -a $log
elif [ "$mode" = ft ]; then
  final=$src/latest_eval/checkpoint_30000
  [ -d $final ] || { echo "ERROR no $final"; exit 1; }
  ed=$F/${run}_eval10k_final; mkdir -p $ed/best_fid; log=$ed/eval.log
  [ -e $ed/best_fid/checkpoint_30000 ] || cp -al $final $ed/best_fid/
  CONFIG_MODE=plain_sit_finetune_1gpu PYTHON=$PY USE_WANDB=False \
    bash scripts/eval_best_fid_steps_plain_sit.sh $ed 4 250 -- \
    --config.dataset.root=$D/${LAT[$ds]} --config.fid.cache_ref=$fid --config.fd_dino.cache_ref=$fdd 2>&1 | tee -a $log
  echo "=== FT FINAL EVAL DONE ===" | tee -a $log
else
  echo "unknown mode $mode"; exit 1
fi
