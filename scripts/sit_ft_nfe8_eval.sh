#!/usr/bin/env bash
# =============================================================================
# SiT FT (DDPM-e, DiT-init) NFE-8 final eval, all 5 datasets. Detached screens
# sitft8_gpuN_<ds>. One GPU per dataset (0-4); all fire in parallel.
#
# WHY: the SiT FT block in the main table has @250x2/@4/@1 but no @8 row (the
# launchers used FINAL_EVAL_STEPS="1 2 4 250" -- 8 was never evaluated). This
# replays the SAME proven eval script (eval_best_fid_steps_plain_sit.sh, which
# forces use_ema=False -> online metric mode) at NFE 8 only, byte-matching the
# recipe that produced the existing @4 cell -- only num_steps differs.
#
# Sources = the current best_fid checkpoint of each run (caltech@17500,
# food@20000 -- stopped mid-train at 75% per user; artbench/cub/stanford@27500
# fully trained). 10k FID samples (standing rule). Output -> eval_best_fid_8steps/
# inside each run dir (no clobber of existing 1/2/4/250 dirs).
# =============================================================================
set -uo pipefail
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
DITW=$MAIN/files/weights/DiT-XL-2-256x256.pt
CONFIG_MODE=caltech_plain_sit_ddpme
NSAMP=10000
STEPS=8
cd "$MAIN"

# ds gpu rundir dsname nclasses root fidnpz fddnpz
run(){
  local gpu="$1" rd="$2" dsname="$3" nc="$4" root="$5" fid="$6" fdd="$7"
  local ds="${rd%%_*}"
  local sess="sitft8_gpu${gpu}_${ds}"
  local wd="$MAIN/files/logs/finetuning/$rd"
  if [[ ! -d "$wd/best_fid" ]]; then echo "SKIP $ds: no best_fid ($wd)"; return; fi
  echo "LAUNCH $sess  src=$rd  (NFE8, 10k)"
  screen -dmS "$sess" bash -c "
    cd $MAIN
    CONFIG_MODE=$CONFIG_MODE PYTHON=$PY USE_WANDB=False \
    CUDA_VISIBLE_DEVICES=$gpu \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash scripts/eval_best_fid_steps_plain_sit.sh \"files/logs/finetuning/$rd\" $STEPS \
      -- --config.load_from=$DITW \
         --config.dataset.num_classes=$nc \
         --config.model.num_classes=$nc \
         --config.dataset.name=$dsname \
         --config.dataset.root=$MAIN/../datasets/$root \
         --config.fid.num_samples=$NSAMP \
         --config.fid.cache_ref=$MAIN/files/fid_stats/$fid \
         --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/$fdd \
      2>&1 | tee -a $MAIN/files/logs/sitft8_${ds}.log
  "
}

run 0 artbench_plain_SiT_ddpme_taylor_20260723_184910_w8dph4 artbench-10   10  artbench-10_processed_latents  artbench-10_processed-fid_stats.npz   artbench-10-fd_dino-vitb14_stats.npz
run 1 caltech_plain_SiT_ddpme_taylor_20260729_000702_uazi51  caltech-101  101 caltech-101_processed_latents   caltech-101-fid_stats.npz             caltech-101-fd_dino-vitb14_stats.npz
run 2 cub_plain_SiT_ddpme_taylor_20260723_184910_z384sy      cub-200-2011 200 cub-200-2011_processed_latents  cub-200-2011_processed-fid_stats.npz  cub-200-2011-fd_dino-vitb14_stats.npz
run 3 food_plain_SiT_ddpme_taylor_20260729_000702_eka4et     food-101     101 food-101_processed_latents      food-101_processed-fid_stats.npz      food-101-fd_dino-vitb14_stats.npz
run 4 stanford_plain_SiT_ddpme_taylor_20260723_184910_lkzwvw stanford-cars 196 stanford-cars_processed_latents stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz

echo "=== all launched ==="; screen -ls | grep sitft8_
