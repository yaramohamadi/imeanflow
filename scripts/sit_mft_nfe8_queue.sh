#!/usr/bin/env bash
# SiT MF-T 8-NFE 10k evals (Tab. 2 best_fid checkpoints), to complete the SiT curve of the 1- vs 2-stage
# ratio figure. Same env as the redo10k SiT evals (scripts/redo10k_watcher.sh, sit branch), NFE 8.
# One job per free GPU in POOL (free = <3 GB for 3 polls); never grabs a busy GPU. EXP-372..376.
set -uo pipefail
M=/opt/dlami/nvme/meanflow/imeanflow; D=/opt/dlami/nvme/meanflow/datasets; F=$M/files/logs/finetuning; PY=$M/.venv/bin/python
cd $M; LOG=$M/files/logs/sit_mft_nfe8_queue.log; POOL="${POOL:-0 1 2 3 4 5 6 7}"
declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A LAT=( [caltech101]=caltech-101_processed_latents [artbench10]=artbench-10_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )
JOBS=( "artbench10 artbench10_SiT_DMF_plain_meanflow_meft_online_20260722_172435_pe502i EXP-372"
       "caltech101 caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w EXP-373"
       "cub200 cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8 EXP-374"
       "food101 food101_SiT_DMF_plain_meanflow_meft_online_20260722_232709_tv7xmz EXP-375"
       "stanfordcars stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo EXP-376" )
log(){ echo "[$(date '+%F %T %Z')] $*" | tee -a $LOG; }
declare -A FREE; declare -A HELD
log "=== sit_mft_nfe8_queue: ${#JOBS[@]} jobs, pool {$POOL} ==="
i=0
while (( i < ${#JOBS[@]} )); do
  for g in $POOL; do
    (( i < ${#JOBS[@]} )) || break
    if [ -n "${HELD[$g]:-}" ] && screen -ls | grep -q "\.${HELD[$g]}\b"; then FREE[$g]=0; continue; fi
    HELD[$g]=""
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
    if (( used < 3000 )); then FREE[$g]=$(( ${FREE[$g]:-0} + 1 )); else FREE[$g]=0; fi
    if (( ${FREE[$g]} >= 3 )); then
      read -r ds rd exp <<< "${JOBS[$i]}"; sess="sitnfe8_${ds}"
      screen -dmS $sess bash -c "cd $M; CONFIG_MODE=caltech_sit_dmf_finetune PYTHON=$PY USE_WANDB=False \
        MODEL_STR=imfSiT_DMF_XL_2 MODEL_USE_DOGFIT=False TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 \
        DATASET_ROOT=$D/${LAT[$ds]} DATASET_NUM_CLASSES=${NC[$ds]} \
        FID_CACHE_REF=$M/files/fid_stats/${FID[$ds]} FD_DINO_CACHE_REF=$M/files/fdd_stats/${FDD[$ds]} FID_NUM_SAMPLES=10000 \
        CUDA_VISIBLE_DEVICES=$g TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
        bash scripts/eval_best_fid_steps.sh $F/$rd 8 > $M/files/logs/${sess}.log 2>&1"
      log "LAUNCH $exp $ds NFE8 on GPU$g (screen $sess)"; HELD[$g]=$sess; FREE[$g]=0; i=$((i+1))
    fi
  done
  sleep 60
done
log "=== all ${#JOBS[@]} jobs launched ==="
