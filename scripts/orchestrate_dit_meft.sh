#!/usr/bin/env bash
# =============================================================================
# Autonomous orchestrator v2 for 5 DiT-XL MeFT (ddpm-v, DogFit OFF) runs.
# Fixes vs v1: internal GPU bookkeeping (no reliance on lagging nvidia-smi mem),
# forgiving flags, crash-retry only for genuinely-stopped screens.
# =============================================================================
set +u
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
STATUS=$REPO/files/logs/ditmeft_ORCH_STATUS.txt
ORCHLOG=$REPO/files/logs/ditmeft_ORCH.log
cd "$REPO"; mkdir -p files/logs

USABLE=(0 1 6 7)   # GPUs 2-5 occupied by another project
BUSY_THRESH=2000   # MiB; a pool GPU above this at startup is externally busy
DIT=$REPO/files/weights/DiT-XL-2-256x256.pt

declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A LAT=( [caltech101]=caltech-101_processed_latents [artbench10]=artbench-10_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )
ALL=(caltech101 artbench10 cub200 food101 stanfordcars)

declare -A RETRIES=(); declare -A GPU_OF=()   # GPU_OF[ds]=gpu currently assigned
for d in "${ALL[@]}"; do RETRIES[$d]=0; GPU_OF[$d]=""; done

log(){ echo "$(cat /proc/uptime|cut -d' ' -f1) $*" >> "$ORCHLOG"; }
if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then mount -o remount,size=64g /dev/shm 2>/dev/null; fi

dataset_ready(){ local t="$DATA/${LAT[$1]}/train"; [[ -d "$t" ]] && [[ -n "$(find "$t" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]]; }

try_extract(){
  local ds=$1 zip="$DATA/${LAT[$ds]}.zip"
  dataset_ready "$ds" && return 0
  [[ -f "$zip" ]] || return 1
  "$REPO/.venv/bin/python" -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$zip') else 1)" 2>/dev/null || return 1
  log "extracting $ds"
  "$REPO/.venv/bin/python" - "$zip" "$DATA/${LAT[$ds]}" <<'PY' 2>/dev/null
import zipfile,os,shutil,sys
zp,dest=sys.argv[1],sys.argv[2]
tmp=dest+"_tmpx"; shutil.rmtree(tmp,ignore_errors=True)
zipfile.ZipFile(zp).extractall(tmp)
base=os.path.basename(dest); src=None
for root,dirs,files in os.walk(tmp):
    if os.path.basename(root)==base and os.path.isdir(os.path.join(root,"train")): src=root; break
if src is None and os.path.isdir(os.path.join(tmp,"train")): src=tmp
if src and src!=dest:
    shutil.rmtree(dest,ignore_errors=True); shutil.move(src,dest)
shutil.rmtree(tmp,ignore_errors=True)
PY
  dataset_ready "$ds"
}

gpu_extern_busy(){ # true if GPU has >threshold mem AND is not one of ours
  local g=$1 m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [[ -n "$m" && "$m" -gt "$BUSY_THRESH" ]]
}
gpu_claimed(){ local g=$1 d; for d in "${ALL[@]}"; do [[ "${GPU_OF[$d]}" == "$g" ]] && return 0; done; return 1; }
pick_gpu(){ # first usable GPU that is neither externally busy nor claimed by us
  local g
  for g in "${USABLE[@]}"; do
    gpu_claimed "$g" && continue
    gpu_extern_busy "$g" && continue
    echo "$g"; return 0
  done
  return 1
}

run_done(){ ls -d "$REPO"/files/logs/finetuning/*"$1"*DiT_DMF*/best_fid/checkpoint_* 2>/dev/null | grep -q . || ls -d "$REPO"/files/logs/finetuning/*DiT_DMF*"$1"*/best_fid/checkpoint_* 2>/dev/null | grep -q .; }
screen_alive(){ screen -ls 2>/dev/null | grep -q "_$1\b" && screen -ls 2>/dev/null | grep "_$1\b" | grep -q ditmeft; }

launch(){
  local ds=$1 gpu=$2 root="$DATA/${LAT[$ds]}"
  local fid="$REPO/files/fid_stats/${FID[$ds]}" fdd="$REPO/files/fdd_stats/${FDD[$ds]}" sess="ditmeft_g${gpu}_${ds}"
  GPU_OF[$ds]=$gpu
  log "LAUNCH $ds on GPU $gpu (retry ${RETRIES[$ds]})"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu
    export DATASET_NAME=$ds DATASET_ROOT='$root' DATASET_NUM_CLASSES=${NC[$ds]}
    export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd' PYTHON='$REPO/.venv/bin/python'
    export ENABLE_DOGFIT=False TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore USE_WANDB=True
    export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False
    bash scripts/run_caltech_dit_dmf_ddpmv_taylor.sh meft_online \
      --config.load_from='$DIT' \
      --config.logging.wandb_project='dit_meft' --config.logging.wandb_entity='ea-fc' \
      --config.logging.wandb_group='dit_meft_ddpmv_20260722' \
      --config.dataset.name=$ds \
      --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
      --config.training.max_train_steps=30000 --config.fid.num_samples=5000 \
      --config.training.force_fid_per_step=2500 --config.training.debug_log_during_train=False \
      2>&1 | tee -a $REPO/files/logs/ditmeft_${ds}_run.log
  "
}

# grace: a freshly-launched run needs time before its screen is 'trusted'; track launch time
declare -A LAUNCHED_AT=()
now(){ cat /proc/uptime | cut -d' ' -f1 | cut -d. -f1; }

log "=== orchestrator v2 start ==="
while true; do
  # STATUS
  { echo "DiT MeFT orchestrator v2 (uptime $(cat /proc/uptime|cut -d' ' -f1)s)"
    for d in "${ALL[@]}"; do
      if run_done "$d"; then st=DONE; elif screen_alive "$d"; then st="RUNNING(gpu ${GPU_OF[$d]})";
      elif dataset_ready "$d"; then st="ready"; else st="waiting-data"; fi
      printf "  %-14s %s (retry %s)\n" "$d" "$st" "${RETRIES[$d]}"
    done
  } > "$STATUS"

  alldone=1; for d in "${ALL[@]}"; do run_done "$d" || alldone=0; done
  [[ $alldone -eq 1 ]] && { log "ALL 5 DONE"; break; }

  # free GPU bookkeeping: if a run's screen died, release its GPU + maybe retry
  for d in "${ALL[@]}"; do
    if [[ -n "${GPU_OF[$d]}" ]] && ! screen_alive "$d"; then
      # grace period: ignore for 90s after launch (screen may not be up yet)
      la=${LAUNCHED_AT[$d]:-0}; [[ $(( $(now) - la )) -lt 90 ]] && continue
      if run_done "$d"; then
        log "$d FINISHED (gpu ${GPU_OF[$d]} released)"; GPU_OF[$d]=""
      else
        log "$d screen gone w/o best_fid (gpu ${GPU_OF[$d]} released)"; GPU_OF[$d]=""
        [[ ${RETRIES[$d]} -lt 2 ]] && RETRIES[$d]=$(( ${RETRIES[$d]} + 1 )) || log "$d hit retry cap"
      fi
    fi
  done

  # launch: one dataset per free GPU, claiming immediately
  for d in "${ALL[@]}"; do
    run_done "$d" && continue
    screen_alive "$d" && continue
    [[ -n "${GPU_OF[$d]}" ]] && continue   # already assigned (mid-launch)
    [[ ${RETRIES[$d]} -le 2 ]] || continue
    dataset_ready "$d" || try_extract "$d" >/dev/null 2>&1 || true
    dataset_ready "$d" || continue
    g=$(pick_gpu) || break
    launch "$d" "$g"; LAUNCHED_AT[$d]=$(now)
    sleep 8
  done

  sleep 90
done
log "=== orchestrator v2 exit ==="
