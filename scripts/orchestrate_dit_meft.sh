#!/usr/bin/env bash
# =============================================================================
# Autonomous orchestrator for 5 DiT-XL MeFT (ddpm-v, DogFit OFF) runs.
# - Launches each dataset on a free GPU from the pool (0,1,6,7 usable; 2-5 busy).
# - Auto-extracts a dataset's latent zip (handles ETS-nested prefix) when ready.
# - Monitors screens; on crash (screen gone but no best_fid), retries up to 2x.
# - Launches queued datasets as GPUs free up.
# - Writes STATUS file each loop for external polling.
# Runs detached (nohup). Safe to re-run: skips datasets that already finished.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
STATUS=$REPO/files/logs/ditmeft_ORCH_STATUS.txt
ORCHLOG=$REPO/files/logs/ditmeft_ORCH.log
cd "$REPO"
mkdir -p files/logs

GPU_POOL=(0 1 6 7)          # usable GPUs (2-5 occupied by another job)
DIT=$REPO/files/weights/DiT-XL-2-256x256.pt

# dataset : num_classes : fid_basename : fdd_basename
declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A LAT=( [caltech101]=caltech-101_processed_latents [artbench10]=artbench-10_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )
ALL=(caltech101 artbench10 cub200 food101 stanfordcars)

declare -A RETRIES=()   # per-dataset retry count
for d in "${ALL[@]}"; do RETRIES[$d]=0; done

log(){ echo "$(cat /proc/uptime|cut -d' ' -f1) $*" >> "$ORCHLOG"; }

# ensure /dev/shm big
if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || true
fi

dataset_ready(){ local ds=$1; local t="$DATA/${LAT[$ds]}/train"; [[ -d "$t" ]] && [[ -n "$(find "$t" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]]; }

try_extract(){ # extract a latent zip if the zip is present & complete; handle ETS-nested prefix
  local ds=$1 zip="$DATA/${LAT[$ds]}.zip"
  dataset_ready "$ds" && return 0
  [[ -f "$zip" ]] || return 1
  # verify zip is a valid complete archive
  "$REPO/.venv/bin/python" -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$zip') else 1)" || return 1
  log "extracting $ds"
  "$REPO/.venv/bin/python" - "$zip" "$DATA/${LAT[$ds]}" <<'PY'
import zipfile,os,shutil,sys
zp,dest=sys.argv[1],sys.argv[2]
tmp=dest+"_tmpx"; shutil.rmtree(tmp,ignore_errors=True)
zipfile.ZipFile(zp).extractall(tmp)
src=None
base=os.path.basename(dest)
for root,dirs,files in os.walk(tmp):
    if os.path.basename(root)==base and os.path.isdir(os.path.join(root,"train")):
        src=root; break
if src is None:  # maybe flat (train/ directly under tmp)
    if os.path.isdir(os.path.join(tmp,"train")): src=tmp
if src and src!=dest:
    shutil.rmtree(dest,ignore_errors=True); shutil.move(src,dest)
shutil.rmtree(tmp,ignore_errors=True)
PY
  dataset_ready "$ds"
}

free_gpus(){ # echo free GPUs from pool (mem < 2000 MiB)
  for g in "${GPU_POOL[@]}"; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ -n "$m" && "$m" -lt 2000 ]] && echo "$g"
  done
}

run_done(){ # best_fid checkpoint exists => finished
  local ds=$1
  ls -d "$REPO"/files/logs/finetuning/*DiT_DMF*"${ds}"*/best_fid/checkpoint_* 2>/dev/null | head -1 | grep -q . \
    || ls -d "$REPO"/files/logs/finetuning/*"${ds}"*DiT_DMF*/best_fid/checkpoint_* 2>/dev/null | head -1 | grep -q .
}

launch(){ # ds gpu
  local ds=$1 gpu=$2
  local root="$DATA/${LAT[$ds]}"
  local fid="$REPO/files/fid_stats/${FID[$ds]}" fdd="$REPO/files/fdd_stats/${FDD[$ds]}"
  local sess="ditmeft_g${gpu}_${ds}"
  log "LAUNCH $ds on GPU $gpu (screen $sess) retry=${RETRIES[$ds]}"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu
    export DATASET_NAME=$ds DATASET_ROOT='$root' DATASET_NUM_CLASSES=${NC[$ds]}
    export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd'
    export PYTHON='$REPO/.venv/bin/python'
    export ENABLE_DOGFIT=False
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    export USE_WANDB=True
    export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False
    bash scripts/run_caltech_dit_dmf_ddpmv_taylor.sh meft_online \
      --config.load_from='$DIT' \
      --config.logging.wandb_project='dit_meft' \
      --config.logging.wandb_entity='ea-fc' \
      --config.logging.wandb_group='dit_meft_ddpmv_20260722' \
      --config.dataset.name=$ds \
      --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
      --config.training.max_train_steps=30000 \
      --config.fid.num_samples=5000 \
      --config.training.force_fid_per_step=2500 \
      --config.training.debug_log_during_train=False \
      2>&1 | tee -a $REPO/files/logs/ditmeft_${ds}_run.log
  "
}

screen_alive(){ screen -ls 2>/dev/null | grep -q "ditmeft_g.*_$1\b"; }

log "=== orchestrator start ==="
while true; do
  # write status
  {
    echo "DiT MeFT orchestrator  (uptime $(cat /proc/uptime|cut -d' ' -f1)s)"
    for d in "${ALL[@]}"; do
      if run_done "$d"; then st="DONE";
      elif screen_alive "$d"; then st="RUNNING";
      elif dataset_ready "$d"; then st="ready/idle";
      else st="waiting-data"; fi
      printf "  %-14s %s (retries %s)\n" "$d" "$st" "${RETRIES[$d]}"
    done
    echo "free GPUs: $(free_gpus | tr '\n' ' ')"
  } > "$STATUS"

  # all done?
  alldone=1; for d in "${ALL[@]}"; do run_done "$d" || alldone=0; done
  if [[ $alldone -eq 1 ]]; then log "ALL 5 DONE"; echo "ALL DONE $(date -u +%FT%TZ 2>/dev/null || echo)" >> "$STATUS"; break; fi

  # detect crashed runs (screen gone, not done, was launched) -> retry
  for d in "${ALL[@]}"; do
    if [[ -f "$REPO/files/logs/ditmeft_${d}_run.log" ]] && ! screen_alive "$d" && ! run_done "$d"; then
      # crashed or finished-without-bestfid; retry up to 2
      if [[ ${RETRIES[$d]} -lt 2 ]]; then
        # only retry if it actually stopped (give a grace file marker)
        RETRIES[$d]=$(( ${RETRIES[$d]} + 1 ))
        log "RETRY $d (crashed, retry ${RETRIES[$d]})"
        # will be relaunched below when a GPU is free
      fi
    fi
  done

  # launch ready+idle datasets on free GPUs
  for g in $(free_gpus); do
    for d in "${ALL[@]}"; do
      run_done "$d" && continue
      screen_alive "$d" && continue
      if ! dataset_ready "$d"; then try_extract "$d" >/dev/null 2>&1 || true; fi
      dataset_ready "$d" || continue
      # don't exceed retry cap
      [[ ${RETRIES[$d]} -le 2 ]] || continue
      launch "$d" "$g"
      sleep 20   # stagger so GPU mem registers before next free_gpus poll
      break
    done
  done

  sleep 120
done
log "=== orchestrator exit ==="
