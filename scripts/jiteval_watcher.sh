#!/usr/bin/env bash
# =============================================================================
# JiT DMF (MF-A) post-hoc eval watcher -- detached (screen jiteval_watcher).
#
# main_imf_jit.py now supports --config.eval_only=True (just_evaluate, verified:
# NFE=4 reproduces the during-training best_fid FID exactly). This watcher, for
# each of the 5 jitdmfclip_<ds> runs, waits until (a) its training screen is gone
# AND (b) best_fid/checkpoint_* exists, then runs eval_only at NFE "1 2 4 50" on
# any stably-free GPU. Writes eval_only rows into the run's own workdir CSV.
#
# Two gates per dataset: TRAINING DONE (screen jitdmf_gpu*_<ds> absent) + GPU free.
# Screens jiteval_gpu<N>_<ds>. Idempotent-ish: skips a ds whose eval CSV already
# has an eval_only row.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
LOG="$REPO/files/logs/jiteval_watcher.log"
PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python

POLL_S=90
STABLE_NEEDED=2
MEM_MB=3000
UTIL_PCT=15
STEPS="1 2 4 50"
ALLOWED="0 1 2 3 4 5 6 7"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ds : imgbase : nc : fid_npz : fdd_npz
QUEUE=(
  "caltech101:caltech-101_images:101:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz"
  "artbench10:artbench-10_images:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz"
  "cub200:cub-200-2011_images:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz"
  "food101:food-101_images:101:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz"
  "stanfordcars:stanford-cars_images:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz"
)

# training-done test: no jitdmf_gpu*_<ds> screen AND a best_fid checkpoint exists
train_done(){  # $1=ds
  local ds="$1"
  local wd="$REPO/files/workdirs/jitdmfclip_${ds}"
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "jitdmf_gpu[0-9]+_${ds}\b" && return 1
  return 0
}

# already-evaluated test: eval_only row present in the run CSV
already_evaled(){  # $1=ds
  local ds="$1"
  local csv="$REPO/files/workdirs/jitdmfclip_${ds}/eval_metrics.csv"
  [[ -f "$csv" ]] && grep -q "^eval_only," "$csv"
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds img nc fid fdd <<< "$2"
  local wd="$REPO/files/workdirs/jitdmfclip_${ds}"
  local ckpt
  ckpt=$(ls -d "$wd/best_fid/checkpoint_"* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  [[ -n "$ckpt" ]] || { log "ERROR [$ds] no best_fid checkpoint"; return 1; }
  [[ -e "$REPO/files/fid_stats/$fid" ]] || { log "ERROR [$ds] missing fid $fid"; return 1; }
  [[ -e "$REPO/files/fdd_stats/$fdd" ]] || { log "ERROR [$ds] missing fdd $fdd"; return 1; }
  local sess="jiteval_gpu${gpu}_${ds}"
  log "LAUNCH eval $ds on GPU $gpu (screen $sess) ckpt=$ckpt steps=[$STEPS]"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/mpl-jiteval-$ds
    $PY main_imf_jit.py \
      --workdir=$wd \
      --config=configs/load_config.py:caltech_jit_dmf_meft \
      --config.eval_only=True \
      --config.load_from=$ckpt \
      --config.dataset.root=$DATA/$img \
      --config.dataset.num_classes=$nc \
      --config.model.num_classes=$nc \
      --config.fid.cache_ref=$REPO/files/fid_stats/$fid \
      --config.fd_dino.cache_ref=$REPO/files/fdd_stats/$fdd \
      --config.logging.use_wandb=False \
      --config.training.force_metric_num_steps=\"$STEPS\" \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/jiteval_${ds}_eval.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== JiT DMF eval watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, steps [$STEPS], gate $STABLE_NEEDED ==="

# mark already-evaluated datasets as done up front
for entry in "${QUEUE[@]}"; do
  ds="${entry%%:*}"
  if already_evaled "$ds"; then DONE[$ds]=1; log "SKIP $ds (eval_only row already present)"; fi
done

remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  # detect finished eval screens -> release the GPU they held
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if ! screen -ls | grep -qE "jiteval_gpu${idx}_${ds}\b"; then
      if already_evaled "$ds"; then DONE[$ds]=1; log "COMPLETE $ds eval (GPU $idx freed)"; else log "WARN $ds eval screen gone but no eval_only CSV row"; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # assign a free GPU to the next training-done, not-yet-evaled, not-in-flight ds
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      # skip if this ds already has an in-flight eval on another GPU
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      train_done "$ds" || continue
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx; remaining $(remaining)"; sleep 15; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all JiT DMF evals complete; watcher exiting ==="
