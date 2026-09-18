#!/usr/bin/env bash
# =============================================================================
# CAIMF (pure-adversarial, lambda_imf=0) SiT-MeFT post-training watcher.
# Detached: screen caimf_sit_watcher.
#
# For each of the 5 SiT DMF MeFT students, runs finite-interval CA-iMF
# adversarial post-training OFF its best_fid checkpoint, on any stably-free GPU.
#
# Config (configs/caltech_sit_meft_caimf_posttrain_config.yml, already patched):
#   lambda_imf=0.0 (PURE adversarial, no iMF retention)  lambda_adv=1.0
#   max_posttrain_batches=150000 (=30k G-updates; warmup 5000 + 4:1 D:G)
#   fid_schedule from 5000 every 5000, metric_num_steps=[4], 10000 samples
#   save_best_fid_only=True
#
# Launcher: scripts/run_sit_meft_adversarial.sh caimf <ds> <ckpt> <workdir>
#   -> derives dev5 dataset/FID/FDD paths itself (DATA_ROOT=$REPO/../datasets),
#      resets optimizer+D state, restores only generator params,
#      then RUN_FINAL_EVAL runs 1/2-step final eval via main.py:just_evaluate.
# We force FINAL_EVAL_STEPS="1 2 4" for the full few-step grid.
#
# Isolation: runs in the ADVERSARIAL clone (imeanflow_adversarial), uses the
# main repo's interpreter (has all deps) via PYTHON=. FID/FDD stats are symlinked
# into the clone. Workdirs are NEW, well-named dirs under the clone's logs.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/caimf_sit_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6          # ~9min stably-free before claiming (don't grab mid train->eval gap)
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2 4"
STAMP=20260725

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ds : best_fid checkpoint (absolute)
QUEUE=(
  "caltech101:$MAIN/files/logs/finetuning/caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w/best_fid/checkpoint_15000"
  "artbench10:$MAIN/files/logs/finetuning/artbench10_SiT_DMF_plain_meanflow_meft_online_20260722_172435_pe502i/best_fid/checkpoint_20000"
  "cub200:$MAIN/files/logs/finetuning/cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8/best_fid/checkpoint_20000"
  "food101:$MAIN/files/logs/finetuning/food101_SiT_DMF_plain_meanflow_meft_online_20260722_232709_tv7xmz/best_fid/checkpoint_20000"
  "stanfordcars:$MAIN/files/logs/finetuning/stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo/best_fid/checkpoint_30000"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_SiT_MeFT_CAIMF_puresadv_${STAMP}"; }

# done test: workdir has a best_fid checkpoint AND its screen is gone
train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "caimf_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="caimf_gpu${gpu}_${ds}"
  log "LAUNCH CAIMF $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-caimf-$ds
    bash scripts/run_sit_meft_adversarial.sh caimf $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/caimf_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== CAIMF SiT-MeFT watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, 150k batches, eval@5k NFE4 10k-samples, best-fid-only, gate $STABLE_NEEDED ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done
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
  # release GPUs whose run finished
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if ! screen -ls | grep -qE "caimf_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check caimf_${ds}.log"; DONE[$ds]=1; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # assign
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx; remaining $(remaining)"; sleep 20; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all CAIMF SiT-MeFT runs complete; watcher exiting ==="
