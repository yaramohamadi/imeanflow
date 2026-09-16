#!/usr/bin/env bash
# =============================================================================
# imnet-CAMF 150k -> 300k CONTINUATION (2026-07-28). Full-state resume of the
# ImageNet-iMF CA-iMF pure-adversarial runs (all 5 datasets) that did NOT
# converge by 150k (FID still monotonically falling). Detached: screen
# imnet_caimf_300k.
#
# Uses the NEW full-state resume path (caimf.resume_from, patched into
# train_caimf.py 2026-07-28): restores the COMPLETE CAIMFTrainState -- gen+EMA
# params, both optimizer states, discriminator params/opt, and the step
# counters -- from each run's final periodic checkpoint_* (~150k). Training
# continues at step ~150k with the D/G equilibrium and post-warmup 4:1 cadence
# intact, up to max_posttrain_batches=300000.
#
# Writes to a FRESH workdir  ${ds}_iMF_imnet_CAIMF_puresadv_20260727_imnet_300k
# so the completed 150k artifacts (checkpoints, best_fid, eval_metrics.csv,
# final NFE1&2) are left UNTOUCHED. The new run keeps its own best_fid over the
# 150k->300k window and auto-runs final NFE 1&2 (10k) on it.
#
# GATED: does nothing until ALL current sweeps are finished -- no screens
# matching imnetcaimf_gpu* (the 150k training), afmrf_* / afmredoeval_*
# (AFM redo), or nfe816_* (NFE 8&16 evals), AND their watchers gone. Then it
# grabs GPUs opportunistically off the shared atomic lock, 1 GPU/run, NO
# early-stop (run the full 300k), 10k eval samples.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/imnet_caimf_300k_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_imf_caimf_posttrain
STAMP=20260727_imnet          # source (150k) workdir stamp
MAXB=300000
RUNNER="$ADV/scripts/run_imf_caimf.sh"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200 food101 stanfordcars)

src_wd_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_imnet_CAIMF_puresadv_${STAMP}"; }
dst_wd_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_imnet_CAIMF_puresadv_${STAMP}_300k"; }

# RESCUED 150k full checkpoints. The original workdirs were polluted by a buggy
# self-restart of the original watcher (2026-07-28): after hitting 150k, those
# runs restarted from step 0 and re-appended a 0->~70k curve, clobbering
# best_fid + final_eval. The genuine checkpoint_150000 survived (restart never
# re-reached 150k), and we copied all 5 here with verified full state
# (gen+ema+both opt+disc+step). Resume from these, NOT the polluted workdirs.
RESCUE="$ADV/files/logs/finetuning/_imnet150k_rescue"

resume_ckpt_for(){  # $1=ds -> verified 150k full-state checkpoint
  echo "$RESCUE/${1}_checkpoint_150000"
}

# the 300k run is done once its NEW workdir has a best_fid checkpoint and no screen
train_done(){  # $1=ds
  local wd; wd=$(dst_wd_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "imnet300k_gpu[0-9]+_${1}\b" && return 1
  return 0
}

# GATE: block until the prior sweeps are fully done.
gate_clear(){
  screen -ls 2>/dev/null | grep -qE "imnetcaimf_gpu[0-9]+_|imnet_caimf_watcher" && return 1
  screen -ls 2>/dev/null | grep -qE "afmrf_gpu[0-9]+_|afm_redo_foodcars"          && return 1
  screen -ls 2>/dev/null | grep -qE "afmredoeval_gpu[0-9]+_"                       && return 1
  screen -ls 2>/dev/null | grep -qE "nfe816_gpu[0-9]+_|nfe816_watcher"             && return 1
  return 0
}

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"
  local ckpt; ckpt=$(resume_ckpt_for "$ds")
  [[ -n "$ckpt" && -d "$ckpt" ]] || { log "ERROR [$ds] no source checkpoint in $(src_wd_for "$ds")"; return 1; }
  # robust: take the trailing run of digits (handles both
  # .../checkpoint_150000 and .../<ds>_checkpoint_150000 rescue names)
  local step; step=$(echo "$ckpt" | grep -oE '[0-9]+$')
  if [[ -z "$step" ]] || (( step < 100000 )); then log "ERROR [$ds] source ckpt step '${step:-?}' < 100000 -- refusing to resume from a too-early checkpoint ($ckpt)"; return 1; fi
  local wd; wd=$(dst_wd_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] 300k already has best_fid"; return 1; fi
  # load_from is required by run_imf_caimf.sh arg-validation but is IGNORED for
  # param init when resume_from is set (full state overrides it). Point it at
  # the same source checkpoint so the [[ -d ]] check passes.
  local sess="imnet300k_gpu${gpu}_${ds}"
  log "LAUNCH imnet-CAMF-300k $ds on GPU $gpu (screen $sess) resume=$ckpt (step $step) -> $wd cap=$MAXB"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export CONFIG_MODE=$CONFIG_MODE
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export RESUME_FROM='$ckpt'
    export MAX_POSTTRAIN_BATCHES=$MAXB
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-imnet300k-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/imnet300k_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== imnet-CAMF 300k watcher: ${#QUEUE[@]} datasets, GATED behind imnetcaimf/afmredo/nfe816, full-state resume ~150k->${MAXB}, eval@5k NFE4 10k, best-fid-only, final NFE{$FINAL_EVAL_STEPS}, gate $STABLE_NEEDED, NO early-stop, GPU-locked ==="

for ds in "${QUEUE[@]}"; do if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (300k best_fid already present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ imnet300k_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "imnet300k_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "imnet300k_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for d in "${QUEUE[@]}"; do [[ -z "${DONE[$d]:-}" ]] && n=$((n+1)); done; echo "$n"; }

# ---- GATE loop ----
GATED=1
while (( GATED )); do
  if gate_clear; then GATED=0; log "GATE CLEAR: all prior sweeps finished; beginning 300k continuation."; break; fi
  sleep "$POLL_S"
done

while (( $(remaining) > 0 )); do
  gpu_gc_locks
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if gpu_locked "$idx"; then STABLE[$idx]=0; continue; fi
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if ! screen -ls | grep -qE "imnet300k_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no 300k best_fid -- check imnet300k_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for ds in "${QUEUE[@]}"; do
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "imnet300k_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; unset 'CLAIMED[$idx]'; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all imnet-CAMF 300k continuations complete; watcher exiting ==="
