#!/usr/bin/env bash
# =============================================================================
# CAMF (CA-iMF finite-interval, pure-adversarial lambda_imf=0) starting DIRECTLY
# from the IMAGENET iMF checkpoint (iMF-XL-2-full) -- NOT from the domain-
# finetuned iMF. This is the "no domain-finetuning" CAMF ablation: adversarial
# post-training must adapt features AND a freshly-reinitialised class embedding
# (ImageNet 1000 -> target N via partial_load skip) straight on the target set.
# Detached: screen imnet_caimf_watcher.
#
# Same config + runner as the finetuned-iMF CAMF sweep
# (configs/caltech_imf_caimf_posttrain_config.yml + scripts/run_imf_caimf.sh);
# the ONLY difference is every dataset loads_from the single ImageNet base
# checkpoint instead of its own plain_iMF_finetune_<ds> best_fid.
#
#   lambda_imf=0 lambda_adv=1 lambda_cp=0.001 ; D-warmup 5000 ; 4:1 D:G
#   max_posttrain_batches=150000 (=30k G-updates) ; gen/dis LR=1e-5
#   fid@5k every 5k, metric_num_steps=[4], 10000 samples, save_best_fid_only
#   native iMF op-point (omega=7.5, t in [0.4,0.65]). Final eval NFE 1&2.
#
# PREDECESSOR GATE: does NOT claim any GPU until BOTH prior sweeps are fully
#   done -- no caimfimf_* (finetuned-iMF CAMF) AND no afmimf_* (iMF AFM) screens
#   remain (this covers the 3 running + 2 queued AFM runs). Polls until clear.
# EARLY-STOP: patience 5. One GPU/run. Shares the atomic GPU lock.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/imnet_caimf_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_imf_caimf_posttrain
STAMP=20260727_imnet
PATIENCE=999999           # early-stop DISABLED -- run full 150k (user request 2026-07-27)
WAIT_FOR_PREDECESSORS=0   # gate DISABLED -- start now, don't wait for caimfimf_*/afmimf_*
FORCE_FIRST_GPU="${FORCE_FIRST_GPU:-}"   # claim this GPU immediately for queue[0], skip stability gate
RUNNER="$ADV/scripts/run_imf_caimf.sh"

# The single ImageNet iMF base checkpoint every dataset starts from.
IMNET_CKPT="$IMF/files/weights/iMF-XL-2-full"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

source "$ADV/scripts/gpu_lock.inc.sh"

# ds : load_from  -- ALL point at the ImageNet iMF base.
# caltech101 intentionally SKIPPED for now (user request 2026-07-27).
QUEUE=(
  "artbench10:$IMNET_CKPT"
  "cub200:$IMNET_CKPT"
  "food101:$IMNET_CKPT"
  "stanfordcars:$IMNET_CKPT"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_imnet_CAIMF_puresadv_${STAMP}"; }

# predecessor gate: true while ANY finetuned-CAMF or AFM run is still alive.
predecessors_active(){
  screen -ls 2>/dev/null | grep -qE "(caimfimf_gpu[0-9]+_|afmimf_gpu[0-9]+_)" && return 0
  return 1
}

# eval priority: evals go BEFORE new training (user request 2026-07-27). Defer
# launching ADDITIONAL imnet runs only while the retro-eval DRIVER is alive --
# it represents PENDING eval work still waiting for a GPU. Already-running eval
# screens hold their own GPUs and do not compete for a free one, so they alone
# must not idle a spare GPU. The running GPU7 imnet run is unaffected regardless.
evals_pending(){
  screen -ls 2>/dev/null | grep -qE "\.retro_eval_imf\b" && return 0
  return 1
}

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "imnetcaimf_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="imnetcaimf_gpu${gpu}_${ds}"
  log "LAUNCH imnet-CAMF $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export CONFIG_MODE=$CONFIG_MODE
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-imnetcaimf-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/imnetcaimf_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== imnet-CAMF watcher: ${#QUEUE[@]} datasets from ImageNet iMF ($IMNET_CKPT), pool {$ALLOWED}, 150k batches, eval@5k NFE4 10k-samples, best-fid-only, final NFE{$FINAL_EVAL_STEPS}, gate $STABLE_NEEDED, GPU-locked; WAITS for finetuned-CAMF + AFM to finish first ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done

# adopt any of OUR runs already in flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ imnetcaimf_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "imnetcaimf_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx (screen imnetcaimf_gpu${gidx}_${gds})"
done < <(screen -ls 2>/dev/null | grep -oE "imnetcaimf_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

# ---- PREDECESSOR GATE (disabled by default now) ----
if (( WAIT_FOR_PREDECESSORS == 1 )); then
  gate_announced=0
  while predecessors_active; do
    if (( gate_announced == 0 )); then
      log "GATE: waiting for finetuned-iMF CAMF (caimfimf_*) and iMF AFM (afmimf_*) runs to finish before starting; polling every ${POLL_S}s."
      gate_announced=1
    fi
    sleep "$POLL_S"
  done
  log "GATE CLEARED: no caimfimf_*/afmimf_* screens remain; starting imnet-CAMF sweep."
else
  log "GATE DISABLED: starting imnet-CAMF sweep immediately (co-exists with finetuned-CAMF training)."
fi

# ---- FORCE first run onto a specific GPU immediately (skip stability gate) ----
if [[ -n "$FORCE_FIRST_GPU" && $(remaining) -gt 0 ]]; then
  fg="$FORCE_FIRST_GPU"
  if [[ -n "${CLAIMED[$fg]:-}" ]]; then
    log "FORCE-START skipped: GPU $fg already claimed by us."
  elif gpu_locked "$fg"; then
    log "FORCE-START skipped: GPU $fg is locked by another job; will fall through to normal scheduling."
  else
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"; [[ -n "${DONE[$ds]:-}" ]] && continue
      if gpu_try_claim "$fg" "imnetcaimf_gpu${fg}_${ds}"; then
        if launch "$fg" "$entry"; then CLAIMED[$fg]="$ds"; log "FORCE-START $ds -> GPU $fg; remaining $(remaining)"; sleep 20; else gpu_release "$fg"; fi
      else
        log "FORCE-START could not lock GPU $fg."
      fi
      break
    done
  fi
fi

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
    if screen -ls | grep -qE "imnetcaimf_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing imnetcaimf_gpu${idx}_${ds}."
        screen -S "imnetcaimf_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    if ! screen -ls | grep -qE "imnetcaimf_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check imnetcaimf_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # EVAL PRIORITY via stability timing (NOT a hard block): imnet needs
  # STABLE_NEEDED=6 stable polls before claiming a freed GPU, while the retro
  # eval driver needs only 3 -- so evals always win a contested freed GPU, yet
  # imnet still fills any GPU no pending eval wants (no idling). See evals_pending.
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "imnetcaimf_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all imnet-CAMF runs complete; watcher exiting ==="
