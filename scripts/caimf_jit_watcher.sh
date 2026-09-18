#!/usr/bin/env bash
# =============================================================================
# CAIMF (pure-adversarial, lambda_imf=0) JiT-MeFT post-training watcher.
# Detached: screen caimf_jit_watcher.  SELF-SMOKING.
#
# JiT-DMF is PIXEL space (256x256x3) and has never run the adversarial path on
# GPU, so this watcher first runs a short caltech GPU smoke to prove memory
# feasibility (batch 2, grad_accum 1) before committing all 5 datasets.
# NOTE: CA-iMF forbids gradient accumulation (train_caimf.py raises if
# grad_accum_steps != 1), so batch_size is the ONLY memory lever -- the OOM
# fallback drops to batch 1 (NOT accum).
#
# Phase 0 (SMOKE): on the first stably-free GPU, run run_jit_meft_caimf.sh with
#   SMOKE=1 on caltech (200 D-warmup, 800 batches, one eval @400, 200 samples).
#   PASS = clean exit AND a best_fid checkpoint appears (=> finite FID, no OOM).
#   If the smoke exits with an OOM, retry once at batch 1 (grad_accum stays 1)
#   and, if that passes, pin all real runs to that fallback via EXTRA_ARGS.
#   If the smoke fails for any other reason, the watcher logs and EXITS without
#   launching the 5 real runs (so a broken JiT path never burns 5 GPUs).
# Phase 1 (RUN): only on smoke PASS, launch all 5 JiT-DMF students off their
#   best_fid MF-A checkpoints, one GPU per run, gated on stably-free GPUs.
#
# Config: configs/caltech_jit_meft_caimf_posttrain_config.yml
#   lambda_imf=0 lambda_adv=1 cp=0.001 ; gen/dis LR=1e-6 (=0.1x JiT base 1e-5)
#   transport/data target mapping ; omega=2.2 t in [0.1,1] (JiT MF-A op point)
#   batch 2 (accum 1) ; use_ema=True ; fid@5k every 5k ; metric_num_steps=[4]
#   10000 samples ; save_best_fid_only ; max_posttrain_batches=150000.
# FINAL EVAL: NFE 1 and 2 only.  One GPU per run.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/caimf_jit_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6          # ~9min stably-free before claiming
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
STAMP=20260726
PATIENCE=5               # consecutive 4-step-FID rises from min -> early stop
RUNNER="$ADV/scripts/run_jit_meft_caimf.sh"

# 4-step FID series (col 4 == num_steps, col 8 == fid) -> consecutive rises from min
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

# Fallback batch config applied to ALL real runs if the batch-2 smoke OOMs but
# the batch-1 smoke passes.  Empty by default (batch 2, accum 1 from config).
# CA-iMF forbids grad accumulation, so the only lever is batch_size.
FALLBACK_ARGS=""

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# shared atomic GPU lock (coordinates with caimf_dit_watcher)
source "$ADV/scripts/gpu_lock.inc.sh"

# ds : best_fid MF-A checkpoint (JiT-DMF transport+clip1 students, workdirs/)
WD=$MAIN/files/workdirs
QUEUE=(
  "artbench10:$WD/jitdmfclip_artbench10/best_fid/checkpoint_32500"
  "caltech101:$WD/jitdmfclip_caltech101/best_fid/checkpoint_40000"
  "cub200:$WD/jitdmfclip_cub200/best_fid/checkpoint_27500"
  "food101:$WD/jitdmfclip_food101/best_fid/checkpoint_30000"
  "stanfordcars:$WD/jitdmfclip_stanfordcars/best_fid/checkpoint_35000"
)
SMOKE_DS=caltech101
SMOKE_CKPT="$WD/jitdmfclip_caltech101/best_fid/checkpoint_40000"

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_JiT_MeFT_CAIMF_puresadv_${STAMP}"; }

# ---- wait for a stably-free GPU, claim its lock, echo its index -------------
# Writes the chosen index to global CLAIMED_GPU (caller must gpu_release it).
wait_free_gpu(){
  declare -A ST=()
  local sess="$1"
  while true; do
    gpu_gc_locks
    mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    for line in "${SMI[@]}"; do
      idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
      mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
      util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
      [[ " $ALLOWED " == *" $idx "* ]] || continue
      if gpu_locked "$idx"; then ST[$idx]=0; continue; fi
      if (( mem < MEM_MB && util < UTIL_PCT )); then ST[$idx]=$(( ${ST[$idx]:-0} + 1 )); else ST[$idx]=0; fi
      if (( ${ST[$idx]:-0} >= STABLE_NEEDED )); then
        if gpu_try_claim "$idx" "$sess"; then CLAIMED_GPU="$idx"; echo "$idx"; return 0; fi
      fi
    done
    sleep "$POLL_S"
  done
}

# ---- run the caltech smoke synchronously on $1=gpu, return 0 on PASS --------
run_smoke(){
  local gpu="$1" extra="$2" tag="$3"
  local swd="$ADV/files/logs/finetuning/caltech101_JiT_MeFT_CAIMF_SMOKE_${tag}_${STAMP}"
  rm -rf "$swd"; mkdir -p "$swd"
  log "SMOKE[$tag] caltech on GPU $gpu (extra='$extra') wd=$swd"
  # PREALLOCATE=true: JiT pixel eval (256x256x3) + the fat CA-iMF state fragments
  # the heap under lazy alloc, OOMing the tiny post-eval save. Each run owns its
  # GPU (watcher-gated), so grabbing the whole pool upfront is safe and defrags.
  CUDA_VISIBLE_DEVICES="$gpu" SMOKE=1 RUN_FINAL_EVAL=False PYTHON="$PY" \
    EXTRA_ARGS="$extra" \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    MPLCONFIGDIR=/tmp/mpl-caimfjit-smoke-$tag \
    bash "$RUNNER" "$SMOKE_DS" "$SMOKE_CKPT" "$swd" \
      > "$ADV/files/logs/caimfjit_smoke_${tag}.log" 2>&1
  local st=$?
  local smokelog="$ADV/files/logs/caimfjit_smoke_${tag}.log"
  if ls -d "$swd/best_fid/checkpoint_"* >/dev/null 2>&1; then
    log "SMOKE[$tag] PASS (exit $st, best_fid checkpoint written -> finite FID, no OOM)"
    return 0
  fi
  if grep -qiE "RESOURCE_EXHAUSTED|out of memory|Out of memory|OOM" "$smokelog"; then
    log "SMOKE[$tag] OOM (exit $st) -- see $smokelog"
    return 2   # distinguishable: caller may try smaller batch
  fi
  log "SMOKE[$tag] FAIL (exit $st, no best_fid, no OOM signature) -- see $smokelog"
  return 1
}

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "caimfjit_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="caimfjit_gpu${gpu}_${ds}"
  log "LAUNCH CAIMF-JiT $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd extra='$FALLBACK_ARGS'"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export EXTRA_ARGS='$FALLBACK_ARGS'
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
    export MPLCONFIGDIR=/tmp/mpl-caimfjit-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/caimfjit_${ds}.log
  "
}

# ============================ PHASE 0: SMOKE =================================
log "=== CAIMF JiT-MeFT watcher start: SMOKE phase (${#QUEUE[@]} runs queued after pass) ==="
[[ -x "$PY" ]] || { log "FATAL python missing: $PY"; exit 1; }
[[ -f "$RUNNER" ]] || { log "FATAL runner missing: $RUNNER"; exit 1; }
[[ -d "$SMOKE_CKPT" ]] || { log "FATAL smoke ckpt missing: $SMOKE_CKPT"; exit 1; }

SMOKE_GPU=$(wait_free_gpu "caimf_jit_watcher_smoke")
log "SMOKE GPU claimed (locked): $SMOKE_GPU"
run_smoke "$SMOKE_GPU" "" "b2a8"
rc=$?
gpu_release "$SMOKE_GPU"
if (( rc == 2 )); then
  log "batch2 OOM; retrying smoke at batch 1 (grad_accum stays 1; CA-iMF forbids accum)"
  SMOKE_GPU=$(wait_free_gpu "caimf_jit_watcher_smoke")
  run_smoke "$SMOKE_GPU" "--config.training.batch_size=1" "b1"
  rc=$?
  gpu_release "$SMOKE_GPU"
  if (( rc == 0 )); then
    FALLBACK_ARGS="--config.training.batch_size=1"
    log "fallback batch 1 adopted for all real runs"
  fi
fi
if (( rc != 0 )); then
  log "=== SMOKE did not pass (rc=$rc); NOT launching the 5 real runs. Watcher exiting for human review. ==="
  exit 1
fi

# ============================ PHASE 1: RUN 5 ================================
declare -A STABLE=() CLAIMED=() DONE=()
for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done
remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }
log "=== SMOKE PASSED. RUN phase: ${#QUEUE[@]} datasets, pool {$ALLOWED}, gate $STABLE_NEEDED, final-eval NFE{$FINAL_EVAL_STEPS} ==="

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
    # integrated early-stop (patience PATIENCE): kill a still-training run whose
    # 4-step FID has risen from its min for >= PATIENCE consecutive evals
    # (best_fid checkpoint is retained).
    if screen -ls | grep -qE "caimfjit_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      if [[ -f "$csv" ]]; then
        series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
        if (( n >= PATIENCE + 1 )); then
          streak=$(echo "$series" | consec_rises_from_min)
          if (( streak >= PATIENCE )); then
            mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
            log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing caimfjit_gpu${idx}_${ds}."
            screen -S "caimfjit_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
          fi
        fi
      fi
    fi
    if ! screen -ls | grep -qE "caimfjit_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check caimfjit_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "caimfjit_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all CAIMF JiT-MeFT runs complete; watcher exiting ==="
