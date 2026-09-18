#!/usr/bin/env bash
# =============================================================================
# CAIMF (pure-adversarial, lambda_imf=0) DiT-MeFT post-training watcher.
# Detached: screen caimf_dit_watcher.
#
# For each of the 5 DiT-DMF MeFT students (online+clip1 MF-A), runs
# finite-interval CA-iMF adversarial post-training OFF its best_fid checkpoint,
# on any stably-free GPU. Reuses the SiT-MeFT launcher + train path verbatim
# (DiT-DMF shares the SiT-DMF backbone class; the create_models guard was
# relaxed to accept imfDiT_DMF_). Only CONFIG_MODE differs.
#
# Config: configs/caltech_dit_meft_caimf_posttrain_config.yml
#   lambda_imf=0.0 (PURE adversarial) lambda_adv=1.0 lambda_cp=0.001
#   ddpm-v (dit_native) target mapping + flipped/scaled native timestep
#   max_posttrain_batches=150000 (=30k G-updates; warmup 5000 + 4:1 D:G)
#   fid_schedule from 5000 every 5000, metric_num_steps=[4], 10000 samples
#   save_best_fid_only=True. sampling omega=1.5 t in [0,1] (DiT MF-A op point).
#
# FINAL EVAL: NFE 1 and 2 only (4-step comes from training-time evals).
# EARLY-STOP: handled inside the run (patience 5). One GPU per run.
# GPU locking: shares an atomic lock with caimf_jit_watcher so the two gated
#   watchers never double-claim the same idle GPU.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/caimf_dit_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6          # ~9min stably-free before claiming (avoid train->eval gaps)
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_dit_meft_caimf_posttrain
STAMP=20260726
PATIENCE=5               # consecutive 4-step-FID rises from min -> early stop

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 4-step FID series (col 4 == num_steps, col 8 == fid) -> consecutive rises from min
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

# shared atomic GPU lock (coordinates with caimf_jit_watcher)
source "$ADV/scripts/gpu_lock.inc.sh"

# ds : best_fid checkpoint (absolute) -- DiT-DMF online+clip1 MF-A students
QUEUE=(
  "artbench10:$MAIN/files/logs/finetuning/artbench10_DiT_DMF_meanflow_taylor_plain_online_clip1_artbench10_20260724_141156_87ttqh/best_fid/checkpoint_12500"
  "caltech101:$MAIN/files/logs/finetuning/caltech101_DiT_DMF_meanflow_taylor_plain_online_clip1_caltech101_20260724_141146_om01oq/best_fid/checkpoint_30000"
  "cub200:$MAIN/files/logs/finetuning/cub200_DiT_DMF_meanflow_taylor_plain_online_clip1_cub200_20260724_141206_euwxyx/best_fid/checkpoint_27500"
  "food101:$MAIN/files/logs/finetuning/food101_DiT_DMF_meanflow_taylor_plain_online_clip1_food101_20260724_141216_68a834/best_fid/checkpoint_30000"
  "stanfordcars:$MAIN/files/logs/finetuning/stanfordcars_DiT_DMF_meanflow_taylor_plain_online_clip1_stanfordcars_20260724_141226_17ll6g/best_fid/checkpoint_12500"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_DiT_MeFT_CAIMF_puresadv_${STAMP}"; }

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "caimfdit_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="caimfdit_gpu${gpu}_${ds}"
  log "LAUNCH CAIMF-DiT $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
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
    export MPLCONFIGDIR=/tmp/mpl-caimfdit-$ds
    bash scripts/run_sit_meft_adversarial.sh caimf $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/caimfdit_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== CAIMF DiT-MeFT watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, 150k batches, eval@5k NFE4 10k-samples, best-fid-only, final-eval NFE{$FINAL_EVAL_STEPS}, gate $STABLE_NEEDED, GPU-locked ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done

# Adopt runs already in flight (e.g. after a watcher restart): re-populate
# CLAIMED + re-take the GPU lock so we never double-launch a live run.
while read -r sess; do
  [[ "$sess" =~ caimfdit_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "caimfdit_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx (screen caimfdit_gpu${gidx}_${gds})"
done < <(screen -ls 2>/dev/null | grep -oE "caimfdit_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

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
    # integrated early-stop: kill a still-training run whose 4-step FID has
    # risen from its min for >= PATIENCE consecutive evals (best_fid retained).
    if screen -ls | grep -qE "caimfdit_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing caimfdit_gpu${idx}_${ds}."
        screen -S "caimfdit_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    # screen gone (finished on its own or just early-stopped)
    if ! screen -ls | grep -qE "caimfdit_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check caimfdit_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "caimfdit_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all CAIMF DiT-MeFT runs complete; watcher exiting ==="
