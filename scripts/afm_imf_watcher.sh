#!/usr/bin/env bash
# =============================================================================
# AFM (discrete endpoint-adversarial, ablation=afm_only) on NATIVE, TARGET-
# FINETUNED iMF checkpoints. Detached: screen afm_imf_watcher.
#
# For each of the 5 plain-iMF (imfDiT_XL_2) domain-finetuned students, runs
# endpoint-AFM pure-adversarial post-training OFF its best_fid checkpoint (the
# SAME 5 checkpoints the CAMF sweep uses -- the domain-finetuned iMF, NOT the
# ImageNet iMF), on any stably-free GPU. Uses main_afm.py DIRECTLY (train_afm
# accepts model_str=imfDiT_* natively -> no _sit_meft shim, no guard relax).
#
# Config: configs/caltech_imf_afm_posttrain_config.yml
#   ablation=afm_only lambda_adv=1.0 lambda_imf=0.0 lambda_cp=0.01
#   D-warmup 5000 + 4:1 D:G ; max_posttrain_batches=150000 (=30k G-updates)
#   gen/dis LR=1e-5 (=0.1x iMF base FT LR 1e-4) -- set in config
#   fid@5k every 5k, metric_num_steps=[4], 10000 samples, save_best_fid_only
#   NATIVE iMF operating point: sampling omega=7.5 t in [0.4,0.65].
#
# FINAL EVAL: NFE 1 and 2 only (via eval_best_fid_steps_plain_imf.sh).
# EARLY-STOP: patience 5 (consecutive 4-step-FID rises from min). One GPU/run.
# GPU locking: shares the atomic lock with the CAMF (caimf) watchers so no two
#   gated watchers double-claim the same idle GPU.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/afm_imf_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6          # ~9min stably-free before claiming (avoid train->eval gaps)
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_imf_afm_posttrain
STAMP=20260727
PATIENCE=5               # consecutive 4-step-FID rises from min -> early stop
RUNNER="$ADV/scripts/run_imf_afm.sh"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 4-step FID series (col 4 == num_steps, col 8 == fid) -> consecutive rises from min
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

# shared atomic GPU lock (coordinates with caimf_imf/dit/jit watchers)
source "$ADV/scripts/gpu_lock.inc.sh"

# ds : best_fid checkpoint (absolute) -- plain-iMF DOMAIN-FINETUNED imfDiT_XL_2 students
QUEUE=(
  "artbench10:$IMF/files/logs/finetuning/plain_iMF_finetune_artbench10_h100_20260722_225848_jt6v93/best_fid/checkpoint_7500"
  "caltech101:$IMF/files/logs/finetuning/plain_iMF_finetune_caltech101_h100_20260722_222718_cwbunp/best_fid/checkpoint_32500"
  "cub200:$IMF/files/logs/finetuning/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/best_fid/checkpoint_35000"
  "food101:$IMF/files/logs/finetuning/plain_iMF_finetune_food101_h100_20260722_231935_s9moo7/best_fid/checkpoint_17500"
  "stanfordcars:$IMF/files/logs/finetuning/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh/best_fid/checkpoint_12500"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_AFM_puresadv_${STAMP}"; }

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "afmimf_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="afmimf_gpu${gpu}_${ds}"
  log "LAUNCH AFM-iMF $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
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
    export MPLCONFIGDIR=/tmp/mpl-afmimf-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/afmimf_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== AFM iMF watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, 150k batches, eval@5k NFE4 10k-samples, best-fid-only, final-eval NFE{$FINAL_EVAL_STEPS}, gate $STABLE_NEEDED, GPU-locked ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done

# Adopt runs already in flight (e.g. after a watcher restart): re-populate
# CLAIMED + re-take the GPU lock so we never double-launch a live run.
while read -r sess; do
  [[ "$sess" =~ afmimf_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "afmimf_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx (screen afmimf_gpu${gidx}_${gds})"
done < <(screen -ls 2>/dev/null | grep -oE "afmimf_gpu[0-9]+_[a-z0-9]+")

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
    if screen -ls | grep -qE "afmimf_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing afmimf_gpu${idx}_${ds}."
        screen -S "afmimf_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    # screen gone (finished on its own or just early-stopped)
    if ! screen -ls | grep -qE "afmimf_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check afmimf_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "afmimf_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all AFM iMF runs complete; watcher exiting ==="
