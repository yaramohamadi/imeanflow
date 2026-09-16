#!/usr/bin/env bash
# =============================================================================
# AFM-redo FINAL EVAL (NFE 1 & 2) on the best_fid checkpoints of the 3 AFM-redo
# runs that were early-stopped for divergence (artbench10, caltech101, cub200).
# Detached: screen afmredo_finaleval. Uses the SAME eval path the runner uses
# (scripts/eval_best_fid_steps_plain_imf.sh -> main.py eval_only), 10k samples.
#
# Each run's best_fid/ holds exactly one checkpoint (the true FID minimum):
#   artbench10 -> checkpoint_10000  (best NFE4 FID 12.34)
#   caltech101 -> checkpoint_2500   (best NFE4 FID 24.45)
#   cub200     -> checkpoint_12500  (best NFE4 FID 3.54)
#
# One GPU per dataset, claimed opportunistically off the shared atomic lock so
# it coexists with the imnet-CAMF training. Writes final_eval_metrics.csv in
# each run dir (NFE 1 & 2). GPUs 1/4/5 were just freed by the early-stop.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/afmredo_finaleval_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=2
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
CONFIG_MODE=caltech_imf_afm_posttrain
STEPS="1 2"
DATA=/opt/dlami/nvme/meanflow/datasets

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200)

wd_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_AFM_puresadv_20260727_redo"; }
ds_case(){ case "$1" in
  artbench10)   LAT=artbench-10_processed_latents;  NC=10;  FID=artbench-10_processed-fid_stats.npz;  FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   LAT=caltech-101_processed_latents;  NC=101; FID=caltech-101-fid_stats.npz;            FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       LAT=cub-200-2011_processed_latents; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
esac; }

has_both(){ local csv=$(wd_for "$1")/final_eval_metrics.csv
  awk -F, 'NR>1 && $4==1{a=1} NR>1 && $4==2{b=1} END{exit !(a&&b)}' "$csv" 2>/dev/null; }

eval_done(){  # $1=ds
  screen -ls | grep -qE "afmredoeval_gpu[0-9]+_${1}\b" && return 1
  has_both "$1"
}

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"; ds_case "$ds"
  local wd; wd=$(wd_for "$ds"); local ROOT="$DATA/$LAT"
  local sess="afmredoeval_gpu${gpu}_${ds}"
  log "LAUNCH $sess (final NFE $STEPS) wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY CONFIG_MODE=$CONFIG_MODE USE_WANDB=False
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-afmredoeval-$ds
    bash $ADV/scripts/eval_best_fid_steps_plain_imf.sh \
      $wd $STEPS -- \
      --config.dataset.name=${ds}_latent \
      --config.dataset.root=$ROOT \
      --config.dataset.class_mapping_root= \
      --config.dataset.num_classes=$NC \
      --config.model.num_classes=$NC \
      --config.sampling.num_classes=$NC \
      --config.fid.cache_ref=$ADV/files/fid_stats/$FID \
      --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD \
      --config.fid.num_samples=10000 \
      2>&1 | tee -a $ADV/files/logs/afmredoeval_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== AFM-redo final-eval watcher: ${#QUEUE[@]} datasets (NFE $STEPS, 10k), pool {$ALLOWED}, GPU-locked ==="
for ds in "${QUEUE[@]}"; do if eval_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (NFE 1&2 already present)"; fi; done

remaining(){ local n=0; for d in "${QUEUE[@]}"; do [[ -z "${DONE[$d]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  gpu_gc_locks
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line"|awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line"|awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line"|awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if gpu_locked "$idx"; then STABLE[$idx]=0; continue; fi
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if ! screen -ls | grep -qE "afmredoeval_gpu${idx}_${ds}\b"; then
      if has_both "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but NFE 1&2 not both present -- check afmredoeval_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "afmredoeval_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all AFM-redo final evals complete; watcher exiting ==="
