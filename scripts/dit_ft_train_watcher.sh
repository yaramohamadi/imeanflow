#!/usr/bin/env bash
# =============================================================================
# DiT FT (plain DDPM-eps DiT fine-tune) training sweep, all 5 datasets.
# Detached: screen dit_ft_train_watcher.
#
# WHY: the table's DiT FT row (FT@250 / @4 / @8) needs real checkpoints, but no
# plain_DiT_finetune_* runs exist on dev5 OR taylor5 -- the FT infra is present
# but was never run. This trains DiT-XL/2 from DiT-XL-2-256x256.pt on each of the
# 5 datasets with the original DDPM objective (config plain_dit_finetune), then
# runs the final best-FID eval at NFE 8/4/2/1/250 (8 FIRST -> the deadline-
# critical number lands earliest).
#
# DEADLINE CAP: max_train_steps=15000 (FT best_fid lands early; iMF-FT ref peaked
# ~7500; fid_schedule evals at 5k/10k/15k so best_fid selection is covered).
# ~4400 steps/hr => ~3.4h train + eval. 10k FID samples (standing rule).
# save_best_fid_only=True (standing rule). use_ema=False -> online metric mode.
#
# One GPU per run via shared atomic lock, opportunistic on pool {0 1 2 3 4 5}.
# 4 start immediately (0/1/2/4); 5th claims GPU 3 or 5 the moment the imnet300k
# finevals free one. Never touches 6/7 (CP-ablation). Screen prefix ditft_.
# =============================================================================
set -uo pipefail
MAIN=/opt/dlami/nvme/meanflow/imeanflow
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
PY=$MAIN/.venv/bin/python
cd "$MAIN"
LOG="$ADV/files/logs/dit_ft_train_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5"
NSAMP=10000
MAX_STEPS=15000
FINAL_STEPS="8 4 2 1 250"
LOADW="$MAIN/files/weights/DiT-XL-2-256x256.pt"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200 food101 stanfordcars)

# ds -> "latent_root:fid_npz:fdd_npz"
entry_for(){ case "$1" in
  artbench10)   echo "artbench-10_processed_latents:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz" ;;
  caltech101)   echo "caltech-101_processed_latents:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz" ;;
  cub200)       echo "cub-200-2011_processed_latents:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      echo "food-101_processed_latents:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) echo "stanford-cars_processed_latents:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz" ;;
esac; }

# marker file so a restarted watcher knows a ds already got launched (WORKDIR is
# stamped with time+salt so we can't recompute it; we drop a marker at launch).
marker_for(){ echo "$ADV/files/logs/ditft_launched_${1}.marker"; }

train_done(){  # $1=ds -> screen gone AND a completed final-eval CSV with NFE8 exists
  screen -ls | grep -qE "ditft_gpu[0-9]+_${1}\b" && return 1
  local m; m="$(marker_for "$1")"
  [[ -f "$m" ]] || return 1
  local wd; wd="$(cat "$m" 2>/dev/null)"
  [[ -n "$wd" && -d "$wd" ]] || return 1
  # final eval writes eval_best_fid_<N>steps/eval_metrics.csv; require the 8-step one
  local csv="$wd/eval_best_fid_8steps/eval_metrics.csv"
  [[ -f "$csv" ]] && awk -F, 'NR>1{f=1} END{exit !f}' "$csv" 2>/dev/null
}

launched(){ [[ -f "$(marker_for "$1")" ]]; }

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"
  IFS=':' read -r lat fid fdd <<< "$(entry_for "$ds")"
  local root="$MAIN/../datasets/$lat"
  [[ -d "$root" ]] || { log "ERROR [$ds] dataset root missing: $root"; return 1; }
  local sess="ditft_gpu${gpu}_${ds}"
  log "LAUNCH $sess (DiT FT DDPM-eps, max_steps=$MAX_STEPS, final NFE {$FINAL_STEPS}) root=$lat"
  screen -dmS "$sess" bash -c "
    cd $MAIN
    DATASET_NAME=$ds \
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHON=$PY \
    USE_WANDB=False \
    LOAD_FROM=$LOADW \
    DATASET_ROOT=$root \
    FID_CACHE_REF=$MAIN/files/fid_stats/$fid \
    FD_DINO_CACHE_REF=$MAIN/files/fdd_stats/$fdd \
    RUN_FINAL_BEST_FID_EVAL=True \
    FINAL_EVAL_STEPS='$FINAL_STEPS' \
    FINAL_EVAL_USE_WANDB=False \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash scripts/train_plain_dit_finetune.sh ftdl \
      --config.training.max_train_steps=$MAX_STEPS \
      --config.training.batch_size=16 \
      --config.training.grad_accum_steps=2 \
      --config.fid.num_samples=$NSAMP \
      2>&1 | tee -a $ADV/files/logs/ditft_${ds}.log
    # record WORKDIR for train_done: grab it from the run's own log line
    wd=\$(grep -oE 'files/logs/finetuning/plain_DiT_finetune_${ds}_ftdl_[0-9_]+_[a-z0-9]+' $ADV/files/logs/ditft_${ds}.log | head -1)
    [[ -n \"\$wd\" ]] && echo \"$MAIN/\$wd\" > $(marker_for "$ds")
    source $ADV/scripts/gpu_lock.inc.sh; gpu_release $gpu
  "
  # pre-seed marker path immediately (empty content); filled after WORKDIR known
  : > "$(marker_for "$ds")"
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== DiT FT training sweep: ${#QUEUE[@]} datasets, pool {$ALLOWED}, max_steps=$MAX_STEPS, 10k FID, final NFE {$FINAL_STEPS}, GPU-locked, opportunistic ==="
for ds in "${QUEUE[@]}"; do if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (final 8-step eval already present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ ditft_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "$sess" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "ditft_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for d in "${QUEUE[@]}"; do [[ -z "${DONE[$d]:-}" ]] && n=$((n+1)); done; echo "$n"; }

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
    if ! screen -ls | grep -qE "ditft_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but final 8-step eval not present -- check ditft_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for ds in "${QUEUE[@]}"; do
      [[ -n "${DONE[$ds]:-}" ]] && continue
      launched "$ds" && continue           # already started once, don't relaunch
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "ditft_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; unset 'CLAIMED[$idx]'; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all DiT FT runs complete (train+final eval); watcher exiting ==="
