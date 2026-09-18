#!/usr/bin/env bash
# =============================================================================
# Catch-up final NFE 1&2 evals for iMF adversarial runs that finished WITHOUT
# their final eval (early-stopped runs: watcher kills the screen before the
# runner's post-training eval step fires).
#
# Scans all three families for workdirs that have best_fid but are MISSING
# eval_best_fid_1steps and/or eval_best_fid_2steps, then runs the plain-imf
# eval path (main.py --eval_only, steps 1 2, 10k samples) on free GPUs.
# Idempotent + safe to re-run: skips runs already evaluated or still training.
# Uses the shared atomic GPU lock so it never collides with the imnet watcher.
# Detached: screen retro_eval_imf.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/retro_eval_imf.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
RESERVE_FREE=0   # evals have priority over imnet training -- grab freed GPUs first (user 2026-07-27)
STEPS="1 2"
DATA_ROOT_BASE="${DATA_ROOT:-$ADV/../datasets}"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

ds_meta(){  # $1=ds -> "NC FID FDD LATENT"
  case "$1" in
    caltech101)   echo "101 caltech-101-fid_stats.npz caltech-101-fd_dino-vitb14_stats.npz caltech-101_processed_latents" ;;
    artbench10)   echo "10 artbench-10_processed-fid_stats.npz artbench-10-fd_dino-vitb14_stats.npz artbench-10_processed_latents" ;;
    cub200)       echo "200 cub-200-2011_processed-fid_stats.npz cub-200-2011-fd_dino-vitb14_stats.npz cub-200-2011_processed_latents" ;;
    food101)      echo "101 food-101_processed-fid_stats.npz food-101-fd_dino-vitb14_stats.npz food-101_processed_latents" ;;
    stanfordcars) echo "196 stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz stanford-cars_processed_latents" ;;
    *) return 1 ;;
  esac
}

# family table: workdir-suffix : config-mode : training-screen-prefix
# NOTE: imnet_CAIMF is intentionally EXCLUDED -- those runs have early-stop
# disabled (run full 150k) and self-eval via run_imf_caimf.sh's final-eval
# block, so retro must not touch them (would double-eval / race). This driver
# only catches the EARLY-STOPPED finetuned-CAMF + AFM runs.
FAMILIES=(
  "iMF_CAIMF_puresadv_20260726:caltech_imf_caimf_posttrain:caimfimf"
  "iMF_AFM_puresadv_20260727:caltech_imf_afm_posttrain:afmimf"
)
DATASETS="artbench10 caltech101 cub200 food101 stanfordcars"

eval_done(){  # $1=workdir -> 0 if both 1&2 step eval csvs present
  [[ -f "$1/eval_best_fid_1steps/eval_metrics.csv" && -f "$1/eval_best_fid_2steps/eval_metrics.csv" ]]
}

build_queue(){  # populates global QUEUE with entries "ds|workdir|configmode"
  QUEUE=()
  for fam in "${FAMILIES[@]}"; do
    IFS=':' read -r suffix cfg scr <<< "$fam"
    for ds in $DATASETS; do
      wd="$ADV/files/logs/finetuning/${ds}_${suffix}"
      ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || continue        # no best_fid yet
      screen -ls 2>/dev/null | grep -qE "${scr}_gpu[0-9]+_${ds}\b" && continue  # still training
      eval_done "$wd" && continue                                          # already evaluated
      # skip if a retro-eval for THIS workdir is already running
      screen -ls 2>/dev/null | grep -qE "retroeval_gpu[0-9]+_$(basename "$wd")\b" && continue
      QUEUE+=("${ds}|${wd}|${cfg}")
    done
  done
}

# training still in flight for any family? (means more eval work may appear later)
training_active(){
  for fam in "${FAMILIES[@]}"; do
    scr="${fam##*:}"
    screen -ls 2>/dev/null | grep -qE "${scr}_gpu[0-9]+_" && return 0
  done
  return 1
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS='|' read -r ds wd cfg <<< "$2"
  local meta; meta=$(ds_meta "$ds") || { log "ERROR unknown ds $ds"; return 1; }
  read -r NC FID FDD LATENT <<< "$meta"
  local DATA_ROOT="$DATA_ROOT_BASE/$LATENT"
  local sess="retroeval_gpu${gpu}_${ds}_${cfg##*_}"
  # dedup screen name collisions across families (caimf vs afm) by tagging suffix
  sess="retroeval_gpu${gpu}_$(basename "$wd")"
  log "LAUNCH retro-eval $ds on GPU $gpu (cfg=$cfg) wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    CONFIG_MODE=$cfg PYTHON=$PY USE_WANDB=False \
    CUDA_VISIBLE_DEVICES=$gpu \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MPLCONFIGDIR=/tmp/mpl-retroeval-$ds \
    bash scripts/eval_best_fid_steps_plain_imf.sh '$wd' $STEPS -- \
      --config.dataset.name='${ds}_latent' \
      --config.dataset.root='$DATA_ROOT' \
      --config.dataset.class_mapping_root='' \
      --config.dataset.num_classes=$NC \
      --config.model.num_classes=$NC \
      --config.sampling.num_classes=$NC \
      --config.fid.cache_ref='$ADV/files/fid_stats/$FID' \
      --config.fd_dino.cache_ref='$ADV/files/fdd_stats/$FDD' \
      2>&1 | tee -a $ADV/files/logs/retroeval_$(basename "$wd").log
  "
  echo "$sess"
}

declare -A STABLE=() CLAIMED=()   # gpu -> entry
QUEUE=()
log "=== retro-eval driver started (PERSISTENT): catches early-stopped ${FAMILIES[*]%%:*} runs; NFE {$STEPS} 10k; rescans every ${POLL_S}s; exits when no eval work AND no training remains ==="
announced_idle=0

while :; do
  gpu_gc_locks
  # reap finished eval screens
  for idx in "${!CLAIMED[@]}"; do
    entry="${CLAIMED[$idx]}"; wd="${entry#*|}"; wd="${wd%%|*}"; ds="${entry%%|*}"
    sess="retroeval_gpu${idx}_$(basename "$wd")"
    if ! screen -ls 2>/dev/null | grep -qE "\.${sess}\b"; then
      if eval_done "$wd"; then log "DONE $ds ($(basename "$wd")) GPU $idx freed"; else log "WARN $ds eval screen gone but csv incomplete -- check retroeval_$(basename "$wd").log"; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done

  # rebuild queue of runs still needing eval (excludes claimed/in-flight/still-training)
  build_queue
  # drop entries whose workdir we're already evaluating (in CLAIMED)
  PENDING=()
  for entry in "${QUEUE[@]}"; do
    e_wd="${entry#*|}"; e_wd="${e_wd%%|*}"; b="$(basename "$e_wd")"
    dup=0
    for c in "${CLAIMED[@]}"; do cwd="${c#*|}"; cwd="${cwd%%|*}"; [[ "$(basename "$cwd")" == "$b" ]] && dup=1; done
    (( dup )) || PENDING+=("$entry")
  done

  # exit condition: nothing pending, nothing in flight, no training that could produce more
  if (( ${#PENDING[@]} == 0 && ${#CLAIMED[@]} == 0 )); then
    if training_active; then
      if (( announced_idle == 0 )); then log "IDLE: no eval work now, but training still running -- staying alive to catch early-stops."; announced_idle=1; fi
      sleep "$POLL_S"; continue
    else
      log "=== all eval work done and no training active; driver exiting ==="; break
    fi
  fi
  announced_idle=0

  # find free GPUs
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  idle_now=0
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    if (( mem < MEM_MB && util < UTIL_PCT )) && ! gpu_locked "$idx"; then idle_now=$((idle_now+1)); fi
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if gpu_locked "$idx"; then STABLE[$idx]=0; continue; fi
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  # assign pending entries to stable-free GPUs (RESERVE_FREE=0 -> evals grab all)
  avail=$(( idle_now - RESERVE_FREE )); pj=0
  for idx in $ALLOWED; do
    (( pj < ${#PENDING[@]} )) || break
    (( avail > 0 )) || break
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    entry="${PENDING[$pj]}"
    e_wd="${entry#*|}"; e_wd="${e_wd%%|*}"
    sess="retroeval_gpu${idx}_$(basename "$e_wd")"
    gpu_try_claim "$idx" "$sess" || { STABLE[$idx]=0; continue; }
    if launch "$idx" "$entry" >/dev/null; then CLAIMED[$idx]="$entry"; pj=$((pj+1)); avail=$((avail-1)); log "PROGRESS ${entry%%|*} -> GPU $idx; pending-left $(( ${#PENDING[@]} - pj )); idle-was $idle_now"; sleep 15; else gpu_release "$idx"; fi
  done
  sleep "$POLL_S"
done
