#!/usr/bin/env bash
# =============================================================================
# DiT MF-T NFE-8 eval, ONLINE weights (2026-07-28 REDO). Detached: screen
# dit_mft_nfe8_online_watcher.
#
# WHY: the earlier dit_mft_nfe8_watcher used main.py with the CAMF-posttrain
# config, which defaulted to metric_mode=EMA. These MF-T runs train with
# use_ema=False, so the EMA shadow is garbage -> every NFE collapsed to noise
# (IS~3, FDD~3000). The GOOD NFE4 column was produced by eval_best_fid_steps.sh,
# which forces --config.training.use_ema=False (metric_mode=online). This redo
# drives that SAME proven script at NFE 8 only, so it byte-matches the NFE4
# recipe (config caltech_dit_dmf_ddpmv, noise/dit_native, omega=1.5, XL_2,
# null-class, cdp=0.1, use_ema_vc=False) -- only num_steps differs.
#
# Per-dataset source = the EXACT checkpoint that produced the table's NFE4 cell
# (mixed provenance: artbench/food use meft_online, others use clip1). Output ->
# fresh eval_best_fid_8steps/ inside each source run dir (no clobber). 10k
# samples (FID_NUM_SAMPLES=10000). One GPU/run via shared atomic lock;
# opportunistic on the free pool.
# =============================================================================
set -uo pipefail
MAIN=/opt/dlami/nvme/meanflow/imeanflow
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
PY=$MAIN/.venv/bin/python
cd "$MAIN"
LOG="$ADV/files/logs/dit_mft_nfe8_online_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 4"
NSAMP=10000
STEPS="8"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200 food101 stanfordcars)

# ds -> "run_dir_basename:latent_root:num_classes:fid_npz:fdd_npz"
# run_dir = the source of the table's NFE4 cell (the exact checkpoint under best_fid)
entry_for(){ case "$1" in
  artbench10)   echo "artbench10_DiT_DMF_meanflow_taylor_plain_online_meft_online_20260722_171741_wp7190:artbench-10_processed_latents:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz" ;;
  caltech101)   echo "caltech101_DiT_DMF_meanflow_taylor_plain_online_clip1_caltech101_20260724_141146_om01oq:caltech-101_processed_latents:101:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz" ;;
  cub200)       echo "cub200_DiT_DMF_meanflow_taylor_plain_online_clip1_cub200_20260724_141206_euwxyx:cub-200-2011_processed_latents:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      echo "food101_DiT_DMF_meanflow_taylor_plain_online_meft_online_20260722_171747_ovqom3:food-101_processed_latents:101:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) echo "stanfordcars_DiT_DMF_meanflow_taylor_plain_online_clip1_stanfordcars_20260724_141226_17ll6g:stanford-cars_processed_latents:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz" ;;
esac; }

rundir_for(){ IFS=':' read -r rd _ <<< "$(entry_for "$1")"; echo "$MAIN/files/logs/finetuning/$rd"; }
edir_for(){   echo "$(rundir_for "$1")/eval_best_fid_${STEPS}steps"; }

has_step(){ awk -F, -v s="$2" 'NR>1 && $4==s{f=1} END{exit !f}' "$1" 2>/dev/null; }

train_done(){  # $1=ds -> NFE8 row present in its eval_best_fid_8steps csv
  screen -ls | grep -qE "ditmft8o_gpu[0-9]+_${1}\b" && return 1
  has_step "$(edir_for "$1")/eval_metrics.csv" "$STEPS"
}

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"
  IFS=':' read -r rd root nc fid fdd <<< "$(entry_for "$ds")"
  local wd="$MAIN/files/logs/finetuning/$rd"
  [[ -d "$wd/best_fid" ]] || { log "ERROR [$ds] $wd/best_fid missing"; return 1; }
  local sess="ditmft8o_gpu${gpu}_${ds}"
  log "LAUNCH $sess (DiT MF-T ONLINE NFE$STEPS, omega=1.5) src=$rd -> eval_best_fid_${STEPS}steps"
  screen -dmS "$sess" bash -c "
    cd $MAIN
    CONFIG_MODE=caltech_dit_dmf_ddpmv \
    PYTHON=$PY \
    USE_WANDB=False \
    MODEL_STR=imfDiT_DMF_XL_2 \
    MODEL_USE_DOGFIT=False \
    TARGET_USE_NULL_CLASS=True \
    CLASS_DROPOUT_PROB=0.1 \
    TARGET_OUTPUT_PREDICTION_SPACE=noise \
    TARGET_VELOCITY_MAP_MODE=dit_native \
    USE_EMA_VC=False \
    DATASET_ROOT=$MAIN/../datasets/$root \
    DATASET_NUM_CLASSES=$nc \
    FID_CACHE_REF=$MAIN/files/fid_stats/$fid \
    FD_DINO_CACHE_REF=$MAIN/files/fdd_stats/$fdd \
    FID_NUM_SAMPLES=$NSAMP \
    CUDA_VISIBLE_DEVICES=$gpu \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash scripts/eval_best_fid_steps.sh \"files/logs/finetuning/$rd\" $STEPS \
      2>&1 | tee -a $ADV/files/logs/ditmft8o_${ds}.log
    source $ADV/scripts/gpu_lock.inc.sh; gpu_release $gpu
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== DiT MF-T NFE$STEPS ONLINE eval: ${#QUEUE[@]} datasets, pool {$ALLOWED}, 10k samples, use_ema=False, GPU-locked, opportunistic ==="
for ds in "${QUEUE[@]}"; do if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (NFE$STEPS already present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ ditmft8o_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "$sess" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "ditmft8o_gpu[0-9]+_[a-z0-9]+")

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
    if ! screen -ls | grep -qE "ditmft8o_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but NFE$STEPS not present -- check ditmft8o_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "ditmft8o_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; unset 'CLAIMED[$idx]'; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all DiT MF-T NFE$STEPS ONLINE evals complete; watcher exiting ==="
