#!/usr/bin/env bash
# =============================================================================
# JiT NFE 8&16 IMAGE-ONLY rerun (2026-07-28). Detached: screen jitimg_watcher.
#
# The JiT NFE8&16 eval (train_imf_jit.py:just_evaluate) computed metrics only --
# it never saved sample grids (unlike the DiT path). train_imf_jit.py has now
# been patched to save a preview grid per NFE. This re-runs the 5 JiT best_fid
# CAMF checkpoints at NFE 8 and 16 to RENDER those grids.
#
# IMAGES-ONLY: --config.fid.num_samples=16 (just enough for the 4x4=16 preview
# grid; num_images_to_log=16). This does NOT recompute the real 10k-sample
# metrics -- output goes to a FRESH eval_nfe8_16_imgs/ subdir, leaving each
# run's eval_nfe8_16/eval_metrics.csv (the true 10k numbers) UNTOUCHED.
#
# One pass renders both NFE 8 and 16 (force_metric_num_steps="8 16"). Grids land
# in eval_nfe8_16_imgs/images/<step>_image_grid_steps_8.png and ..._steps_16.png.
#
# GPU pool RESTRICTED to 1 (the only free GPU; 0/2/3/4/5=imnet-CAMF 300k,
# 6/7=ArtBench cpabl). Sequential, claims GPU 1 off the shared atomic lock.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$MAIN"
LOG="$ADV/files/logs/jit_nfe816_images_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=2
MEM_MB=3000
UTIL_PCT=15
ALLOWED="1"
NSAMP=16                 # images-only: tiny sample count, just renders the grid
DATA=/opt/dlami/nvme/meanflow/datasets

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200 food101 stanfordcars)

jit_case(){ case "$1" in
  artbench10)   IMG=artbench-10_images;   NC=10  ;;
  caltech101)   IMG=caltech-101_images;   NC=101 ;;
  cub200)       IMG=cub-200-2011_images;  NC=200 ;;
  food101)      IMG=food-101_images;      NC=101 ;;
  stanfordcars) IMG=stanford-cars_images; NC=196 ;;
esac; }

wd_for(){ echo "$ADV/files/logs/finetuning/${1}_JiT_MeFT_CAIMF_puresadv_20260726"; }
ckpt_of(){ ls -d "$(wd_for "$1")/best_fid/checkpoint_"* 2>/dev/null | head -1; }
img_dir(){ echo "$(wd_for "$1")/eval_nfe8_16_imgs"; }

# done when both step_8 and step_16 grids exist
imgs_done(){  # $1=ds
  local ed; ed=$(img_dir "$1")
  ls "$ed/images/"*_image_grid_steps_8.png  >/dev/null 2>&1 || return 1
  ls "$ed/images/"*_image_grid_steps_16.png >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "jitimg_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"; jit_case "$ds"
  local ckpt; ckpt=$(ckpt_of "$ds")
  [[ -n "$ckpt" && -d "$ckpt" ]] || { log "ERROR [$ds] no best_fid ckpt"; return 1; }
  local ed; ed=$(img_dir "$ds"); local ROOT="$DATA/$IMG"
  local sess="jitimg_gpu${gpu}_${ds}"
  log "LAUNCH $sess (JiT NFE8&16 images) ckpt=$ckpt -> $ed"
  screen -dmS "$sess" bash -c "
    cd $MAIN
    export CUDA_VISIBLE_DEVICES=$gpu
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-jitimg-$ds
    mkdir -p $ed
    $PY main_imf_jit.py \
      --workdir=$ed \
      --config=$MAIN/configs/load_config.py:caltech_jit_dmf_meft \
      --config.eval_only=True \
      --config.load_from=$ckpt \
      --config.dataset.num_classes_from_data=False \
      --config.dataset.root=$ROOT \
      --config.dataset.num_classes=$NC \
      --config.model.num_classes=$NC \
      --config.logging.use_wandb=False \
      --config.training.force_metric_num_steps=\"8 16\" \
      --config.fid.num_samples=$NSAMP \
      --config.fid.num_images_to_log=16 \
      2>&1 | tee -a $ADV/files/logs/jitimg_${ds}.log
    source $ADV/scripts/gpu_lock.inc.sh; gpu_release $gpu
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== JiT NFE8&16 image rerun: ${#QUEUE[@]} datasets, GPU {$ALLOWED}, images-only (NSAMP=$NSAMP), fresh eval_nfe8_16_imgs/, metrics UNTOUCHED ==="
for ds in "${QUEUE[@]}"; do if imgs_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (grids present)"; fi; done
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
    if ! screen -ls | grep -qE "jitimg_gpu${idx}_${ds}\b"; then
      if imgs_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but grids missing -- check jitimg_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "jitimg_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; unset 'CLAIMED[$idx]'; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all JiT NFE8&16 image reruns complete; watcher exiting ==="
