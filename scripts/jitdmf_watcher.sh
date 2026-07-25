#!/usr/bin/env bash
# =============================================================================
# JiT DMF (transport-space) rerun watcher -- detached (screen jitdmf_watcher).
# caltech already launched manually on GPU7. This queues the REMAINING 4
# datasets and drops one per stably-free GPU as capacity frees (DiT clip reruns
# on 0/1/2/3/5 finish at 30k; JiT-plain on 4/6 finish at 40k).
#
# JiT DMF = imfJiT_DMF_H_16, pixel-space ImageFolder, transport space
# (data->velocity). clip=1.0 (x->v conversion => same amplification risk), 40k
# steps (JiT horizon), no-EMA, 4-step FID every 2500 (force_fid_per_step).
# CRITICAL: config flag MUST be load_config.py:MODE, NOT the raw .yml (ml_collections
# DEFINE_config_file execs a .yml as Python -> SyntaxError). Screens jitdmf_gpu<N>_<ds>.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
LOG="$REPO/files/logs/jitdmf_watcher.log"

POLL_S=90
STABLE_NEEDED=6      # ~9 min free (outlast DiT train->final-eval gaps)
MEM_MB=3000
UTIL_PCT=15
CLIP=1.0
STEPS=40000
ALLOWED="0 1 2 3 4 5 6 7"   # all GPUs authorized; caltech already on 7
WEIGHTS="$REPO/files/weights/JiT-H-16-256.pth"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# remaining 4 (caltech launched manually). dataset:imgbase:nc:fid:fdd
QUEUE=(
  "artbench10:artbench-10_images:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz"
  "cub200:cub-200-2011_images:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz"
  "food101:food-101_images:101:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz"
  "stanfordcars:stanford-cars_images:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz"
)

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds img nc fid fdd <<< "$2"
  local root="$DATA/$img"
  [[ -d "$root/train" ]] || { log "ERROR [$ds] $root/train missing"; return 1; }
  [[ -e "$REPO/files/fid_stats/$fid" ]] || { log "ERROR [$ds] missing fid $fid"; return 1; }
  [[ -e "$REPO/files/fdd_stats/$fdd" ]] || { log "ERROR [$ds] missing fdd $fdd"; return 1; }
  local wd="$REPO/files/workdirs/jitdmfclip_${ds}"; mkdir -p "$wd"
  local sess="jitdmf_gpu${gpu}_${ds}"
  log "LAUNCH $ds on GPU $gpu clip=$CLIP steps=$STEPS (screen $sess)"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/mpl-jitdmf-$ds
    .venv/bin/python main_imf_jit.py \
      --workdir=$wd \
      --config=configs/load_config.py:caltech_jit_dmf_meft \
      --config.dataset.root=$root \
      --config.dataset.num_classes=$nc \
      --config.model.num_classes=$nc \
      --config.fid.cache_ref=$REPO/files/fid_stats/$fid \
      --config.fd_dino.cache_ref=$REPO/files/fdd_stats/$fdd \
      --config.load_from=$WEIGHTS \
      --config.logging.use_wandb=False \
      --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
      --config.training.max_train_steps=$STEPS \
      --config.training.grad_clip_norm=$CLIP \
      --config.training.force_fid_per_step=2500 \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/jitdmf_${ds}_launch.log
  "
}

declare -A STABLE=() CLAIMED=()
QIDX=0
log "=== JiT DMF watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, ${STEPS}step clip $CLIP, need $STABLE_NEEDED free polls ==="
while (( QIDX < ${#QUEUE[@]} )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( QIDX < ${#QUEUE[@]} )) || break
    if (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )); then
      if launch "$idx" "${QUEUE[$QIDX]}"; then
        CLAIMED[$idx]=1; QIDX=$(( QIDX + 1 ))
        log "PROGRESS launched $QIDX/${#QUEUE[@]} on GPU $idx; remaining: ${QUEUE[*]:$QIDX}"
        sleep 10
      fi
    fi
  done
  (( QIDX < ${#QUEUE[@]} )) && sleep "$POLL_S"
done
log "=== all ${#QUEUE[@]} remaining JiT DMF runs launched; watcher exiting ==="
