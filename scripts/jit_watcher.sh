#!/usr/bin/env bash
# =============================================================================
# JiT launch watcher -- runs detached (screen jit_watcher), survives disconnect.
#
# Launches plain-JiT fine-tunes on all 5 datasets, ONE per GPU, but only on GPUs
# that have been *stably* free. "Stably free" = mem < MEM_MB and util < UTIL_PCT
# for STABLE_NEEDED consecutive polls (POLL_S apart). The long window (~9 min)
# outlasts the brief train-end -> final-eval and final-eval -> 4-step-eval gaps
# on the ddpm-e GPUs (2,3,5) and the DiT-EMA GPUs (0,1,4,6,7), so we never grab a
# GPU during the couple-minute lull between a run's phases.
#
# Faithful naive JiT: flaxJiT_H_16, use_ema=False, 4-step during-training metric
# (config metric_num_steps=[4], sampling.num_steps=4), final eval 1/2/50, 40k
# steps, batch 32 fp32 adamw. Detached screen per run: jit_gpu<N>_<dataset>.
# =============================================================================
set -uo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
LOG="$REPO/files/logs/jit_watcher.log"

POLL_S=90            # seconds between polls
STABLE_NEEDED=6      # consecutive free polls required (~9 min) before claiming
MEM_MB=3000          # a GPU with less than this used MiB counts as idle
UTIL_PCT=15          # ...and util below this percent

JIT_WEIGHTS="$REPO/files/weights/JiT-H-16-256.pth"
[[ -f "$JIT_WEIGHTS" ]] || { echo "FATAL: missing JiT weights $JIT_WEIGHTS" | tee -a "$LOG"; exit 2; }

# Ensure /dev/shm large enough for PyTorch DataLoader workers.
if [[ "$(df --output=size -k /dev/shm | tail -1)" -lt 33554432 ]]; then
  mount -o remount,size=64g /dev/shm 2>/dev/null || echo "WARN: could not remount /dev/shm" | tee -a "$LOG"
fi

# Ordered launch queue: dataset : imgbase : num_classes(unused, from_data)
QUEUE=(
  "caltech101:caltech-101_images"
  "artbench10:artbench-10_images"
  "cub200:cub-200-2011_images"
  "food101:food-101_images"
  "stanfordcars:stanford-cars_images"
)
declare -A FID_REF=(
  [caltech101]="caltech-101-fid_stats.npz"
  [artbench10]="artbench-10_processed-fid_stats.npz"
  [cub200]="cub-200-2011_processed-fid_stats.npz"
  [food101]="food-101_processed-fid_stats.npz"
  [stanfordcars]="stanford_cars_processed-fid_stats.npz"
)
declare -A FDD_REF=(
  [caltech101]="caltech-101-fd_dino-vitb14_stats.npz"
  [artbench10]="artbench-10-fd_dino-vitb14_stats.npz"
  [cub200]="cub-200-2011-fd_dino-vitb14_stats.npz"
  [food101]="food-101-fd_dino-vitb14_stats.npz"
  [stanfordcars]="stanford-cars-fd_dino-vitb14_stats.npz"
)

declare -A STABLE=()   # gpu -> consecutive free-poll count
declare -A CLAIMED=()  # gpu -> 1 once we launched a JiT run on it
QIDX=0                 # next queue entry to launch

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ensure_images(){  # $1=imgbase ; extract zip if train/ missing
  local base="$1" root="$DATA/$1"
  if [[ ! -d "$root/train" ]]; then
    if [[ -f "$DATA/$base.zip" ]]; then
      log "extracting $base.zip ..."
      "$REPO/.venv/bin/python" -c "import zipfile; zipfile.ZipFile('$DATA/$base.zip').extractall('$DATA/')" \
        || { log "ERROR extracting $base.zip"; return 1; }
    else
      log "WAIT: images for $base not present yet (no train/ and no $base.zip) -- will retry"
      return 1
    fi
  fi
  [[ -d "$root/train" ]]
}

launch_jit(){  # $1=gpu $2=dataset $3=imgbase
  local gpu="$1" ds="$2" imgbase="$3"
  local root="$DATA/$imgbase"
  local fid="$REPO/files/fid_stats/${FID_REF[$ds]}"
  local fdd="$REPO/files/fdd_stats/${FDD_REF[$ds]}"
  [[ -e "$fid" ]] || { log "ERROR [$ds] missing fid ref $fid"; return 1; }
  [[ -e "$fdd" ]] || { log "ERROR [$ds] missing fdd ref $fdd"; return 1; }
  local sess="jit_gpu${gpu}_${ds}"
  log "LAUNCH $ds on GPU $gpu  (screen $sess)"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu
    export DATASET_NAME=$ds DATASET_ROOT='$root'
    export FID_CACHE_REF='$fid' FD_DINO_CACHE_REF='$fdd'
    export LOAD_FROM='$JIT_WEIGHTS'
    export PYTHON='$REPO/.venv/bin/python'
    export HALF_PRECISION=False SAMPLING_HALF_PRECISION=False
    export OPTIMIZER=adamw OPTIMIZER_MU_DTYPE=float32 TRAIN_BATCH_SIZE=32
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer='
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    export USE_WANDB=True WANDB_PROJECT=plain_jit_finetune
    export WANDB_NAME=${ds}_plain_jit_online
    export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 50' FINAL_EVAL_USE_WANDB=False
    bash scripts/train_plain_jit_finetune.sh online --config.logging.wandb_entity='' \
      --config.dataset.num_workers=16 \
      --config.dataset.prefetch_factor=4 \
      --config.dataset.pin_memory=True \
      --config.training.max_train_steps=40000 \
      --config.training.use_ema=False \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/jit_${ds}_launch.log
  "
}

log "=== JiT watcher started; ${#QUEUE[@]} datasets queued; poll ${POLL_S}s, need ${STABLE_NEEDED} free polls (~$((POLL_S*STABLE_NEEDED/60)) min) ==="

while (( QIDX < ${#QUEUE[@]} )); do
  # snapshot GPU mem+util
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then
      STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 ))
    else
      STABLE[$idx]=0
    fi
  done

  # claim the freest stably-idle GPU(s)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( QIDX < ${#QUEUE[@]} )) || break
    if (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )); then
      IFS=':' read -r ds imgbase <<< "${QUEUE[$QIDX]}"
      if ! ensure_images "$imgbase"; then
        # images not ready (e.g. stanford zip still uploading): skip this GPU
        # this poll, keep it counted free, try again next round.
        continue
      fi
      if launch_jit "$idx" "$ds" "$imgbase"; then
        CLAIMED[$idx]=1
        QIDX=$(( QIDX + 1 ))
        log "PROGRESS: launched $((QIDX))/${#QUEUE[@]} (${ds} on GPU ${idx}); remaining queue: ${QUEUE[*]:$QIDX}"
        sleep 8   # let it start allocating before re-polling
      fi
    fi
  done

  sleep "$POLL_S"
done

log "=== all ${#QUEUE[@]} JiT runs launched; watcher exiting ==="
