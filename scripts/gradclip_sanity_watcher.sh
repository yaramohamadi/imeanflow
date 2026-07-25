#!/usr/bin/env bash
# =============================================================================
# Grad-clip sanity watcher -- runs detached (screen gradclip_sanity), survives
# disconnect. Waits for 2 stably-free GPUs (never disturbs the running emavc
# evals / JiT trainings) then launches the ONLINE DiT-DMF ddpm-v recipe with
# global-norm gradient clipping = 1.0, capped at 20k steps.
#
#   GPU A: caltech101  (control: online was already ~stable-ish -> should stay flat)
#   GPU B: cub200      (worst-case: online + emavc both spiked/diverged)
#
# Everything else is BYTE-IDENTICAL to the validated online DiT-DMF recipe
# (config caltech_dit_dmf_ddpmv, imfDiT_DMF_XL_2, no-dogfit, no-EMA, dit_native
# ddpm-v, null-class, adam_b2 0.95, ga4 bs8=eff32, lr 1e-4). The ONLY deltas:
#   --config.training.grad_clip_norm=1.0   (the thing under test)
#   --config.training.max_train_steps=20000
# 4-step FID every 2500 (config fid_schedule). No final 1/2/250 eval (sanity).
# save_best_fid_only=True is inherited -> only best_fid checkpoint kept.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
LOG="$REPO/files/logs/gradclip_sanity_watcher.log"

POLL_S=90
STABLE_NEEDED=6      # ~9 min continuously free before claiming (outlast eval gaps)
MEM_MB=3000
UTIL_PCT=15
CLIP=1.0
STEPS=20000

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# dataset : num_classes : fid_stats : fdd_stats : latent_dir
QUEUE=(
  "caltech101:101:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz:caltech-101_processed_latents"
  "cub200:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz:cub-200-2011_processed_latents"
)

launch(){  # $1=gpu $2=queue-entry
  local gpu="$1"; IFS=':' read -r ds nc fid fdd lat <<< "$2"
  local root="$DATA/$lat"
  local fidp="$REPO/files/fid_stats/$fid" fddp="$REPO/files/fdd_stats/$fdd"
  [[ -d "$root" ]]  || { log "ERROR [$ds] missing latents $root"; return 1; }
  [[ -e "$fidp" ]]  || { log "ERROR [$ds] missing fid ref $fidp"; return 1; }
  [[ -e "$fddp" ]]  || { log "ERROR [$ds] missing fdd ref $fddp"; return 1; }
  local sess="gcsan_gpu${gpu}_${ds}"
  log "LAUNCH $ds on GPU $gpu clip=$CLIP steps=$STEPS (screen $sess)"
  screen -dmS "$sess" bash -c "
    cd $REPO
    CONFIG_MODE=caltech_dit_dmf_ddpmv \
    PYTHON=$REPO/.venv/bin/python \
    USE_WANDB=True WANDB_PROJECT=dit_dmf_gradclip_sanity \
    DATASET_NAME=$ds \
    DATASET_ROOT='$root' \
    DATASET_NUM_CLASSES=$nc \
    FID_CACHE_REF='$fidp' \
    FD_DINO_CACHE_REF='$fddp' \
    ENABLE_DOGFIT=False \
    TRAIN_USE_EMA=False \
    USE_EMA_VC=False \
    RUN_FINAL_BEST_FID_EVAL=False \
    CUDA_VISIBLE_DEVICES=$gpu \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
    bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh gradclip1_${ds} \
      --config.load_from=$REPO/files/weights/DiT-XL-2-256x256.pt \
      --config.logging.wandb_group=dit_dmf_gradclip_sanity_20260724 \
      --config.logging.wandb_name=${ds}_online_gradclip1 \
      --config.training.grad_clip_norm=$CLIP \
      --config.training.max_train_steps=$STEPS \
      --config.fid.num_samples=5000 \
      2>&1 | tee -a $REPO/files/logs/gcsan_${ds}_launch.log
  "
}

declare -A STABLE=() CLAIMED=()
QIDX=0
log "=== gradclip sanity watcher started; ${#QUEUE[@]} runs queued; need ${STABLE_NEEDED} free polls (~$((POLL_S*STABLE_NEEDED/60)) min) ==="

while (( QIDX < ${#QUEUE[@]} )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( QIDX < ${#QUEUE[@]} )) || break
    if (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )); then
      if launch "$idx" "${QUEUE[$QIDX]}"; then
        CLAIMED[$idx]=1; QIDX=$(( QIDX + 1 ))
        log "PROGRESS launched $QIDX/${#QUEUE[@]} on GPU $idx"
        sleep 8
      fi
    fi
  done
  sleep "$POLL_S"
done
log "=== both grad-clip sanity runs launched; watcher exiting ==="
