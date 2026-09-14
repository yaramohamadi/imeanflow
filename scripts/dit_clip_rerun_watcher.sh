#!/usr/bin/env bash
# =============================================================================
# DiT-DMF ddpm-v ONLINE + grad-clip reruns, all 5 datasets. Detached (screen
# ditclip_watcher) -> survives disconnect. Drops EMA entirely (online v_c);
# grad_clip_norm=1.0 is THE stabilizer (sanity confirmed loss_u stays ~10-40 vs
# unclipped 300-1800 + 1e10 spikes). 30k steps (DiT horizon; 40k is JiT-only).
#
# Byte-identical to the validated ONLINE DiT-DMF recipe (CONFIG_MODE
# caltech_dit_dmf_ddpmv, imfDiT_DMF_XL_2, no-dogfit, no-EMA, adam_b2 0.95, ga4
# bs8=eff32, lr1e-4, dit_native, null-class) EXCEPT the 3 required overrides:
#   --config.load_from=<local nvme DiT-XL>       (config default = dead taylor5 path)
#   --config.training.debug_log_during_train=False (single-GPU step-1 axis-3 crash)
#   --config.training.grad_clip_norm=1.0          (the fix under test)
# 4-step FID every 2500 (config fid_schedule), final eval 1/2/250, best_fid only.
#
# GPUs 0 and 7 are OFF-LIMITS (rule). Usable pool: 1,2,3,4,5,6. JiT holds 4,6 for
# now; watcher claims any stably-free GPU in the pool (~4.5min gate) until all 5
# launched. One dataset per GPU. Screens ditclip_gpu<N>_<ds>.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
LOG="$REPO/files/logs/ditclip_watcher.log"

POLL_S=60
STABLE_NEEDED=4      # ~4 min free before claiming (JiT->eval gaps already passed)
MEM_MB=3000
UTIL_PCT=15
CLIP=1.0
STEPS=30000
ALLOWED="0 1 2 3 5 7"   # user authorized ALL GPUs; 4,6 already busy with JiT
WEIGHTS="$REPO/files/weights/DiT-XL-2-256x256.pt"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# dataset : num_classes : fid_stats : fdd_stats : latent_dir
QUEUE=(
  "caltech101:101:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz:caltech-101_processed_latents"
  "artbench10:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz:artbench-10_processed_latents"
  "cub200:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz:cub-200-2011_processed_latents"
  "food101:101:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz:food-101_processed_latents"
  "stanfordcars:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz:stanford-cars_processed_latents"
)

launch(){  # $1=gpu $2=queue-entry
  local gpu="$1"; IFS=':' read -r ds nc fid fdd lat <<< "$2"
  local root="$DATA/$lat" fidp="$REPO/files/fid_stats/$fid" fddp="$REPO/files/fdd_stats/$fdd"
  [[ -d "$root" ]] || { log "ERROR [$ds] missing latents $root"; return 1; }
  [[ -e "$fidp" ]] || { log "ERROR [$ds] missing fid $fidp"; return 1; }
  [[ -e "$fddp" ]] || { log "ERROR [$ds] missing fdd $fddp"; return 1; }
  local sess="ditclip_gpu${gpu}_${ds}"
  log "LAUNCH $ds on GPU $gpu clip=$CLIP steps=$STEPS (screen $sess)"
  screen -dmS "$sess" bash -c "
    cd $REPO
    CONFIG_MODE=caltech_dit_dmf_ddpmv \
    PYTHON=$REPO/.venv/bin/python \
    USE_WANDB=True WANDB_PROJECT=dit_dmf_online_clip \
    DATASET_NAME=$ds DATASET_ROOT='$root' DATASET_NUM_CLASSES=$nc \
    FID_CACHE_REF='$fidp' FD_DINO_CACHE_REF='$fddp' \
    ENABLE_DOGFIT=False TRAIN_USE_EMA=False USE_EMA_VC=False \
    RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False \
    CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
    bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh clip1_${ds} \
      --config.load_from=$WEIGHTS \
      --config.training.debug_log_during_train=False \
      --config.training.grad_clip_norm=$CLIP \
      --config.training.max_train_steps=$STEPS \
      --config.fid.num_samples=5000 \
      --config.logging.wandb_group=dit_dmf_online_clip_20260724 \
      --config.logging.wandb_name=${ds}_online_clip1 \
      2>&1 | tee -a $REPO/files/logs/ditclip_${ds}_launch.log
  "
}

declare -A STABLE=() CLAIMED=()
QIDX=0
log "=== DiT clip rerun watcher: ${#QUEUE[@]} datasets, pool GPUs {$ALLOWED}, ${STEPS}step, clip $CLIP, need $STABLE_NEEDED free polls (~$((POLL_S*STABLE_NEEDED/60))min) ==="

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
log "=== all ${#QUEUE[@]} DiT clip reruns launched; watcher exiting ==="
