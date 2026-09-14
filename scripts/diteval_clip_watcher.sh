#!/usr/bin/env bash
# =============================================================================
# DiT MF-A (online+clip) FINAL EVAL fix -- detached (screen diteval_clip_watcher).
#
# The 5 clip1 training runs finished (best_fid saved) but their auto final-eval
# CRASHED: run_caltech_dit_dmf_meanflow_taylor.sh hardcodes
# TARGET_USE_NULL_CLASS=False / CLASS_DROPOUT_PROB=0.0, but training used the null
# class (target_use_null_class=true, class_dropout_prob=0.1) => 102-class embedder
# vs 101 => ScopeParamShapeError. This reruns the eval OFF the saved best_fid
# checkpoints with the PROVEN run_diteval recipe (null-class True, dit_native,
# ddpmv config that byte-matches training). NO retraining.
#
# Evals 1/2/4/250-step. GPU-gating: launches on any stably-free GPU; the 3 idle
# now (0/2/5) start immediately, datasets 4/5 land as those eval jobs finish and
# free their own GPU (~40min each). Screens diteval_gpu<N>_<ds>.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
cd "$REPO"
LOG="$REPO/files/logs/diteval_clip_watcher.log"
PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python

POLL_S=60
STABLE_NEEDED=2      # eval is short-lived; light gate so it grabs idle GPUs fast
MEM_MB=3000
UTIL_PCT=15
STEPS="1 2 4 250"
ALLOWED="0 1 2 3 4 5 6 7"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ds : ckpt_dir : root : nc : fid_npz : fdd_npz
QUEUE=(
  "caltech101:caltech101_DiT_DMF_meanflow_taylor_plain_online_clip1_caltech101_20260724_141146_om01oq:caltech-101_processed_latents:101:caltech-101-fid_stats.npz:caltech-101-fd_dino-vitb14_stats.npz"
  "cub200:cub200_DiT_DMF_meanflow_taylor_plain_online_clip1_cub200_20260724_141206_euwxyx:cub-200-2011_processed_latents:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz"
  "food101:food101_DiT_DMF_meanflow_taylor_plain_online_clip1_food101_20260724_141216_68a834:food-101_processed_latents:101:food-101_processed-fid_stats.npz:food-101-fd_dino-vitb14_stats.npz"
  "artbench10:artbench10_DiT_DMF_meanflow_taylor_plain_online_clip1_artbench10_20260724_141156_87ttqh:artbench-10_processed_latents:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz"
  "stanfordcars:stanfordcars_DiT_DMF_meanflow_taylor_plain_online_clip1_stanfordcars_20260724_141226_17ll6g:stanford-cars_processed_latents:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz"
)

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckptdir root nc fid fdd <<< "$2"
  local wd="$REPO/files/logs/finetuning/$ckptdir"
  [[ -d "$wd/best_fid" ]] || { log "ERROR [$ds] $wd/best_fid missing"; return 1; }
  local sess="diteval_gpu${gpu}_${ds}"
  log "LAUNCH eval $ds on GPU $gpu (screen $sess) ckpt=$ckptdir"
  screen -dmS "$sess" bash -c "
    cd $REPO
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
    DATASET_ROOT=$REPO/../datasets/$root \
    DATASET_NUM_CLASSES=$nc \
    FID_CACHE_REF=$REPO/files/fid_stats/$fid \
    FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd \
    CUDA_VISIBLE_DEVICES=$gpu \
    TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
    bash scripts/eval_best_fid_steps.sh \"files/logs/finetuning/$ckptdir\" $STEPS \
      2>&1 | tee -a $REPO/files/logs/diteval_clip_${ds}_eval.log
  "
}

declare -A STABLE=() CLAIMED=()
QIDX=0
log "=== DiT clip eval watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, steps [$STEPS], gate $STABLE_NEEDED ==="
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
        log "PROGRESS launched $QIDX/${#QUEUE[@]} on GPU $idx"
        sleep 15
      fi
    fi
  done
  (( QIDX < ${#QUEUE[@]} )) && sleep "$POLL_S"
done
log "=== all ${#QUEUE[@]} DiT clip evals launched; watcher exiting ==="
