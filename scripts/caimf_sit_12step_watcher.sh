#!/usr/bin/env bash
# Final eval of the 5 SiT-MeFT CA-iMF best_fid checkpoints at NFE=1 AND 2 (10k).
# CA-iMF SiT runs tracked only NFE=4 during training; the 4-step FID/FDD/IS is
# taken from those training-time evals. This fills the 1- and 2-step rows.
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python
DATA_ROOT=/opt/dlami/nvme/meanflow/datasets
cd "$ADV"
LOG="$ADV/files/logs/caimf_sit_12step_watcher.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

POLL_S=60; STABLE_NEEDED=3; MEM_MB=3000; UTIL_PCT=15; ALLOWED="0 1 2 3 4 5 6 7"
DSES=(artbench10 caltech101 cub200 food101 stanfordcars)
wd_for(){ echo "$ADV/files/logs/finetuning/${1}_SiT_MeFT_CAIMF_puresadv_20260725"; }

# per-dataset (latent dir, num_classes, fid stat, fdd stat)
paths_for(){ case "$1" in
  caltech101)   echo "caltech-101_processed_latents 101 caltech-101-fid_stats.npz caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   echo "artbench-10_processed_latents 10 artbench-10_processed-fid_stats.npz artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       echo "cub-200-2011_processed_latents 200 cub-200-2011_processed-fid_stats.npz cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      echo "food-101_processed_latents 101 food-101_processed-fid_stats.npz food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) echo "stanford-cars_processed_latents 196 stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz" ;;
esac; }

# done when BOTH 1-step and 2-step csvs exist
done_for(){ local wd; wd=$(wd_for "$1"); [[ -f "$wd/eval_best_fid_1steps/eval_metrics.csv" && -f "$wd/eval_best_fid_2steps/eval_metrics.csv" ]]; }

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2" wd; wd=$(wd_for "$ds")
  [[ -d "$wd/best_fid" ]] || { log "ERROR [$ds] no best_fid"; return 1; }
  read -r LATENT NC FID FDD <<< "$(paths_for "$ds")"
  local sess="c12_gpu${gpu}_${ds}"
  log "LAUNCH 1&2-step eval $ds on GPU $gpu (screen $sess) NC=$NC"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu PYTHON=$PY USE_WANDB=False CONFIG_MODE=caltech_sit_meft_caimf_posttrain
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-c12-$ds
    bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh '$wd' 1 2 -- \
      --config.dataset.name=${ds}_latent \
      --config.dataset.root=$DATA_ROOT/$LATENT \
      --config.dataset.class_mapping_root= \
      --config.dataset.num_classes=$NC \
      --config.model.num_classes=$NC \
      --config.sampling.num_classes=$NC \
      --config.fid.cache_ref=$ADV/files/fid_stats/$FID \
      --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD \
      2>&1 | tee -a $ADV/files/logs/caimf_sit_12step_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
for d in "${DSES[@]}"; do done_for "$d" && { DONE[$d]=1; log "SKIP $d (1&2-step present)"; }; done
remaining(){ local n=0; for d in "${DSES[@]}"; do [[ -z "${DONE[$d]:-}" ]] && n=$((n+1)); done; echo "$n"; }

log "=== SiT CA-iMF 1&2-step watcher: $(remaining) datasets, gate $STABLE_NEEDED ==="
while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line"|awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line"|awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line"|awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0}+1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if ! screen -ls | grep -qE "c12_gpu${idx}_${ds}\b"; then
      if done_for "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx)"; else log "WARN $ds screen gone, missing csv -- check caimf_sit_12step_${ds}.log"; DONE[$ds]=1; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for ds in "${DSES[@]}"; do
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; sleep 15; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all 5 SiT CA-iMF 1&2-step evals done ==="
