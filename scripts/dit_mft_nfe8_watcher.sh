#!/usr/bin/env bash
# =============================================================================
# DiT MF-T (pre-CAMF) NFE-8 eval-only sweep, all 5 datasets. Detached: screen
# dit_mft_nfe8_watcher.
#
# Purpose: the table shows DiT MF-T + CAMF at NFE 8, but the plain DiT MF-T row
# had no 8-step number. This evaluates the pre-CAMF DiT MF-T checkpoints (the
# online_clip1 best_fid ckpts that DiT CAMF was initialized from) at NFE 8,
# using the SAME config/eval path as the CAMF NFE8 sweep -- i.e. exactly how the
# DiT models were already trained and sampled (config caltech_dit_meft_caimf_
# posttrain carries the DiT operating point omega=1.5). Only load_from differs.
#
# EVAL-ONLY replay off each clip1 best_fid checkpoint. 10k samples (standing
# rule). Output -> fresh eval_mft_nfe8/ subdir inside each clip1 run dir; nothing
# existing is touched. One GPU/run, opportunistic via shared atomic lock, so it
# coexists with the training runs and never double-claims.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$ADV"
LOG="$ADV/files/logs/dit_mft_nfe8_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
NSAMP=10000
NFE=8

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

QUEUE=(artbench10 caltech101 cub200 food101 stanfordcars)

# ds -> clip1 run dir : best_fid step (the pre-CAMF DiT MF-T checkpoints)
ckpt_for(){ case "$1" in
  artbench10)   echo "$MAIN/files/logs/finetuning/artbench10_DiT_DMF_meanflow_taylor_plain_online_clip1_artbench10_20260724_141156_87ttqh/best_fid/checkpoint_12500" ;;
  caltech101)   echo "$MAIN/files/logs/finetuning/caltech101_DiT_DMF_meanflow_taylor_plain_online_clip1_caltech101_20260724_141146_om01oq/best_fid/checkpoint_30000" ;;
  cub200)       echo "$MAIN/files/logs/finetuning/cub200_DiT_DMF_meanflow_taylor_plain_online_clip1_cub200_20260724_141206_euwxyx/best_fid/checkpoint_27500" ;;
  food101)      echo "$MAIN/files/logs/finetuning/food101_DiT_DMF_meanflow_taylor_plain_online_clip1_food101_20260724_141216_68a834/best_fid/checkpoint_30000" ;;
  stanfordcars) echo "$MAIN/files/logs/finetuning/stanfordcars_DiT_DMF_meanflow_taylor_plain_online_clip1_stanfordcars_20260724_141226_17ll6g/best_fid/checkpoint_12500" ;;
esac; }

# eval-out dir = a fresh subdir INSIDE the clip1 run dir
ed_for(){ echo "$(dirname "$(dirname "$(ckpt_for "$1")")")/eval_mft_nfe8"; }

dit_case(){ case "$1" in
  artbench10)   LAT=artbench-10_processed_latents;   NC=10;  FID=artbench-10_processed-fid_stats.npz;   FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   LAT=caltech-101_processed_latents;   NC=101; FID=caltech-101-fid_stats.npz;             FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       LAT=cub-200-2011_processed_latents;  NC=200; FID=cub-200-2011_processed-fid_stats.npz;  FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      LAT=food-101_processed_latents;      NC=101; FID=food-101_processed-fid_stats.npz;      FDD=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) LAT=stanford-cars_processed_latents; NC=196; FID=stanford_cars_processed-fid_stats.npz; FDD=stanford-cars-fd_dino-vitb14_stats.npz ;;
esac; }

has_step(){ awk -F, -v s="$2" 'NR>1 && $4==s{f=1} END{exit !f}' "$1" 2>/dev/null; }

train_done(){  # $1=ds -> NFE8 row present
  screen -ls | grep -qE "ditmft8_gpu[0-9]+_${1}\b" && return 1
  has_step "$(ed_for "$1")/eval_metrics.csv" "$NFE"
}

launch(){  # $1=gpu $2=ds
  local gpu="$1" ds="$2"
  local ckpt; ckpt=$(ckpt_for "$ds")
  [[ -n "$ckpt" && -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  dit_case "$ds"; local ROOT="$DATA/$LAT"; local ED; ED=$(ed_for "$ds")
  local sess="ditmft8_gpu${gpu}_${ds}"
  log "LAUNCH $sess (DiT MF-T latent NFE$NFE, omega from config=1.5) ckpt=$ckpt -> $ED"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-ditmft8-$ds
    mkdir -p $ED
    $PY main.py \
      --workdir=$ED \
      --config=$ADV/configs/load_config.py:caltech_dit_meft_caimf_posttrain \
      --config.eval_only=True \
      --config.partial_load=False \
      --config.load_from=$ckpt \
      --config.dataset.name=${ds}_latent \
      --config.dataset.root=$ROOT \
      --config.dataset.class_mapping_root= \
      --config.dataset.num_classes=$NC \
      --config.model.num_classes=$NC \
      --config.sampling.num_classes=$NC \
      --config.sampling.num_steps=$NFE \
      --config.training.force_metric_num_steps=$NFE \
      --config.fid.cache_ref=$ADV/files/fid_stats/$FID \
      --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD \
      --config.fid.num_samples=$NSAMP \
      --config.logging.use_wandb=False \
      2>&1 | tee -a $ADV/files/logs/ditmft8_${ds}.log
    source $ADV/scripts/gpu_lock.inc.sh; gpu_release $gpu
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== DiT MF-T NFE$NFE eval watcher: ${#QUEUE[@]} datasets, pool {$ALLOWED}, 10k samples, eval-only, config omega=1.5, GPU-locked, opportunistic ==="
for ds in "${QUEUE[@]}"; do if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (NFE$NFE already present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ ditmft8_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "$sess" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "ditmft8_gpu[0-9]+_[a-z0-9]+")

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
    if ! screen -ls | grep -qE "ditmft8_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but NFE$NFE not present -- check ditmft8_${ds}.log"; DONE[$ds]=1; fi
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
      gpu_try_claim "$idx" "ditmft8_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$ds"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; unset 'CLAIMED[$idx]'; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all DiT MF-T NFE$NFE evals complete; watcher exiting ==="
