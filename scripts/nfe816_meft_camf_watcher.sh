#!/usr/bin/env bash
# =============================================================================
# NFE 8 & 16 eval-only sweep for MF-T + CAMF (CA-iMF pure-adv) runs, DiT + JiT,
# all 5 datasets. Detached: screen nfe816_watcher.
#
# These are EVAL-ONLY replays off each run's existing best_fid checkpoint --
# no training. For the main table we already have NFE 4 (DiT/JiT) and NFE 1&2
# (final_eval_metrics.csv); this adds the 8- and 16-step columns.
#
# Two proven eval paths (verified against the runs' existing 1/2/4 evals):
#   DiT (latent):  main.py eval_only, config caltech_dit_meft_caimf_posttrain,
#                  VAE/latent. One pass PER step (sampling.num_steps +
#                  force_metric_num_steps both set), into eval_nfe{N} subdirs.
#   JiT (pixel):   main_imf_jit.py eval_only, config caltech_jit_dmf_meft,
#                  no VAE. force_metric_num_steps="8 16" -> both in ONE pass,
#                  into an eval_nfe8_16 subdir.
#
# 10k samples (standing rule). NEVER touches final_eval_metrics.csv (NFE1&2) or
# the runs' own eval_metrics.csv -- all output goes to fresh eval_nfe* subdirs.
# One GPU/run, opportunistic: claims GPUs AS THEY FREE via the shared atomic
# lock, so it coexists with the imnetcaimf + afmredo training runs and never
# double-claims. No early-stop (eval-only).
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$ADV"
LOG="$ADV/files/logs/nfe816_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
NSAMP=10000

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
source "$ADV/scripts/gpu_lock.inc.sh"

# kind:ds  -- 10 jobs (DiT x5, JiT x5). CAMF pure-adv workdirs (STAMP 20260726).
QUEUE=(
  "dit:artbench10" "dit:caltech101" "dit:cub200" "dit:food101" "dit:stanfordcars"
  "jit:artbench10" "jit:caltech101" "jit:cub200" "jit:food101" "jit:stanfordcars"
)

wd_for(){  # $1=kind $2=ds
  local famd; [[ "$1" == "dit" ]] && famd=DiT || famd=JiT
  echo "$ADV/files/logs/finetuning/${2}_${famd}_MeFT_CAIMF_puresadv_20260726"
}

# per-dataset dataset params
dit_case(){ case "$1" in
  artbench10)   LAT=artbench-10_processed_latents;  NC=10;  FID=artbench-10_processed-fid_stats.npz;  FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   LAT=caltech-101_processed_latents;  NC=101; FID=caltech-101-fid_stats.npz;            FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       LAT=cub-200-2011_processed_latents; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      LAT=food-101_processed_latents;     NC=101; FID=food-101_processed-fid_stats.npz;     FDD=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) LAT=stanford-cars_processed_latents; NC=196; FID=stanford_cars_processed-fid_stats.npz; FDD=stanford-cars-fd_dino-vitb14_stats.npz ;;
esac; }
jit_case(){ case "$1" in
  artbench10)   IMG=artbench-10_images;  NC=10;  FID=artbench-10_processed-fid_stats.npz;  FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   IMG=caltech-101_images;  NC=101; FID=caltech-101-fid_stats.npz;            FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       IMG=cub-200-2011_images; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      IMG=food-101_images;     NC=101; FID=food-101_processed-fid_stats.npz;     FDD=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) IMG=stanford-cars_images; NC=196; FID=stanford_cars_processed-fid_stats.npz; FDD=stanford-cars-fd_dino-vitb14_stats.npz ;;
esac; }

has_step(){ awk -F, -v s="$2" 'NR>1 && $4==s{f=1} END{exit !f}' "$1" 2>/dev/null; }  # $1=csv $2=step

train_done(){  # $1=kind $2=ds  -> both 8 & 16 present
  local wd; wd=$(wd_for "$1" "$2")
  screen -ls | grep -qE "nfe816_gpu[0-9]+_${1}_${2}\b" && return 1
  if [[ "$1" == "dit" ]]; then
    has_step "$wd/eval_nfe8/eval_metrics.csv" 8 && has_step "$wd/eval_nfe16/eval_metrics.csv" 16
  else
    has_step "$wd/eval_nfe8_16/eval_metrics.csv" 8 && has_step "$wd/eval_nfe8_16/eval_metrics.csv" 16
  fi
}

ckpt_of(){ ls -d "$1/best_fid/checkpoint_"* 2>/dev/null | sort -t_ -k2 -n | tail -1; }

launch(){  # $1=gpu $2=entry(kind:ds)
  local gpu="$1"; IFS=':' read -r kind ds <<< "$2"
  local wd; wd=$(wd_for "$kind" "$ds")
  local ckpt; ckpt=$(ckpt_of "$wd")
  [[ -n "$ckpt" && -d "$ckpt" ]] || { log "ERROR [$kind/$ds] no best_fid ckpt in $wd"; return 1; }
  local sess="nfe816_gpu${gpu}_${kind}_${ds}"
  if [[ "$kind" == "dit" ]]; then
    dit_case "$ds"; local ROOT="$DATA/$LAT"
    log "LAUNCH $sess (DiT latent 8&16) ckpt=$ckpt"
    screen -dmS "$sess" bash -c "
      cd $ADV
      export CUDA_VISIBLE_DEVICES=$gpu
      export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
      export MPLCONFIGDIR=/tmp/mpl-nfe816-$kind-$ds
      for N in 8 16; do
        ED=$wd/eval_nfe\$N; mkdir -p \$ED
        $PY main.py \
          --workdir=\$ED \
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
          --config.sampling.num_steps=\$N \
          --config.training.force_metric_num_steps=\$N \
          --config.fid.cache_ref=$ADV/files/fid_stats/$FID \
          --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD \
          --config.fid.num_samples=$NSAMP \
          --config.logging.use_wandb=False \
          2>&1 | tee -a $ADV/files/logs/nfe816_${kind}_${ds}.log
      done
    "
  else
    jit_case "$ds"; local ROOT="$DATA/$IMG"; local ED="$wd/eval_nfe8_16"
    log "LAUNCH $sess (JiT pixel 8&16 one pass) ckpt=$ckpt"
    screen -dmS "$sess" bash -c "
      cd $MAIN
      export CUDA_VISIBLE_DEVICES=$gpu
      export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
      export MPLCONFIGDIR=/tmp/mpl-nfe816-$kind-$ds
      mkdir -p $ED
      $PY main_imf_jit.py \
        --workdir=$ED \
        --config=$MAIN/configs/load_config.py:caltech_jit_dmf_meft \
        --config.eval_only=True \
        --config.load_from=$ckpt \
        --config.dataset.num_classes_from_data=False \
        --config.dataset.root=$ROOT \
        --config.dataset.num_classes=$NC \
        --config.model.num_classes=$NC \
        --config.fid.cache_ref=$MAIN/files/fid_stats/$FID \
        --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/$FDD \
        --config.logging.use_wandb=False \
        --config.training.force_metric_num_steps=\"8 16\" \
        --config.fid.num_samples=$NSAMP \
        2>&1 | tee -a $ADV/files/logs/nfe816_${kind}_${ds}.log
    "
  fi
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== NFE 8&16 eval watcher: ${#QUEUE[@]} jobs (DiT+JiT x5), pool {$ALLOWED}, 10k samples, eval-only, GPU-locked, opportunistic ==="

for entry in "${QUEUE[@]}"; do IFS=':' read -r k d <<< "$entry"; if train_done "$k" "$d"; then DONE[$entry]=1; log "SKIP $entry (8&16 already present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ nfe816_gpu([0-9]+)_([a-z]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; entry="${BASH_REMATCH[2]}:${BASH_REMATCH[3]}"
  CLAIMED[$gidx]="$entry"; gpu_try_claim "$gidx" "$sess" >/dev/null 2>&1
  log "ADOPT in-flight $entry on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "nfe816_gpu[0-9]+_[a-z]+_[a-z0-9]+")

remaining(){ local n=0; for e in "${QUEUE[@]}"; do [[ -z "${DONE[$e]:-}" ]] && n=$((n+1)); done; echo "$n"; }

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
  # reap finished
  for idx in "${!CLAIMED[@]}"; do
    entry="${CLAIMED[$idx]}"; IFS=':' read -r k d <<< "$entry"
    if ! screen -ls | grep -qE "nfe816_gpu${idx}_${k}_${d}\b"; then
      if train_done "$k" "$d"; then DONE[$entry]=1; log "COMPLETE $entry (GPU $idx freed)"; else log "WARN $entry screen gone but 8&16 not both present -- check nfe816_${k}_${d}.log"; DONE[$entry]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # assign
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      [[ -n "${DONE[$entry]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$entry" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "nfe816_gpu${idx}_${entry/:/_}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$entry"; log "PROGRESS $entry -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all NFE 8&16 evals complete; watcher exiting ==="
