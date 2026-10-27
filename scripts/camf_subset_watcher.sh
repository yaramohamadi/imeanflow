#!/usr/bin/env bash
# MF-T + CAMF on the seed-42 data subsets (appendix data-efficiency table), iMF, CUB + Cars at 10% / 25%.
# Copy of scripts/camf_ablation_watcher.sh: same config (configs/cub_imf_caimf_cpablation_config.yml, full
# score, cap 1.0 = the 100% CAMF run), 155K batches = 5K D warm-up + 30K G updates, patience-5 early stop on
# 4-step FID, then best checkpoint evaluated at NFE 1/2/4/8 with the full-dataset FID/FDD refs.
# Changes: source = the subset MF-T checkpoints (TwinMOS drive copies, Zahra's runs), training data = subset dir.
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/camf_subset_watcher.log"
POLL_S=90; STABLE_NEEDED=3; MEM_MB=3000; UTIL_PCT=15; PATIENCE=5
ALLOWED="${ALLOWED:-0 1 2 3 4 5 6 7}"
STAMP=20261027
DATA=/opt/dlami/nvme/meanflow/datasets
SRC=$MAIN/files/logs/finetuning
CKPT_cub200_10=$SRC/imf_mft_subset_cub200_10pct_drive/best_fid/checkpoint_5000
CKPT_cub200_25=$SRC/imf_mft_subset_cub200_25pct_drive/best_fid/checkpoint_5000
CKPT_stanfordcars_10=$SRC/imf_mft_subset_stanfordcars_10pct_drive/best_fid/checkpoint_5000
CKPT_stanfordcars_25=$SRC/imf_mft_subset_stanfordcars_25pct_drive/best_fid/checkpoint_15000
CARS_ARGS="--config.dataset.num_classes=196 --config.model.num_classes=196 --config.sampling.num_classes=196 --config.fid.cache_ref=$MAIN/files/fid_stats/stanford_cars_processed-fid_stats.npz --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz"
DS_ARGS_cub200_10="--config.dataset.root=$DATA/cub-200-2011_processed_latents_10pct_seed42"
DS_ARGS_cub200_25="--config.dataset.root=$DATA/cub-200-2011_processed_latents_25pct_seed42"
DS_ARGS_stanfordcars_10="--config.dataset.root=$DATA/stanford-cars_processed_latents_10pct_seed42 $CARS_ARGS"
DS_ARGS_stanfordcars_25="--config.dataset.root=$DATA/stanford-cars_processed_latents_25pct_seed42 $CARS_ARGS"
DS_ARGS_stanfordcars="--config.dataset.root=$DATA/stanford-cars_processed_latents $CARS_ARGS"
EVAL_ARGS_cub200="--config.dataset.root=$DATA/cub-200-2011_processed_latents --config.dataset.num_classes=200 --config.model.num_classes=200 --config.sampling.num_classes=200 --config.fid.cache_ref=$MAIN/files/fid_stats/cub-200-2011_processed-fid_stats.npz --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz"
EVAL_ARGS_stanfordcars="$DS_ARGS_stanfordcars"
# job = dataset:percent
JOBS=(cub200:10 cub200:25 stanfordcars:10 stanfordcars:25)
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wd_of(){ echo "$ADV/files/logs/finetuning/${1%%:*}_iMF_CAIMF_sub${1##*:}pct_s42_${STAMP}"; }
sess_of(){ echo "camfsub_${1%%:*}_${1##*:}"; }
has_best(){ ls -d "$(wd_of "$1")/best_fid/checkpoint_"* >/dev/null 2>&1; }
eval_done(){ local f; f="$(wd_of "$1")/final_eval_metrics.csv"; [[ -f "$f" ]] && (( $(grep -c . "$f") >= 5 )); }
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }
launch_train(){  # gpu job
  local gpu=$1 job=$2 ds=${2%%:*} p=${2##*:} wd s args="" ckpt ds_args v
  wd=$(wd_of "$job"); s=$(sess_of "$job")
  v="CKPT_${ds}_$p"; ckpt=${!v}; v="DS_ARGS_${ds}_$p"; ds_args=${!v}
  [[ -d "$ckpt" ]] || { log "ERROR ckpt missing $ckpt"; return 1; }
  mkdir -p "$wd"
  log "LAUNCH train $job on GPU $gpu ($s) wd=$wd args=[$args]"
  screen -dmS "$s" bash -c "cd $ADV; export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false REQUIRE_FD_DINO=1 MPLCONFIGDIR=/tmp/mpl-$s; $PY main_caimf.py --workdir='$wd' --config=$ADV/configs/load_config.py:cub_imf_caimf_cpablation --config.load_from='$ckpt' --config.logging.use_wandb=False $args $ds_args 2>&1 | tee -a $ADV/files/logs/$s.log"
}
launch_eval(){  # gpu job
  local gpu=$1 job=$2 ds=${2%%:*} wd s v
  wd=$(wd_of "$job"); s="$(sess_of "$job")_eval"; v="EVAL_ARGS_$ds"
  log "LAUNCH eval $job (NFE 1 2 4 8) on GPU $gpu ($s)"
  screen -dmS "$s" bash -c "cd $ADV; export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false; CONFIG_MODE=cub_imf_caimf_cpablation PYTHON=$PY USE_WANDB=False bash $ADV/scripts/eval_best_fid_steps_plain_imf.sh '$wd' 1 2 4 8 -- --config.dataset.name=${ds}_latent --config.dataset.class_mapping_root= ${!v} 2>&1 | tee -a $ADV/files/logs/$s.log"
}
declare -A STABLE=() CLAIMED=() PHASE=() DONE=()
for j in "${JOBS[@]}"; do if eval_done "$j"; then DONE[$j]=1; log "SKIP $j (already evaluated)"; fi; done
remaining(){ local n=0 j; for j in "${JOBS[@]}"; do [[ -z "${DONE[$j]:-}" ]] && n=$((n+1)); done; echo $n; }
log "=== CAMF subset watcher: ${#JOBS[@]} jobs, GPUs [$ALLOWED], patience $PATIENCE ==="
while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    IFS=', ' read -r idx mem util <<< "$line"
    [[ " $ALLOWED " == *" $idx "* ]] || continue; [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    job=${CLAIMED[$idx]}; s=$(sess_of "$job")
    if [[ "${PHASE[$idx]}" == train ]]; then
      if screen -ls | grep -qE "\.${s}[[:space:]]"; then
        csv=$(wd_of "$job")/eval_metrics.csv; [[ -f "$csv" ]] || continue
        series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .); (( n < PATIENCE + 1 )) && continue
        streak=$(echo "$series" | consec_rises_from_min); (( streak >= PATIENCE )) || continue
        log "EARLY-STOP $job: 4-step FID rose $streak consecutive evals (min=$(echo "$series" | sort -g | head -1), n=$n)"
        screen -S "$s" -X quit; sleep 5
      fi
      if has_best "$job"; then launch_eval "$idx" "$job"; PHASE[$idx]=eval
      else log "WARN $job: train screen gone and no best_fid; see files/logs/$s.log"; DONE[$job]=1; unset "CLAIMED[$idx]"; STABLE[$idx]=0; fi
    else
      screen -ls | grep -qE "\.${s}_eval[[:space:]]" && continue
      if eval_done "$job"; then log "COMPLETE $job"; else log "WARN $job: eval ended without 4 NFE rows; see files/logs/${s}_eval.log"; fi
      DONE[$job]=1; unset "CLAIMED[$idx]"; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue; (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for job in "${JOBS[@]}"; do
      [[ -n "${DONE[$job]:-}" ]] && continue
      busy=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$job" ]] && busy=1; done; (( busy )) && continue
      if launch_train "$idx" "$job"; then CLAIMED[$idx]=$job; PHASE[$idx]=train; STABLE[$idx]=0; sleep 20; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep $POLL_S
done
log "=== all CAMF subset jobs done ==="
