#!/usr/bin/env bash
# CAMF core ablations (rebuttal, camf_rebuttal_short.md items 2 + 3) on iMF, CUB + Cars.
# Same operating point as the Tab. 3 CP ablation (configs/cub_imf_caimf_cpablation_config.yml,
# same MF-T source checkpoints, 155K batches = 5K D warm-up + 30K G updates, patience-5 early stop on
# 4-step FID); one switch changes per arm. After training, the best checkpoint is evaluated at NFE 1/2/4/8.
# The reference arm (score full, cap 1.0) is the existing main CAMF run of each dataset (Tab. 2 / Tab. 3).
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/camf_ablation_watcher.log"
POLL_S=90; STABLE_NEEDED=3; MEM_MB=3000; UTIL_PCT=15; PATIENCE=5
ALLOWED="${ALLOWED:-0 1 2 3 4 5 6 7}"
STAMP=20261025
DATA=/opt/dlami/nvme/meanflow/datasets
CKPT_cub200=$MAIN/files/logs/finetuning/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/best_fid/checkpoint_35000
CKPT_stanfordcars=$MAIN/files/logs/finetuning/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh/best_fid/checkpoint_12500
DS_ARGS_cub200=""
DS_ARGS_stanfordcars="--config.dataset.root=$DATA/stanford-cars_processed_latents --config.dataset.num_classes=196 --config.model.num_classes=196 --config.sampling.num_classes=196 --config.fid.cache_ref=$MAIN/files/fid_stats/stanford_cars_processed-fid_stats.npz --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz"
EVAL_ARGS_cub200="--config.dataset.root=$DATA/cub-200-2011_processed_latents --config.dataset.num_classes=200 --config.model.num_classes=200 --config.sampling.num_classes=200 --config.fid.cache_ref=$MAIN/files/fid_stats/cub-200-2011_processed-fid_stats.npz --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz"
EVAL_ARGS_stanfordcars="$DS_ARGS_stanfordcars"
arm_args(){ case "$1" in
  no_anchor|no_norm|endpoint) echo "--config.caimf.score_mode=$1" ;;
  cap0.25)  echo "--config.caimf.interval_cap=0.25" ;;
  cap0.001) echo "--config.caimf.interval_cap=0.001" ;;
  *) return 1 ;; esac; }
# job = dataset:arm, in launch order (CUB score arms first)
JOBS=(cub200:no_anchor cub200:no_norm cub200:endpoint stanfordcars:no_anchor stanfordcars:no_norm stanfordcars:endpoint
      cub200:cap0.25 cub200:cap0.001 stanfordcars:cap0.25 stanfordcars:cap0.001)
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wd_of(){ echo "$ADV/files/logs/finetuning/${1%%:*}_iMF_CAIMF_abl_${1##*:}_${STAMP}"; }
sess_of(){ echo "camfabl_${1%%:*}_${1##*:}" | tr . p; }
has_best(){ ls -d "$(wd_of "$1")/best_fid/checkpoint_"* >/dev/null 2>&1; }
eval_done(){ local f; f="$(wd_of "$1")/final_eval_metrics.csv"; [[ -f "$f" ]] && (( $(grep -c . "$f") >= 5 )); }
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }
launch_train(){  # gpu job
  local gpu=$1 job=$2 ds=${2%%:*} arm=${2##*:} wd s args ckpt ds_args v
  wd=$(wd_of "$job"); s=$(sess_of "$job"); args=$(arm_args "$arm") || { log "BAD arm $arm"; return 1; }
  v="CKPT_$ds"; ckpt=${!v}; v="DS_ARGS_$ds"; ds_args=${!v}
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
log "=== CAMF ablation watcher: ${#JOBS[@]} jobs, GPUs [$ALLOWED], patience $PATIENCE ==="
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
log "=== all CAMF ablation jobs done ==="
