#!/usr/bin/env bash
# =============================================================================
# FT + AFM REDO (2026-07-27): corrected AFM schedule. Endpoint-AFM
# (ablation=afm_only) on the NATIVE, TARGET-FINETUNED iMF checkpoints
# (plain_iMF_finetune_<ds>) -- NOT the ImageNet iMF. Detached: screen
# afm_redo_watcher.
#
# WHY REDO: the previous FT+AFM sweep used the CAMF-aligned schedule (4:1 D:G,
# 5000 D-warmup, 150k batches=30k G, FID@5000). The AFM-native schedule is:
#   * 1:1 D:G                    (d_steps_per_g_step=1)
#   * NO discriminator warmup    (discriminator_warmup_steps=0)
#   * 40k GENERATOR updates      (max_posttrain_batches=80000 = 40k D + 40k G)
#   * FID eval every 2500 steps  (fid_schedule from_step=2500 every_steps=2500)
# All four are now set in configs/caltech_imf_afm_posttrain_config.yml.
#
#   ablation=afm_only lambda_adv=1.0 lambda_imf=0.0 lambda_cp=0.01
#   gen/dis LR=1e-5 ; metric_num_steps=[4] ; 10000 samples ; save_best_fid_only
#   native iMF op-point (omega=7.5, t in [0.4,0.65]). Final eval NFE 1&2.
#
# NO EARLY-STOP: run the full 40k generator updates (user request 2026-07-27).
# One GPU/run. Shares the atomic GPU lock with the imnet-CAMF watcher.
# NOTE: ImageNet-AFM is intentionally NOT run here (user deferred it); this
# watcher only redoes the 5 FT+AFM (finetuned-iMF) runs.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/afm_redo_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=60
STABLE_NEEDED=3          # GPUs 1/4/5 are already free; claim promptly
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_imf_afm_posttrain
STAMP=20260727_redo
PATIENCE=999999          # early-stop DISABLED -- run full 40k generator updates
RUNNER="$ADV/scripts/run_imf_afm.sh"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

source "$ADV/scripts/gpu_lock.inc.sh"

# ds : best_fid checkpoint -- plain-iMF DOMAIN-FINETUNED imfDiT_XL_2 students
QUEUE=(
  "artbench10:$IMF/files/logs/finetuning/plain_iMF_finetune_artbench10_h100_20260722_225848_jt6v93/best_fid/checkpoint_7500"
  "caltech101:$IMF/files/logs/finetuning/plain_iMF_finetune_caltech101_h100_20260722_222718_cwbunp/best_fid/checkpoint_32500"
  "cub200:$IMF/files/logs/finetuning/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/best_fid/checkpoint_35000"
  "food101:$IMF/files/logs/finetuning/plain_iMF_finetune_food101_h100_20260722_231935_s9moo7/best_fid/checkpoint_17500"
  "stanfordcars:$IMF/files/logs/finetuning/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh/best_fid/checkpoint_12500"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_AFM_puresadv_${STAMP}"; }

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "afmredo_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="afmredo_gpu${gpu}_${ds}"
  log "LAUNCH AFM-redo $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export CONFIG_MODE=$CONFIG_MODE
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-afmredo-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/afmredo_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== AFM REDO watcher: ${#QUEUE[@]} finetuned-iMF datasets, pool {$ALLOWED}, CORRECTED schedule (1:1 D:G, no warmup, 40k G=80k batches, FID@2500), 10k-samples, best-fid-only, final NFE{$FINAL_EVAL_STEPS}, NO early-stop, gate $STABLE_NEEDED, GPU-locked ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present in $STAMP workdir)"; fi; done

# adopt any of OUR redo runs already in flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ afmredo_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "afmredo_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx (screen afmredo_gpu${gidx}_${gds})"
done < <(screen -ls 2>/dev/null | grep -oE "afmredo_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

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
    if screen -ls | grep -qE "afmredo_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $ds (disabled path): streak=$streak. Killing afmredo_gpu${idx}_${ds}."
        screen -S "afmredo_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    if ! screen -ls | grep -qE "afmredo_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check afmredo_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "afmredo_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all AFM REDO runs complete; watcher exiting ==="
