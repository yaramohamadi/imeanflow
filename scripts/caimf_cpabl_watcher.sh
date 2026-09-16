#!/usr/bin/env bash
# =============================================================================
# CA-iMF CP-ablation watcher -- detached (screen caimf_cpabl_watcher).
#
# Runs the 6 CP-regularization variants on CUB-200 from the iMF/XL-2 MeFT
# checkpoint. "Ours" (full) is intentionally excluded (reused from main runs).
#
# SEQUENCING (per user): case-by-case. This watcher claims each GPU as it
# frees up (stable-free gate), WITHOUT waiting for all in-flight redo/CAIMF
# runs to finish. It co-exists with the other watchers; since the redo queue
# has no jobs waiting (all remaining are in-flight), there is no contention.
#
# Per run: pure-adversarial iMF CA-iMF, cp_mode overridden on CLI, eval every
# 5000 steps, 10000 samples, save_best_fid_only. Integrated early-stop with
# PATIENCE=5 consecutive 4-step-FID rises (new standard). Best_fid retained.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/caimf_cpabl_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=3          # free-poll gate before claiming a GPU
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
PATIENCE=5               # consecutive 4-step-FID rises -> early stop
STAMP=20260726
CKPT=$MAIN/files/logs/finetuning/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/best_fid/checkpoint_35000

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MODES=(afm_sum2 none zt zt_zr zt_zhatr full_sum2)
workdir_for(){ echo "$ADV/files/logs/finetuning/cub200_iMF_CAIMF_cpabl_${1}_${STAMP}"; }

# a variant is "done" when its workdir has a best_fid ckpt AND no training screen
train_done(){  # $1=mode
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "cpabl_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=mode
  local gpu="$1" mode="$2"
  [[ -d "$CKPT" ]] || { log "ERROR ckpt missing: $CKPT"; return 1; }
  local wd; wd=$(workdir_for "$mode")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$mode] already has best_fid"; return 1; fi
  local sess="cpabl_gpu${gpu}_${mode}"
  log "LAUNCH cp-ablation $mode on GPU $gpu (screen $sess) wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export REQUIRE_FD_DINO=1 MPLCONFIGDIR=/tmp/mpl-cpabl-$mode
    $PY main_caimf.py \
      --workdir='$wd' \
      --config=$ADV/configs/load_config.py:cub_imf_caimf_cpablation \
      --config.load_from='$CKPT' \
      --config.caimf.cp_mode=$mode \
      2>&1 | tee -a $ADV/files/logs/caimf_cpabl_${mode}.log
  "
}

# 4-step FID series -> consecutive rises from running min
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

# ---- case-by-case: claim GPUs as they free up (no global wait) ----
log "=== CP-ablation watcher: case-by-case GPU claiming, ${#MODES[@]} variants, gate $STABLE_NEEDED, patience $PATIENCE ==="

declare -A STABLE=() CLAIMED=() DONE=()
for m in "${MODES[@]}"; do if train_done "$m"; then DONE[$m]=1; log "SKIP $m (best_fid present)"; fi; done
remaining(){ local n=0; for m in "${MODES[@]}"; do [[ -z "${DONE[$m]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  # release finished GPUs + integrated early-stop
  for idx in "${!CLAIMED[@]}"; do
    mode="${CLAIMED[$idx]}"
    # early-stop check while still training
    if screen -ls | grep -qE "cpabl_gpu${idx}_${mode}\b"; then
      csv=$(workdir_for "$mode")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $mode: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing cpabl_gpu${idx}_${mode}."
        screen -S "cpabl_gpu${idx}_${mode}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    # screen gone (finished or just killed)
    if ! screen -ls | grep -qE "cpabl_gpu${idx}_${mode}\b"; then
      if train_done "$mode"; then DONE[$mode]=1; log "COMPLETE $mode (GPU $idx freed)"; else log "WARN $mode screen gone, no best_fid -- check caimf_cpabl_${mode}.log"; DONE[$mode]=1; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # assign free GPUs to remaining modes
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for mode in "${MODES[@]}"; do
      [[ -n "${DONE[$mode]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$mode" ]] && inflight=1; done
      (( inflight )) && continue
      if launch "$idx" "$mode"; then CLAIMED[$idx]="$mode"; log "PROGRESS $mode -> GPU $idx; remaining $(remaining)"; sleep 20; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all 6 CP-ablation variants complete; watcher exiting ==="
