#!/usr/bin/env bash
# =============================================================================
# Autonomous launch orchestrator (run detached in screen on dev5).
# Drains QUEUE onto GPUs as they free up. Each item = "type:dataset".
#   imf -> scripts/launch_imf_one.sh <dataset> <gpu>   (40k, use_ema=False)
#   sit -> scripts/gen_launch_sit_meft.sh <dataset> <gpu> (30k baseline)
# A GPU is "available" when <2000 MiB used AND not already assigned by us this run.
# Waits between launches so the new job claims memory before the next check.
# Logs to files/logs/orchestrator.log. Exits when queue empty.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
cd "$REPO"
LOG="$REPO/files/logs/orchestrator.log"

# order matters: finish the iMF sweep first, then the 3 remaining SiT
QUEUE=(
  "imf:food101"
  "imf:stanfordcars"
  "sit:cub200"
  "sit:food101"
  "sit:stanfordcars"
)

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

free_gpus(){
  # print indices with <2000 MiB used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2); if($2+0<2000) print $1}'
}

launch_one(){
  local type=$1 ds=$2 gpu=$3
  case "$type" in
    imf) bash scripts/launch_imf_one.sh "$ds" "$gpu" >>"$LOG" 2>&1 ;;
    sit) bash scripts/gen_launch_sit_meft.sh "$ds" "$gpu" >>"$LOG" 2>&1 ;;
    *) log "ERROR unknown type $type"; return 1 ;;
  esac
}

log "orchestrator start; queue=${QUEUE[*]}"
idx=0
while [[ $idx -lt ${#QUEUE[@]} ]]; do
  item="${QUEUE[$idx]}"
  type="${item%%:*}"; ds="${item##*:}"
  # find a free gpu
  gpu=""
  for g in $(free_gpus); do gpu=$g; break; done
  if [[ -z "$gpu" ]]; then
    sleep 60; continue
  fi
  log "launching $type $ds on GPU $gpu (item $((idx+1))/${#QUEUE[@]})"
  if launch_one "$type" "$ds" "$gpu"; then
    log "launched $type $ds on GPU $gpu; waiting 150s for it to claim GPU mem"
    idx=$((idx+1))
    sleep 150
  else
    log "launch FAILED for $type $ds on GPU $gpu (guard/collision?); retrying in 60s"
    sleep 60
  fi
done
log "orchestrator done; all ${#QUEUE[@]} items launched"
