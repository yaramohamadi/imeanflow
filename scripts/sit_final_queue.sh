#!/usr/bin/env bash
# Queue for sit_final_ckpt.sh: one job per free GPU in POOL (free = <3 GB used for 3 polls), never
# grabs a busy GPU. MF-T resumes first (longest), then FT evals. Log: files/logs/sit_final_queue.log
set -uo pipefail
M=/opt/dlami/nvme/meanflow/imeanflow; cd $M; LOG=$M/files/logs/sit_final_queue.log
POOL="${POOL:-1 2 3 5 7}"
JOBS=(
 "mft artbench10_SiT_DMF_plain_meanflow_meft_online_20260722_172435_pe502i artbench10 EXP-364"
 "mft caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w caltech101 EXP-365"
 "mft cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8 cub200 EXP-366"
 "mft food101_SiT_DMF_plain_meanflow_meft_online_20260722_232709_tv7xmz food101 EXP-367"
 "ft plain_SiT_finetune_artbench10_rebuttal_20261024_155255_caq1ih artbench10 EXP-368"
 "ft plain_SiT_finetune_caltech101_rebuttal_20261024_155255_b7w6x4 caltech101 EXP-369"
 "ft plain_SiT_finetune_cub200_rebuttal_20261024_155255_00x0lt cub200 EXP-370"
 "ft plain_SiT_finetune_stanfordcars_rebuttal_20261024_155255_07uczo stanfordcars EXP-371"
)
log(){ echo "[$(date '+%F %T %Z')] $*" | tee -a $LOG; }
declare -A HELD; declare -A FREE
log "=== sit_final_queue: ${#JOBS[@]} jobs, pool {$POOL} ==="
i=0
while (( i < ${#JOBS[@]} )); do
  for g in $POOL; do
    (( i < ${#JOBS[@]} )) || break
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
    if [ -n "${HELD[$g]:-}" ] && screen -ls | grep -q "\.${HELD[$g]}\b"; then FREE[$g]=0; continue; fi
    HELD[$g]=""
    if (( used < 3000 )); then FREE[$g]=$(( ${FREE[$g]:-0} + 1 )); else FREE[$g]=0; fi
    if (( ${FREE[$g]} >= 3 )); then
      read -r mode run ds exp <<< "${JOBS[$i]}"
      sess="sitfinal_${mode}_${ds}"
      screen -dmS $sess bash -c "bash scripts/sit_final_ckpt.sh $mode $run $ds $g > $M/files/logs/${sess}.log 2>&1"
      log "LAUNCH $exp $mode $ds on GPU$g (screen $sess)"
      HELD[$g]=$sess; FREE[$g]=0; i=$((i+1))
    fi
  done
  sleep 60
done
log "=== all ${#JOBS[@]} jobs launched ==="
