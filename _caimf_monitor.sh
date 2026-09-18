#!/usr/bin/env bash
# Detached progress monitor for the 5 CAIMF runs. Polls every 10 min, appends a
# snapshot to files/logs/caimf_monitor.log. Flags any OOM/crash. Exits when all
# 5 have a best_fid checkpoint AND their training screen is gone.
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
cd "$ADV"
MON="$ADV/files/logs/caimf_monitor.log"
DSES=(caltech101 artbench10 cub200 food101 stanfordcars)
STAMP=20260725
snap(){
  echo "==================== $(date "+%F %T") ===================="
  for ds in "${DSES[@]}"; do
    log=files/logs/caimf_${ds}.log
    wd=files/logs/finetuning/${ds}_SiT_MeFT_CAIMF_puresadv_${STAMP}
    alive=no; screen -ls | grep -qE "caimf_gpu[0-9]+_${ds}\b" && alive=yes
    step=$(grep -oE "batch_step=[0-9]+" "$log" 2>/dev/null | tail -1)
    fidline=$(grep -oE "FID_single_head_ema_steps_4=[0-9.]+.*FD_DINO_single_head_ema_steps_4=[0-9.]+" "$log" 2>/dev/null | tail -1)
    bf=$(ls -d "$wd"/best_fid/checkpoint_* 2>/dev/null | tail -1)
    # only TODAY-dated OOM (log line date is embedded; check screen liveness instead)
    printf "  [%s] alive=%s %s | %s | best_fid=%s\n" "$ds" "$alive" "${step:-step?}" "${fidline:-<no eval yet>}" "${bf##*/}"
  done
}
alldone(){
  for ds in "${DSES[@]}"; do
    wd=files/logs/finetuning/${ds}_SiT_MeFT_CAIMF_puresadv_${STAMP}
    ls -d "$wd"/best_fid/checkpoint_* >/dev/null 2>&1 || return 1
    screen -ls | grep -qE "caimf_gpu[0-9]+_${ds}\b" && return 1
  done
  return 0
}
echo "=== CAIMF monitor started $(date "+%F %T") ===" >> "$MON"
while true; do
  snap >> "$MON"
  if alldone; then echo "=== ALL 5 CAIMF RUNS DONE $(date "+%F %T") ===" >> "$MON"; break; fi
  sleep 600
done
