#!/usr/bin/env bash
# Lightweight watchdog for the 5 DiT MeFT runs. Does NOT launch anything (all 5
# already running). Just: tracks completion (best_fid + final_eval), detects a
# dead screen without completion (crash), writes a status file, exits when all 5
# are done. Re-launches a crashed run at most twice on its original GPU.
set +u
REPO=/opt/dlami/nvme/meanflow/imeanflow
STATUS=$REPO/files/logs/ditmeft_WATCH_STATUS.txt
WLOG=$REPO/files/logs/ditmeft_WATCH.log
cd "$REPO"
ALL=(caltech101 artbench10 cub200 food101 stanfordcars)
declare -A GPU=( [caltech101]=0 [artbench10]=1 [cub200]=6 [food101]=7 [stanfordcars]=2 )
declare -A RETRY=(); for d in "${ALL[@]}"; do RETRY[$d]=0; done
log(){ echo "$(date -u +%FT%TZ 2>/dev/null||cat /proc/uptime) $*" >> "$WLOG"; }

done_ds(){ ls -d "$REPO"/files/logs/finetuning/*"$1"*DiT_DMF*/final_eval_metrics.csv 2>/dev/null | grep -q . \
        || ls -d "$REPO"/files/logs/finetuning/*DiT_DMF*"$1"*/final_eval_metrics.csv 2>/dev/null | grep -q .; }
bestfid_ds(){ ls -d "$REPO"/files/logs/finetuning/*"$1"*DiT_DMF*/best_fid/checkpoint_* 2>/dev/null | grep -q . \
           || ls -d "$REPO"/files/logs/finetuning/*DiT_DMF*"$1"*/best_fid/checkpoint_* 2>/dev/null | grep -q .; }
alive(){ screen -ls 2>/dev/null | grep -q "ditmeft_$1\b"; }
laststep(){ grep -oE "\[[0-9]+\]" "$REPO/files/logs/ditmeft_$1_run.log" 2>/dev/null | tail -1 | tr -d '[]'; }

log "watchdog start"
while true; do
  { echo "DiT MeFT watchdog  $(date -u +%FT%TZ 2>/dev/null)"
    for d in "${ALL[@]}"; do
      if done_ds "$d"; then s="DONE (final_eval)"; elif bestfid_ds "$d" && ! alive "$d"; then s="best_fid saved, finalizing";
      elif alive "$d"; then s="RUNNING step $(laststep "$d")"; else s="STOPPED (retry ${RETRY[$d]})"; fi
      printf "  %-14s %s\n" "$d" "$s"
    done
  } > "$STATUS"

  # all done?
  a=1; for d in "${ALL[@]}"; do done_ds "$d" || a=0; done
  [[ $a -eq 1 ]] && { log "ALL 5 DONE"; echo "ALL 5 DONE $(date -u +%FT%TZ 2>/dev/null)" >> "$STATUS"; break; }

  # crash detection: screen dead, no best_fid, has a run log => relaunch (max 2)
  for d in "${ALL[@]}"; do
    if ! alive "$d" && ! bestfid_ds "$d" && [[ -f "$REPO/files/logs/ditmeft_${d}_run.log" ]]; then
      if [[ ${RETRY[$d]} -lt 2 ]]; then
        RETRY[$d]=$(( ${RETRY[$d]} + 1 ))
        log "RELAUNCH $d (crash, retry ${RETRY[$d]}) on GPU ${GPU[$d]}"
        bash "$REPO/scripts/gen_and_launch.sh" "$d" "${GPU[$d]}" >> "$WLOG" 2>&1
        sleep 10
      fi
    fi
  done
  sleep 180
done
log "watchdog exit"
