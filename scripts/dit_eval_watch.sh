#!/usr/bin/env bash
# Drain all 5 DiT evals. Assigns each remaining dataset to ANY GPU that is free
# (<2000 MiB) and has no diteval screen. Waits 150s after a launch so the new
# eval claims memory before another GPU is considered free (avoids double-book).
# A dataset is "done" when it has 1/2/250 eval CSVs. Exits when all 5 done.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
cd "$REPO"
DSETS=(caltech101 artbench10 cub200 food101 stanfordcars)

done_ds(){ local rd; rd=$(ls -d files/logs/finetuning/*$1*DiT_DMF*meft_online*1717*/ 2>/dev/null | head -1); rd=${rd%/}
  [ -n "$rd" ] && [ -f "$rd/eval_best_fid_250steps/eval_metrics.csv" ] && [ -f "$rd/eval_best_fid_1steps/eval_metrics.csv" ]; }
running_ds(){ screen -ls 2>/dev/null | grep -qE "diteval_$1[[:space:]]"; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ "${u:-9999}" -lt 2000 ]; }

for pass in $(seq 1 1000); do
  all=1; for d in "${DSETS[@]}"; do done_ds "$d" || all=0; done
  [ "$all" = "1" ] && { echo "[dit-watch] all 5 DiT evals done"; break; }
  for d in "${DSETS[@]}"; do
    done_ds "$d" && continue
    running_ds "$d" && continue
    # find a free GPU
    for g in 0 1 2 3 4 5 6 7; do
      if gpu_free "$g"; then
        echo "[dit-watch] GPU $g free -> launching DiT $d"
        bash scripts/dit_eval_one.sh "$d" "$g" 2>&1 | sed 's/^/[dit-watch] /'
        sleep 150   # let it claim mem before reusing GPUs
        break 2     # restart the scan
      fi
    done
  done
  sleep 60
done
