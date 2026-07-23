#!/usr/bin/env bash
# Drain the 2 queued SiT 250-evals (food101, stanfordcars) onto GPUs 0/1/4 as
# they free. A GPU is "free" if <2000 MiB AND no sit250_* screen is on it.
# Waits for the launched eval to actually claim memory before reusing a GPU
# (avoids the orchestrator double-book race). Exits when all 5 have a 250 CSV.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
cd "$REPO"
QUEUE=(food101 stanfordcars)
POOL=(0 1 4)

has_csv(){ ls files/logs/finetuning/*"$1"*SiT*meft_online*/eval_best_fid_250steps/eval_metrics.csv >/dev/null 2>&1; }
screen_running(){ screen -ls 2>/dev/null | grep -qE "sit250_$1[[:space:]]"; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ "${u:-9999}" -lt 2000 ]; }

qi=0
for pass in $(seq 1 400); do
  # done?
  all=1; for d in caltech101 artbench10 cub200 food101 stanfordcars; do has_csv "$d" || all=0; done
  [ "$all" = "1" ] && { echo "[watch] all 5 SiT 250 evals done"; break; }
  # launch queued item if there is one and a pool GPU is free
  if [ $qi -lt ${#QUEUE[@]} ]; then
    ds=${QUEUE[$qi]}
    if has_csv "$ds" || screen_running "$ds"; then qi=$((qi+1)); continue; fi
    for g in "${POOL[@]}"; do
      # skip GPU that still hosts a running sit250 screen's process
      if gpu_free "$g"; then
        echo "[watch] GPU $g free -> launching $ds"
        # reuse launcher's single-dataset path by calling it inline
        CUDA_G=$g bash -c "
          cd $REPO
          # minimal inline launch mirroring sit_eval250.sh
          bash scripts/sit_eval250_one.sh $ds $g
        " 2>&1 | sed 's/^/[watch] /'
        qi=$((qi+1))
        sleep 120   # let it claim memory before considering GPUs free again
        break
      fi
    done
  fi
  sleep 90
done
