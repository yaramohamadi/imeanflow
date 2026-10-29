#!/usr/bin/env bash
# One worker per GPU: run the assigned first job (if any), then take jobs from the shared queue (flock).
R=/opt/dlami/nvme/meanflow/imeanflow_adversarial; Q=$R/files/eval250_queue.txt; GPU=$1; shift
LOG=$R/files/eval250_logs; mkdir -p $LOG
run() { echo "$(date -u +%FT%TZ) gpu$GPU start $1 $2" >> $LOG/events.log
        bash $R/scripts/eval250_tab2_imf.sh $1 $2 $GPU 32 > $LOG/$1_$2.log 2>&1
        echo "$(date -u +%FT%TZ) gpu$GPU end $1 $2 rc=$?" >> $LOG/events.log; }
[[ $# -eq 2 ]] && run "$1" "$2"
while true; do
  job=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q")
  [[ -z "$job" ]] && break
  run $job
done
echo "$(date -u +%FT%TZ) gpu$GPU worker done" >> $LOG/events.log
