#!/usr/bin/env bash
# One worker per GPU for the iMF NFE sweep: waits until its GPU is free (no other screen holding it, <3000 MiB),
# then takes "<kind> <dataset>" jobs from files/sweep_queue.txt (flock). A job whose run dir is still being written
# by another eval (e.g. a 250-NFE eval) is put back at the end of the queue.
R=/opt/dlami/nvme/meanflow/imeanflow_adversarial; Q=$R/files/sweep_queue.txt; GPU=$1; NFES=${2:-"8 16 32 64"}
LOG=$R/files/sweep_logs; mkdir -p $LOG
until (( $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits) < 3000 )) && ! screen -ls | grep -qE "\.(w250_gpu$GPU|d_gpu$GPU)\b"; do sleep 120; done
echo "$(date -u +%FT%TZ) gpu$GPU worker start" >> $LOG/events.log
while true; do
  job=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [[ -z "$job" ]] && break
  set -- $job; run=$(bash $R/scripts/eval_sweep_imf.sh $1 $2 $GPU print)
  if pgrep -f "$run" >/dev/null; then flock $Q.lock bash -c "echo '$job' >> $Q"; echo "$(date -u +%FT%TZ) gpu$GPU defer $job (run dir busy)" >> $LOG/events.log; sleep 300; continue; fi
  echo "$(date -u +%FT%TZ) gpu$GPU start $job" >> $LOG/events.log
  bash $R/scripts/eval_sweep_imf.sh $1 $2 $GPU "$NFES" 32 > $LOG/$1_$2.log 2>&1
  echo "$(date -u +%FT%TZ) gpu$GPU end $job rc=$?" >> $LOG/events.log
done
echo "$(date -u +%FT%TZ) gpu$GPU worker done" >> $LOG/events.log
