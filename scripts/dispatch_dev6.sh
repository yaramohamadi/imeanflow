#!/usr/bin/env bash
# dev6 GPU dispatcher (screen "disp"). Takes jobs from files/dev6_queue.txt (one per line, top first) and runs each
# on a free GPU in its own screen d_gpu<g>. A GPU is free when no t250 / w250_gpu<g> / d_gpu<g> screen holds it and
# it has used <3000 MiB for 3 polls in a row. Jobs:
#   train <ds>          iMF AFM standalone from ImageNet iMF (same recipe as imnet_afm_imf_watcher.sh: run_imf_afm.sh,
#                       caltech_imf_afm_posttrain_imnet0727 = July values: 4D:1G, D-warmup 5000, 150k batches; no early stop); on success queues "eval imnetafm <ds>"
#   eval <kind> <ds>    eval250_tab2.sh <kind> <ds> <gpu> 250 32
R=/opt/dlami/nvme/meanflow/imeanflow_adversarial; IMF=/opt/dlami/nvme/meanflow/imeanflow
Q=$R/files/dev6_queue.txt; LOG=$R/files/dev6_dispatch.log; JL=$R/files/dev6_jobs; mkdir -p $JL
declare -A FREE=()
log(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
log "dispatcher start"
while true; do
  for g in 0 1 2 3 4 5 6 7; do
    held=0; screen -ls | grep -qE "\.(w250_gpu$g|d_gpu$g)\b" && held=1
    [[ $g == 0 ]] && screen -ls | grep -q "\.t250\b" && held=1
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( held == 0 && m < 3000 )); then FREE[$g]=$(( ${FREE[$g]:-0} + 1 )); else FREE[$g]=0; fi
    (( ${FREE[$g]} >= 3 )) || continue
    job=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [[ -n "$job" ]] || continue
    FREE[$g]=0; set -- $job
    if [[ $1 == train ]]; then
      ds=$2; wd=$R/files/logs/finetuning/${ds}_iMF_imnet_AFM_puresadv_20261029
      cmd="cd $R && CUDA_VISIBLE_DEVICES=$g CONFIG_MODE=caltech_imf_afm_posttrain_imnet0727 PYTHON=$IMF/.venv/bin/python RUN_FINAL_EVAL=False bash scripts/run_imf_afm.sh $ds $IMF/files/weights/iMF-XL-2-full $wd > $JL/train_$ds.log 2>&1 && (flock $Q.lock bash -c 'echo eval imnetafm $ds >> $Q')"
    else
      cmd="bash $R/scripts/eval250_tab2.sh $2 $3 $g 250 32 > $JL/eval_$2_$3.log 2>&1"
    fi
    log "gpu$g START $job :: $cmd"
    screen -dmS d_gpu$g bash -c "$cmd; echo \"\$(date -u +%FT%TZ) gpu$g END $job rc=\$?\" >> $LOG"
  done
  [[ -s $Q ]] || screen -ls | grep -q "\.d_gpu" || { log "queue empty and no jobs running: exit"; exit 0; }
  sleep 90
done
