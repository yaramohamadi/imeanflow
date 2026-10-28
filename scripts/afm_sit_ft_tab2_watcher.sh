#!/usr/bin/env bash
# Tab. 2 rows "FT + AFM" (4x2) for SiT-XL/2, all 5 datasets: fixed-grid (K=4) AFM from the SiT FT best checkpoints
# (scripts/run_sit_ft_afm.sh, configs/sit_ft_afm_tab2; no guidance in training, CFG 1.5 at sampling, Euler).
# Waits until afm_sit_tab2_watcher has launched its last job (food101) so the two never race for a GPU, then takes
# any GPU that is stably free. Early stop: patience 5 on 4-step FID. CPU threads capped per job (CPU-side FID evals
# of 8 concurrent jobs saturated 192 cores on 2026-10-28).
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python; cd $ADV
F=/opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning
LOG=$ADV/files/logs/afm_sit_ft_tab2_watcher.log; POOL="0 1 2 3 4 5 6 7"; PATIENCE=5; STABLE_NEEDED=3; THREADS=24
ck(){ ls -d $F/plain_SiT_finetune_${1}_rebuttal_20261024_155255_*/best_fid/checkpoint_* | grep -E "_155255_[a-z0-9]{6}/best_fid/" | sort -V | tail -1; }
QUEUE=(cub200 stanfordcars artbench10 caltech101 food101)
log(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
wd(){ echo $ADV/files/logs/finetuning/${1}_SiT_FT_AFM_tab2_20261028; }
sess(){ echo afmsitft_$1; }
streak(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null | awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }
declare -A GPU_OF=() STOPPED=() STABLE=()
next=0
log "=== start: ${#QUEUE[@]} runs, pool {$POOL}, patience $PATIENCE, threads $THREADS; waiting for afm_sit_tab2 to launch food101 ==="
until grep -q "LAUNCH food101" $ADV/files/logs/afm_sit_tab2_watcher.log 2>/dev/null; do sleep 120; done
log "afm_sit_tab2 has launched food101; claiming free GPUs"
while :; do
  for ds in "${!GPU_OF[@]}"; do
    if screen -ls | grep -q "$(sess $ds)\b"; then
      s=$(streak "$(wd $ds)/eval_metrics.csv")
      if (( s >= PATIENCE )) && [[ -z "${STOPPED[$ds]:-}" ]]; then
        log "EARLY-STOP $ds (4-step FID rose $s evals in a row from its min): SIGINT trainer; best_fid kept, final eval follows"
        pkill -INT -f "main_afm_sit_ft.py.*$(wd $ds)" ; STOPPED[$ds]=1
      fi
    else
      log "DONE $ds (GPU ${GPU_OF[$ds]} free)"; unset "GPU_OF[$ds]"
    fi
  done
  (( next >= ${#QUEUE[@]} )) && (( ${#GPU_OF[@]} == 0 )) && { log "=== all done ==="; exit 0; }
  for g in $POOL; do
    busy=0; for d in "${!GPU_OF[@]}"; do [[ ${GPU_OF[$d]} == "$g" ]] && busy=1; done
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( busy == 0 && m < 3000 )); then STABLE[$g]=$(( ${STABLE[$g]:-0} + 1 )); else STABLE[$g]=0; fi
  done
  for g in $POOL; do
    (( next < ${#QUEUE[@]} )) || break
    (( ${STABLE[$g]:-0} >= STABLE_NEEDED )) || continue
    ds=${QUEUE[$next]}; c=$(ck $ds); next=$((next+1)); GPU_OF[$ds]=$g; STABLE[$g]=0
    log "LAUNCH $ds on GPU $g from $c -> $(wd $ds)"
    screen -dmS $(sess $ds) bash -c "cd $ADV; export CUDA_VISIBLE_DEVICES=$g PYTHON=$PY REPO=$ADV OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/mpl-afmsitft-$ds; bash scripts/run_sit_ft_afm.sh $ds $c $(wd $ds) 2>&1 | tee -a $ADV/files/logs/afmsitft_$ds.log"
    sleep 20
  done
  sleep 90
done
