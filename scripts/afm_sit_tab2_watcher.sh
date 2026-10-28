#!/usr/bin/env bash
# Tab. 2 row "MF-T + AFM" for SiT-XL/2, all 5 datasets. AFM post-training (configs/sit_meft_afm_tab2 = caltech_sit_meft_afm_posttrain
# with FID every 5000 instead of 2500) from the SAME best MF-T checkpoints as Tab. 2's SiT MF-T + CAMF row
# (caimf_sit_watcher.sh). Best-FID selection + final eval at NFE 1 2 4 (Tab. 2 protocol). Early stop: patience 5 on 4-step FID.
# GPUs 4-7 only (0-3 hold the CAMF final-checkpoint runs).
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python; cd $ADV
F=/opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning
LOG=$ADV/files/logs/afm_sit_tab2_watcher.log; POOL="4 5 6 7"; PATIENCE=5
QUEUE=(cub200:$F/cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8/best_fid/checkpoint_20000
       stanfordcars:$F/stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo/best_fid/checkpoint_30000
       artbench10:$F/artbench10_SiT_DMF_plain_meanflow_meft_online_20260722_172435_pe502i/best_fid/checkpoint_20000
       caltech101:$F/caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w/best_fid/checkpoint_15000
       food101:$F/food101_SiT_DMF_plain_meanflow_meft_online_20260722_232709_tv7xmz/best_fid/checkpoint_20000)
log(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
wd(){ echo $ADV/files/logs/finetuning/${1}_SiT_MeFT_AFM_tab2_20261028; }
sess(){ echo afmsit_$1; }
streak(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null | awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }
declare -A GPU_OF=() STOPPED=()
next=0
log "=== start: ${#QUEUE[@]} runs, pool {$POOL}, patience $PATIENCE ==="
while :; do
  for ds in "${!GPU_OF[@]}"; do
    if screen -ls | grep -q "$(sess $ds)\b"; then
      s=$(streak "$(wd $ds)/eval_metrics.csv")
      if (( s >= PATIENCE )) && [[ -z "${STOPPED[$ds]:-}" ]]; then
        log "EARLY-STOP $ds (4-step FID rose $s evals in a row from its min): SIGINT the trainer; best_fid kept, final eval follows"
        pkill -INT -f "main_afm_sit_meft.py.*$(wd $ds)"; STOPPED[$ds]=1
      fi
    else
      log "DONE $ds (GPU ${GPU_OF[$ds]} free)"; unset "GPU_OF[$ds]"
    fi
  done
  (( next >= ${#QUEUE[@]} )) && (( ${#GPU_OF[@]} == 0 )) && { log "=== all done ==="; exit 0; }
  for g in $POOL; do
    (( next < ${#QUEUE[@]} )) || break
    busy=0; for d in "${!GPU_OF[@]}"; do [[ ${GPU_OF[$d]} == "$g" ]] && busy=1; done; (( busy )) && continue
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); (( m < 3000 )) || continue
    IFS=: read -r ds ck <<< "${QUEUE[$next]}"; next=$((next+1)); GPU_OF[$ds]=$g
    log "LAUNCH $ds on GPU $g from $ck -> $(wd $ds)"
    screen -dmS $(sess $ds) bash -c "cd $ADV; export CUDA_VISIBLE_DEVICES=$g PYTHON=$PY REPO=$ADV USE_WANDB=False RUN_FINAL_EVAL=True FINAL_EVAL_STEPS='1 2 4' CONFIG_MODE=sit_meft_afm_tab2 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/mpl-afmsit-$ds; bash scripts/run_sit_meft_adversarial.sh afm $ds $ck $(wd $ds) 2>&1 | tee -a $ADV/files/logs/afmsit_$ds.log"
    sleep 20
  done
  sleep 90
done
