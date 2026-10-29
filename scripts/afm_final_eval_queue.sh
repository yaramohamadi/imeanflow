#!/usr/bin/env bash
# Final evals the AFM launchers skipped: the watchers' early stop (SIGINT) makes training exit non-zero, and both
# launchers then skip their final eval. For every AFM run whose screen is gone, that has a best_fid checkpoint and no
# final eval yet, run it on a stably free GPU: SiT MF-T + AFM at NFE 1 2 4 (Tab. 2 protocol); SiT FT + AFM at 4 Euler
# steps with CFG 1.5 and without (final_evaluate). Starts claiming GPUs only after afm_sit_ft_tab2_watcher has
# launched its last job, so the two never race. Idempotent: re-running skips runs that already have outputs.
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python; cd $ADV
F=/opt/dlami/nvme/meanflow/imeanflow/files/logs/finetuning; W=$ADV/files/logs/finetuning
LOG=$ADV/files/logs/afm_final_eval_queue.log; POOL="0 1 2 3 4 5 6 7"; STABLE_NEEDED=3; THREADS=24
DS=(cub200 stanfordcars artbench10 caltech101 food101)
log(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
args(){ case $1 in
  caltech101)   L=caltech-101_processed_latents; N=101; FI=caltech-101-fid_stats.npz; FD=caltech-101-fd_dino-vitb14_stats.npz ;;
  artbench10)   L=artbench-10_processed_latents; N=10;  FI=artbench-10_processed-fid_stats.npz; FD=artbench-10-fd_dino-vitb14_stats.npz ;;
  cub200)       L=cub-200-2011_processed_latents; N=200; FI=cub-200-2011_processed-fid_stats.npz; FD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      L=food-101_processed_latents; N=101; FI=food-101_processed-fid_stats.npz; FD=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) L=stanford-cars_processed_latents; N=196; FI=stanford_cars_processed-fid_stats.npz; FD=stanford-cars-fd_dino-vitb14_stats.npz ;;
  esac
  echo "--config.dataset.name=${1}_latent --config.dataset.root=$ADV/../datasets/$L --config.dataset.class_mapping_root= --config.dataset.num_classes=$N --config.model.num_classes=$N --config.sampling.num_classes=$N --config.fid.cache_ref=$ADV/files/fid_stats/$FI --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FD"; }
ftck(){ ls -d $F/plain_SiT_finetune_${1}_rebuttal_20261024_155255_*/best_fid/checkpoint_* | grep -E "_155255_[a-z0-9]{6}/best_fid/" | sort -V | tail -1; }
# job = kind:ds ; prints the eval command if the job is ready, nothing otherwise
ready(){ local kind=${1%%:*} ds=${1#*:} wd sess
  if [[ $kind == meft ]]; then wd=$W/${ds}_SiT_MeFT_AFM_tab2_20261028; sess=afmsit_$ds; [[ -f $wd/eval_best_fid_4steps/eval_metrics.csv ]] && return
  else wd=$W/${ds}_SiT_FT_AFM_tab2_20261028; sess=afmsitft_$ds; ls $wd/eval_best_fid_4steps_omega*/eval_metrics.csv >/dev/null 2>&1 && return; fi
  screen -ls | grep -q "\.$sess\b" && return
  local best; best=$(ls -d $wd/best_fid/checkpoint_* 2>/dev/null | sort -V | tail -1); [[ -n $best ]] || return
  if [[ $kind == meft ]]; then echo "CONFIG_MODE=sit_meft_afm_tab2 PYTHON=$PY USE_WANDB=False bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh $wd 1 2 4 -- $(args $ds)"
  else echo "$PY main_afm_sit_ft.py --config=$ADV/configs/load_config.py:sit_ft_afm_tab2 $(args $ds) --workdir=$wd --config.load_from=$(ftck $ds) --final_eval_ckpt=$best"; fi; }
declare -A DONE=() GPU_OF=() STABLE=()
log "=== start; waiting for afm_sit_ft_tab2_watcher to launch its last job (food101) ==="
until grep -q "LAUNCH food101" $ADV/files/logs/afm_sit_ft_tab2_watcher.log 2>/dev/null; do sleep 120; done
log "FT watcher has launched food101; claiming free GPUs"
while :; do
  for j in "${!GPU_OF[@]}"; do screen -ls | grep -q "\.afmeval_${j/:/_}\b" || { log "EVAL DONE $j (GPU ${GPU_OF[$j]})"; DONE[$j]=1; unset "GPU_OF[$j]"; }; done
  pending=0; for ds in "${DS[@]}"; do for k in meft ft; do j=$k:$ds; [[ -n ${DONE[$j]:-} || -n ${GPU_OF[$j]:-} ]] && continue
    w=$W/${ds}_SiT_$([[ $k == meft ]] && echo MeFT || echo FT)_AFM_tab2_20261028; [[ -d $w ]] && pending=$((pending+1)); done; done
  (( pending == 0 && ${#GPU_OF[@]} == 0 )) && ! screen -ls | grep -qE "\.afmsit(ft)?_" && { log "=== all final evals done ==="; exit 0; }
  for g in $POOL; do busy=0; for j in "${!GPU_OF[@]}"; do [[ ${GPU_OF[$j]} == "$g" ]] && busy=1; done
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( busy == 0 && m < 3000 )); then STABLE[$g]=$(( ${STABLE[$g]:-0} + 1 )); else STABLE[$g]=0; fi; done
  for ds in "${DS[@]}"; do for k in meft ft; do j=$k:$ds; [[ -n ${DONE[$j]:-} || -n ${GPU_OF[$j]:-} ]] && continue
    cmd=$(ready $j); [[ -n $cmd ]] || continue
    g=""; for c in $POOL; do (( ${STABLE[$c]:-0} >= STABLE_NEEDED )) && { g=$c; break; }; done; [[ -n $g ]] || continue 2
    STABLE[$g]=0; GPU_OF[$j]=$g; log "EVAL $j on GPU $g: $cmd"
    screen -dmS afmeval_${j/:/_} bash -c "cd $ADV; export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/mpl-afmeval-$k-$ds; $cmd 2>&1 | tee -a $ADV/files/logs/afmeval_${k}_$ds.log"
    sleep 20; done; done
  sleep 90
done
