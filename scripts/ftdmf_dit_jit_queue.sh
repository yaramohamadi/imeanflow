#!/usr/bin/env bash
# 2-stage baseline FT -> DMF for DiT and JiT (rebuttal Table R1), all 5 datasets, run on dev6.
# Stage 2 = the exact Tab. 2 MF-T recipe of each backbone, with load_from = the target-domain FT checkpoint
# converted to the official torch layout (scripts/ft_sit_to_torch.py for DiT, scripts/ft_jit_to_torch.py for JiT;
# round trips bit-identical):
#   DiT: scripts/dit_clip_rerun_watcher.sh recipe (CONFIG_MODE caltech_dit_dmf_ddpmv, grad clip 1.0, 30K steps,
#        5K-sample in-training FID), then its final best-ckpt eval at NFE 1 2 4 8 (matches Tab. 2's 10K evals).
#   JiT: scripts/jitdmf_watcher.sh recipe (caltech_jit_dmf_meft, grad clip 1.0, 40K steps, 5K-sample in-training
#        FID), then the scripts/redo10k_watcher.sh 10K re-eval of the best checkpoint at NFE 1 2 4 8.
# GPUs are claimed when free; one job per GPU; JiT first (longer).
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
PY=$REPO/.venv/bin/python
cd "$REPO"
LOG=$REPO/files/logs/ftdmf_dit_jit_queue.log; mkdir -p $REPO/files/logs $REPO/files/workdirs $REPO/files/logs/redo10k
POLL_S=60; STABLE_NEEDED=2; MEM_MB=3000; UTIL_PCT=15; ALLOWED="${ALLOWED:-0 1 2 3 4 5 6 7}"
declare -A NC=([artbench10]=10 [caltech101]=101 [cub200]=200 [food101]=101 [stanfordcars]=196)
declare -A LAT=([artbench10]=artbench-10_processed_latents [caltech101]=caltech-101_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents)
declare -A IMG=([artbench10]=artbench-10_images [caltech101]=caltech-101_images [cub200]=cub-200-2011_images [food101]=food-101_images [stanfordcars]=stanford-cars_images)
declare -A FID=([artbench10]=artbench-10_processed-fid_stats.npz [caltech101]=caltech-101-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz)
declare -A FDD=([artbench10]=artbench-10-fd_dino-vitb14_stats.npz [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz)
JOBS=(jit:cub200 jit:stanfordcars jit:artbench10 jit:caltech101 jit:food101 dit:cub200 dit:stanfordcars dit:artbench10 dit:caltech101 dit:food101)
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_marker(){ echo "$REPO/files/logs/ftdmf_${1%%:*}_${1##*:}.done"; }
job_cmd(){  # job gpu -> one bash line: train then eval, then touch the done marker
  local kind=${1%%:*} ds=${1##*:} gpu=$2 fid="$REPO/files/fid_stats/${FID[${1##*:}]}" fdd="$REPO/files/fdd_stats/${FDD[${1##*:}]}"
  local env="export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/mpl-ftdmf-$kind-$ds WANDB_MODE=offline"
  if [[ $kind == dit ]]; then
    echo "$env; cd $REPO; CONFIG_MODE=caltech_dit_dmf_ddpmv PYTHON=$PY USE_WANDB=False DATASET_NAME=$ds DATASET_ROOT=$DATA/${LAT[$ds]} DATASET_NUM_CLASSES=${NC[$ds]} FID_CACHE_REF=$fid FD_DINO_CACHE_REF=$fdd ENABLE_DOGFIT=False TRAIN_USE_EMA=False USE_EMA_VC=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8' FINAL_EVAL_USE_WANDB=False bash scripts/run_caltech_dit_dmf_meanflow_taylor.sh ftdmf_$ds --config.load_from=$REPO/files/weights/dit_ft_rebuttal/$ds.pt --config.training.debug_log_during_train=False --config.training.grad_clip_norm=1.0 --config.training.max_train_steps=30000 --config.fid.num_samples=5000 2>&1 | tee -a $REPO/files/logs/ftdmf_dit_$ds.log; touch $(done_marker $1)"
  else
    local wd=$REPO/files/workdirs/jitftdmf_$ds out=$REPO/files/logs/redo10k/jitftdmf_$ds
    echo "$env; cd $REPO; mkdir -p $wd $out; $PY main_imf_jit.py --workdir=$wd --config=configs/load_config.py:caltech_jit_dmf_meft --config.dataset.root=$DATA/${IMG[$ds]} --config.dataset.num_classes=${NC[$ds]} --config.model.num_classes=${NC[$ds]} --config.fid.cache_ref=$fid --config.fd_dino.cache_ref=$fdd --config.load_from=$REPO/files/weights/jit_ft_rebuttal/$ds.pt --config.logging.use_wandb=False --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True --config.training.max_train_steps=40000 --config.training.grad_clip_norm=1.0 --config.training.force_fid_per_step=2500 --config.fid.num_samples=5000 2>&1 | tee -a $REPO/files/logs/ftdmf_jit_$ds.log; ck=\$(ls -d $wd/best_fid/checkpoint_* | tail -1); $PY main_imf_jit.py --workdir=$out --config=configs/load_config.py:caltech_jit_dmf_meft --config.eval_only=True --config.load_from=\$ck --config.dataset.root=$DATA/${IMG[$ds]} --config.dataset.num_classes=${NC[$ds]} --config.model.num_classes=${NC[$ds]} --config.fid.cache_ref=$fid --config.fd_dino.cache_ref=$fdd --config.logging.use_wandb=False --config.training.force_metric_num_steps='1 2 4 8' --config.fid.num_samples=10000 2>&1 | tee -a $out/run.log; touch $(done_marker $1)"
  fi
}
declare -A STABLE=() CLAIMED=() DONE=()
for j in "${JOBS[@]}"; do [[ -f $(done_marker $j) ]] && DONE[$j]=1; done
remaining(){ local n=0 j; for j in "${JOBS[@]}"; do [[ -z "${DONE[$j]:-}" ]] && n=$((n+1)); done; echo $n; }
log "=== FT->DMF DiT/JiT queue: ${#JOBS[@]} jobs, GPUs [$ALLOWED] ==="
while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for line in "${SMI[@]}"; do IFS=', ' read -r idx mem util <<< "$line"
    [[ " $ALLOWED " == *" $idx "* ]] || continue; [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi; done
  for idx in "${!CLAIMED[@]}"; do j=${CLAIMED[$idx]}; s="ftdmf_${j%%:*}_${j##*:}"
    screen -ls | grep -qE "\.${s}[[:space:]]" && continue
    log "FINISHED $j on GPU $idx (marker: $([[ -f $(done_marker $j) ]] && echo yes || echo NO))"; DONE[$j]=1; unset "CLAIMED[$idx]"; STABLE[$idx]=0; done
  for idx in $ALLOWED; do [[ -n "${CLAIMED[$idx]:-}" ]] && continue; (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for j in "${JOBS[@]}"; do [[ -n "${DONE[$j]:-}" ]] && continue
      busy=0; for c in "${CLAIMED[@]}"; do [[ $c == "$j" ]] && busy=1; done; (( busy )) && continue
      s="ftdmf_${j%%:*}_${j##*:}"; log "LAUNCH $j on GPU $idx (screen $s)"
      screen -dmS "$s" bash -c "$(job_cmd $j $idx)"; CLAIMED[$idx]=$j; STABLE[$idx]=0; sleep 15; break; done; done
  (( $(remaining) > 0 )) && sleep $POLL_S
done
log "=== all FT->DMF DiT/JiT jobs finished ==="
