#!/usr/bin/env bash
# Run the MISSING 250-step final best_fid eval for all 5 finished SiT runs.
# 4-step is intentionally skipped (identical to the during-training metric).
# Picks the real meft_online run (never *smoke*). Pins to free GPUs 0 1 4,
# queueing the last 2 datasets. Idempotent: skips a dataset that already has
# eval_best_fid_250steps/eval_metrics.csv. Refuses a GPU with >2000 MiB used.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
FREE_GPUS=(0 1 4)

params(){ case "$1" in
  caltech101)   nc=101; fid=caltech-101-fid_stats.npz;            fdd=caltech-101-fd_dino-vitb14_stats.npz;   lat=caltech-101_processed_latents ;;
  artbench10)   nc=10;  fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz;   lat=artbench-10_processed_latents ;;
  cub200)       nc=200; fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz;  lat=cub-200-2011_processed_latents ;;
  food101)      nc=101; fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz;      lat=food-101_processed_latents ;;
  stanfordcars) nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz; lat=stanford-cars_processed_latents ;;
esac; }

# pick the newest NON-smoke meft_online SiT run dir for a dataset
run_dir(){ ls -dt files/logs/finetuning/*"$1"*SiT*meft_online*/ 2>/dev/null | grep -v smoke | head -1; }

launch_one(){ local ds=$1 gpu=$2
  params "$ds"
  local rd; rd=$(run_dir "$ds"); rd=${rd%/}
  [ -z "$rd" ] && { echo "SKIP $ds: no meft_online run dir"; return 1; }
  [ -f "$rd/eval_best_fid_250steps/eval_metrics.csv" ] && { echo "SKIP $ds: 250 already done"; return 0; }
  local ck; ck=$(ls -d "$rd"/best_fid/checkpoint_* 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "SKIP $ds: no best_fid checkpoint in $rd"; return 1; }
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  [ "${used:-9999}" -gt 2000 ] && { echo "REFUSE $ds: GPU $gpu busy (${used} MiB)"; return 1; }
  local root="$DATA/$lat"
  local sh="files/logs/run_sit250_${ds}.sh"
  cat > "$sh" <<EOF
#!/bin/bash
cd $REPO
CONFIG_MODE=caltech_sit_dmf_finetune \\
PYTHON=$REPO/.venv/bin/python \\
USE_WANDB=False \\
MODEL_STR=imfSiT_DMF_XL_2 \\
MODEL_USE_DOGFIT=False \\
TARGET_USE_NULL_CLASS=True \\
CLASS_DROPOUT_PROB=0.1 \\
DATASET_ROOT=$root \\
DATASET_NUM_CLASSES=$nc \\
FID_CACHE_REF=$REPO/files/fid_stats/$fid \\
FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd \\
CUDA_VISIBLE_DEVICES=$gpu \\
TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \\
bash scripts/eval_best_fid_steps.sh "$rd" 250
EOF
  chmod +x "$sh"
  screen -dmS "sit250_${ds}" bash -c "bash $sh 2>&1 | tee -a $REPO/files/logs/sit250_${ds}_run.log"
  echo "LAUNCHED $ds on GPU $gpu (screen sit250_${ds}, ckpt $(basename $ck))"
}

# initial fill: one dataset per free GPU
DSETS=(caltech101 artbench10 cub200 food101 stanfordcars)
declare -A GPU_OF
i=0
for g in "${FREE_GPUS[@]}"; do
  [ $i -ge ${#DSETS[@]} ] && break
  launch_one "${DSETS[$i]}" "$g" && GPU_OF["${DSETS[$i]}"]=$g
  i=$((i+1)); sleep 8
done
echo "REMAINING_QUEUE: ${DSETS[@]:$i}"
echo "Started ${i} evals; ${#DSETS[@]} total. Queue drains via companion watcher."
