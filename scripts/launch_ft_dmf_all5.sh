#!/usr/bin/env bash
# 2-stage baseline A: target-domain SiT FT (EXP-303..307) -> DMF on the target domain.
# Stage 2 is the exact Tab. 2 SiT MF-T recipe (scripts/gen_launch_sit_meft.sh: run_dmf_no_dogfit_taylor.sh,
# CONFIG_MODE caltech_sit_dmf_finetune, 30K steps, batch 16 x accum 2, partial_load) with one change:
# load_from = the FT checkpoint converted to torch layout by scripts/ft_sit_to_torch.py (round-trip verified).
# wandb is offline (dev5's key is ea-fc); sync later.
# Usage: RUN_LABEL=ftdmf GPUS="0 1 2 3 4" bash scripts/launch_ft_dmf_all5.sh
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
RUN_LABEL="${RUN_LABEL:-ftdmf}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"
mkdir -p $REPO/files/wandb_offline
SPECS=(
  "artbench10   artbench-10_processed_latents   10  artbench-10_processed-fid_stats.npz   artbench-10-fd_dino-vitb14_stats.npz"
  "caltech101   caltech-101_processed_latents   101 caltech-101-fid_stats.npz             caltech-101-fd_dino-vitb14_stats.npz"
  "cub200       cub-200-2011_processed_latents  200 cub-200-2011_processed-fid_stats.npz  cub-200-2011-fd_dino-vitb14_stats.npz"
  "food101      food-101_processed_latents      101 food-101_processed-fid_stats.npz      food-101-fd_dino-vitb14_stats.npz"
  "stanfordcars stanford-cars_processed_latents 196 stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz"
)
for i in "${!SPECS[@]}"; do
  read -r ds lat nc fid fdd <<< "${SPECS[$i]}"
  gpu="${GPUS[$i]}"; sess="ftdmf_${ds}"; init="$REPO/files/weights/sit_ft_rebuttal/${ds}.pt"
  [[ -f "$init" ]] || { echo "SKIP $ds: missing $init"; continue; }
  if screen -ls | grep -q "\.${sess}\b"; then echo "SKIP $ds: $sess running"; continue; fi
  echo "LAUNCH $ds on GPU $gpu from $init"
  screen -dmS "$sess" bash -c "
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    export WANDB_MODE=offline WANDB_DIR=$REPO/files/wandb_offline
    export DATASET_ROOT=$DATA/$lat DATASET_NUM_CLASSES=$nc
    export FID_CACHE_REF=$REPO/files/fid_stats/$fid FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd
    export LOAD_FROM=$init PYTHON=$REPO/.venv/bin/python
    export USE_WANDB=True FINAL_EVAL_USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8'
    export WANDB_PROJECT=sit_ft_dmf TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
    bash scripts/train_sit_dmf_no_dogfit_dataset.sh $ds $RUN_LABEL \
      --config.load_from=$init \
      --config.logging.wandb_project=sit_ft_dmf \
      --config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
      --config.training.max_train_steps=30000 \
      --config.fid.num_samples=5000 \
      --config.training.force_fid_per_step=2500 \
      --config.training.debug_log_during_train=False \
      2>&1 | tee -a $REPO/files/logs/ftdmf_${RUN_LABEL}_${ds}.log
  "
done
screen -ls | grep ftdmf_
