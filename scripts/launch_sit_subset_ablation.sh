#!/usr/bin/env bash
# SiT data-subset ablation: MF-T vs 2-stage FT -> DMF on CUB-200 and Stanford Cars at 10% / 25% of the train set.
# Subsets = seed-42 class-stratified lists in datasets/subsets_seed42/ (floor per class, min 1 per class;
# verified against our latent dirs: name-list sha256, per-class counts, no missing files).
# Each subset gets two screens:
#   subabl_mft_<ds>_<p>  MF-T from ImageNet SiT        (recipe = scripts/gen_launch_sit_meft.sh)
#   subabl_ft_<ds>_<p>   FT (scripts/launch_sit_ft_all5.sh recipe), then ft_sit_to_torch.py, then DMF from it
#                        (scripts/launch_ft_dmf_all5.sh recipe), chained on the same GPU.
# Same protocol as the 100% runs: 30K optimizer steps, same configs. Two fields depend on dataset size and are reset:
#   num_epochs 100000, so max_train_steps alone ends training (200 epochs of CUB 10% would stop at ~7K steps);
#   warmup_epochs chosen so warmup has the 100%-run step count (FT 2.5 x steps/epoch, MF-T/DMF 5 x steps/epoch).
# FID/FDD use the full-dataset reference stats, as the 100% runs do. wandb offline.
# Usage: GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_sit_subset_ablation.sh
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
SUBS=$DATA/subsets_seed42
PY=$REPO/.venv/bin/python
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p $REPO/files/wandb_offline $REPO/files/weights/sit_ft_subset
cd $REPO

# dataset  json_prefix  latent_dir  classes  full_steps_per_epoch  fid_stats  fdd_stats
SPECS=(
  "cub200       cub-200-2011  cub-200-2011_processed_latents  200 737  cub-200-2011_processed-fid_stats.npz  cub-200-2011-fd_dino-vitb14_stats.npz"
  "stanfordcars stanford-cars stanford-cars_processed_latents 196 1008 stanford_cars_processed-fid_stats.npz stanford-cars-fd_dino-vitb14_stats.npz"
)
COMMON_ARGS="--config.dataset.num_workers=12 --config.dataset.prefetch_factor=4 --config.dataset.pin_memory=True \
  --config.training.max_train_steps=30000 --config.training.num_epochs=100000"

g=0
for spec in "${SPECS[@]}"; do
  read -r ds jp lat nc full_spe fid fdd <<< "$spec"
  for p in 10 25; do
    sub=$DATA/${lat}_${p}pct_seed42
    # Build the subset dir of symlinks (idempotent; checks the count).
    n=$($PY -I - "$SUBS/${jp}_train_${p}pct_seed42.json" "$DATA/$lat/train" "$sub/train" <<'EOF'
import json, os, sys
js, src, dst = sys.argv[1:]
sel = json.load(open(js))["selected_files"]
os.makedirs(dst, exist_ok=True)
for f in sel:
    if not os.path.lexists(f"{dst}/{f}"):
        os.symlink(f"{src}/{f}", f"{dst}/{f}")
have = [f for f in os.listdir(dst) if f.endswith(".pt")]
assert sorted(have) == sorted(sel), (len(have), len(sel))
print(len(sel))
EOF
    ) || { echo "ERROR building $sub"; exit 1; }
    spe=$((n / 16))  # loader steps per epoch at batch 16, drop_last
    w_ft=$(awk "BEGIN{printf \"%.6f\", (int(2.5*$full_spe)+0.5)/$spe}")
    w_mf=$(awk "BEGIN{printf \"%.6f\", (int(5*$full_spe)+0.5)/$spe}")
    tag=sub${p}pct_s42
    env_common="export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$REPO/files/wandb_offline
      export DATASET_ROOT=$sub DATASET_NUM_CLASSES=$nc PYTHON=$PY TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
      export FID_CACHE_REF=$REPO/files/fid_stats/$fid FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd"
    echo "$ds $p%: $n images, $spe steps/epoch, warmup_epochs FT $w_ft, MF-T/DMF $w_mf"

    # MF-T from ImageNet SiT.
    gpu=${GPUS[$g]}; g=$((g + 1)); sess=subabl_mft_${ds}_${p}
    if screen -ls | grep -q "\.${sess}\b"; then echo "SKIP $sess running"; else
      echo "LAUNCH $sess on GPU $gpu"
      screen -dmS "$sess" bash -c "
        cd $REPO; export CUDA_VISIBLE_DEVICES=$gpu
        $env_common
        export LOAD_FROM=$REPO/files/weights/SiT-XL-2-256.pt
        export USE_WANDB=True FINAL_EVAL_USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8'
        export WANDB_PROJECT=sit_meft
        bash scripts/train_sit_dmf_no_dogfit_dataset.sh $ds meft_$tag \
          --config.load_from=$REPO/files/weights/SiT-XL-2-256.pt --config.logging.wandb_project=sit_meft \
          $COMMON_ARGS --config.training.warmup_epochs=$w_mf \
          --config.fid.num_samples=5000 --config.training.force_fid_per_step=2500 \
          --config.training.debug_log_during_train=False \
          2>&1 | tee -a $REPO/files/logs/subabl_mft_${tag}_${ds}.log
      "
    fi

    # FT, then convert its best-FID checkpoint, then DMF from it.
    gpu=${GPUS[$g]}; g=$((g + 1)); sess=subabl_ft_${ds}_${p}; init=$REPO/files/weights/sit_ft_subset/${ds}_${p}pct.pt
    if screen -ls | grep -q "\.${sess}\b"; then echo "SKIP $sess running"; else
      echo "LAUNCH $sess on GPU $gpu"
      screen -dmS "$sess" bash -c "
        cd $REPO; export CUDA_VISIBLE_DEVICES=$gpu
        $env_common
        CONFIG_MODE=plain_sit_finetune_1gpu USE_WANDB=True BACKBONE=sit LOAD_FROM=$REPO/files/weights/SiT-XL-2-256.pt \
        DATASET_NAME=$ds RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8 250' \
        bash scripts/train_plain_sit_finetune.sh $tag $COMMON_ARGS --config.training.warmup_epochs=$w_ft \
          2>&1 | tee -a $REPO/files/logs/subabl_ft_${tag}_${ds}.log
        ftdir=\$(ls -dt files/logs/finetuning/plain_SiT_finetune_${ds}_${tag}_* | head -1)
        ck=\$(ls -d \$ftdir/best_fid/checkpoint_* | sort -V | tail -1)
        [ -n \"\$ck\" ] || { echo 'ERROR no FT best_fid checkpoint' | tee -a $REPO/files/logs/subabl_ft_${tag}_${ds}.log; exit 1; }
        echo \"CONVERT \$ck -> $init\" | tee -a $REPO/files/logs/subabl_ft_${tag}_${ds}.log
        JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY scripts/ft_sit_to_torch.py \$ck $init --check 2>&1 | tee -a $REPO/files/logs/subabl_ft_${tag}_${ds}.log
        [ \${PIPESTATUS[0]} -eq 0 ] || { echo 'ERROR conversion failed' | tee -a $REPO/files/logs/subabl_ft_${tag}_${ds}.log; exit 1; }
        export LOAD_FROM=$init
        export USE_WANDB=True FINAL_EVAL_USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 4 8'
        export WANDB_PROJECT=sit_ft_dmf
        bash scripts/train_sit_dmf_no_dogfit_dataset.sh $ds ftdmf_$tag \
          --config.load_from=$init --config.logging.wandb_project=sit_ft_dmf \
          $COMMON_ARGS --config.training.warmup_epochs=$w_mf \
          --config.fid.num_samples=5000 --config.training.force_fid_per_step=2500 \
          --config.training.debug_log_during_train=False \
          2>&1 | tee -a $REPO/files/logs/subabl_ftdmf_${tag}_${ds}.log
      "
    fi
  done
done
screen -ls | grep subabl_
