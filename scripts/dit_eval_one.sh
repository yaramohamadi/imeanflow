#!/usr/bin/env bash
# Run the deferred DiT final best_fid eval (1 2 250) for ONE dataset on ONE GPU.
# Mirrors the training wrapper's final-eval env: DMF DiT backbone, dogfit off,
# null-class OFF, dropout 0.0 (DiT-specific). Loads the run's best_fid checkpoint.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
ds=$1; gpu=$2
case "$ds" in
  caltech101)   nc=101; fid=caltech-101-fid_stats.npz;            fdd=caltech-101-fd_dino-vitb14_stats.npz;   lat=caltech-101_processed_latents ;;
  artbench10)   nc=10;  fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz;   lat=artbench-10_processed_latents ;;
  cub200)       nc=200; fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz;  lat=cub-200-2011_processed_latents ;;
  food101)      nc=101; fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz;      lat=food-101_processed_latents ;;
  stanfordcars) nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz; lat=stanford-cars_processed_latents ;;
  *) echo "unknown $ds"; exit 1;;
esac
# the real meft_online DiT run for this dataset (the 1717xx one, has best_fid ckpt)
rd=$(ls -d files/logs/finetuning/*${ds}*DiT_DMF*meft_online*1717*/ 2>/dev/null | head -1); rd=${rd%/}
[ -z "$rd" ] && { echo "no DiT run dir for $ds"; exit 1; }
ls -d "$rd"/best_fid/checkpoint_* >/dev/null 2>&1 || { echo "no best_fid ckpt in $rd"; exit 1; }
[ -f "$rd/eval_best_fid_250steps/eval_metrics.csv" ] && [ -f "$rd/eval_best_fid_1steps/eval_metrics.csv" ] && { echo "already done $ds"; exit 0; }
root="$DATA/$lat"
sh="files/logs/run_diteval_${ds}.sh"
cat > "$sh" <<EOF
#!/bin/bash
cd $REPO
# Use the ddpmv config which byte-matches the DiT-DMF training model section,
# INCLUDING target_model_time_flip=True / target_model_time_scale=999.0 /
# target_wrapper_eps=1e-3 (the dit_native time remapping). dogfit_meanflow left
# those on defaults (flip=false, scale=1.0) -> nonsense timesteps -> noise (FID ~388).
# Validated: ddpmv config reproduces training FID 54.50 exactly on caltech101.
CONFIG_MODE=caltech_dit_dmf_ddpmv \\
PYTHON=$REPO/.venv/bin/python \\
USE_WANDB=False \\
MODEL_STR=imfDiT_DMF_XL_2 \\
MODEL_USE_DOGFIT=False \\
TARGET_USE_NULL_CLASS=True \\
CLASS_DROPOUT_PROB=0.1 \\
TARGET_OUTPUT_PREDICTION_SPACE=noise \\
TARGET_VELOCITY_MAP_MODE=dit_native \\
USE_EMA_VC=False \\
DATASET_ROOT=$root \\
DATASET_NUM_CLASSES=$nc \\
FID_CACHE_REF=$REPO/files/fid_stats/$fid \\
FD_DINO_CACHE_REF=$REPO/files/fdd_stats/$fdd \\
CUDA_VISIBLE_DEVICES=$gpu \\
TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \\
bash scripts/eval_best_fid_steps.sh "$rd" 1 2 250
EOF
chmod +x "$sh"
screen -dmS "diteval_${ds}" bash -c "bash $sh 2>&1 | tee -a $REPO/files/logs/diteval_${ds}_run.log"
echo "LAUNCHED DiT eval $ds on GPU $gpu (screen diteval_${ds})"
