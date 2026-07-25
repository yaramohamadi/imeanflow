#!/usr/bin/env bash
# Final best_fid eval (1 2 250) for the 5 DiT-DMF EMA-vc runs, one per free GPU.
# Mirrors scripts/dit_eval_one.sh (the VALIDATED online-DiT recipe) but points at
# the *emavc* run dirs. Key correctness bits (see memories dit_eval_prediction_space
# + dit_eval_nullclass): CONFIG_MODE=caltech_dit_dmf_ddpmv byte-matches the training
# model section (dit_native + target_model_time_flip=True + scale=999.0 + wrapper_eps
# =1e-3), TARGET_USE_NULL_CLASS=True + CLASS_DROPOUT_PROB=0.1 (102-row y_embedder),
# USE_EMA_VC=False (eval samples online student weights; the EMA teacher is train-only).
# Idempotent: skips a dataset whose 1/2/250 CSVs already exist.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"

# dataset : gpu   (GPUs 4 and 6 are busy with JiT -> use 0,1,2,3,5)
RUNS=(
  "caltech101:0"
  "artbench10:1"
  "cub200:2"
  "food101:3"
  "stanfordcars:5"
)

meta() {  # sets nc/fid/fdd/lat for $1
  case "$1" in
    caltech101)   nc=101; fid=caltech-101-fid_stats.npz;            fdd=caltech-101-fd_dino-vitb14_stats.npz;   lat=caltech-101_processed_latents ;;
    artbench10)   nc=10;  fid=artbench-10_processed-fid_stats.npz;   fdd=artbench-10-fd_dino-vitb14_stats.npz;   lat=artbench-10_processed_latents ;;
    cub200)       nc=200; fid=cub-200-2011_processed-fid_stats.npz;  fdd=cub-200-2011-fd_dino-vitb14_stats.npz;  lat=cub-200-2011_processed_latents ;;
    food101)      nc=101; fid=food-101_processed-fid_stats.npz;      fdd=food-101-fd_dino-vitb14_stats.npz;      lat=food-101_processed_latents ;;
    stanfordcars) nc=196; fid=stanford_cars_processed-fid_stats.npz; fdd=stanford-cars-fd_dino-vitb14_stats.npz; lat=stanford-cars_processed_latents ;;
    *) echo "unknown $1"; return 1;;
  esac
}

for entry in "${RUNS[@]}"; do
  IFS=':' read -r ds gpu <<< "$entry"
  meta "$ds" || continue
  rd=$(ls -d files/logs/finetuning/*${ds}*DiT_DMF*emavc*/ 2>/dev/null | head -1); rd=${rd%/}
  [ -z "$rd" ] && { echo "[$ds] no emavc run dir"; continue; }
  ls -d "$rd"/best_fid/checkpoint_* >/dev/null 2>&1 || { echo "[$ds] no best_fid ckpt in $rd"; continue; }
  if [ -f "$rd/eval_best_fid_1steps/eval_metrics.csv" ] && \
     [ -f "$rd/eval_best_fid_2steps/eval_metrics.csv" ] && \
     [ -f "$rd/eval_best_fid_250steps/eval_metrics.csv" ]; then
    echo "[$ds] already done"; continue
  fi
  root="$DATA/$lat"
  sh="files/logs/run_ditemavceval_${ds}.sh"
  cat > "$sh" <<EOF
#!/bin/bash
cd $REPO
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
  screen -dmS "ditemavceval_${ds}" bash -c "bash $sh 2>&1 | tee -a $REPO/files/logs/ditemavceval_${ds}_run.log"
  echo "LAUNCHED emavc eval $ds on GPU $gpu (screen ditemavceval_${ds}) rd=$(basename $rd)"
  sleep 2
done
echo "=== screens ==="; screen -ls | grep ditemavceval || true
