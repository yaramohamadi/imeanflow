#!/usr/bin/env bash
# Run CA-iMF (CAMF) pure-adversarial post-training on a NATIVE iMF checkpoint
# (plain imfDiT_XL_2 improved-MeanFlow, NOT a DMF student).
#
# Usage:
#   bash scripts/run_imf_caimf.sh <dataset> <load_from> <workdir>
#
# train_caimf.py accepts model_str=imfDiT_* directly, so this uses main_caimf.py
# (no _sit_meft shim). Optimizer/discriminator state is reset; only generator
# params are restored from <load_from>. Final eval (NFE 1 & 2) runs through the
# plain-imf eval script. One GPU per run (CUDA_VISIBLE_DEVICES from caller).
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <dataset> <load_from> <workdir>" >&2
  exit 2
fi

DATASET="$1"
LOAD_FROM="$2"
WORKDIR="$3"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON:-$REPO/.venv/bin/python}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_MODE="${CONFIG_MODE:-caltech_imf_caimf_posttrain}"
MAIN="$REPO/main_caimf.py"

DATA_ROOT_BASE="${DATA_ROOT:-$REPO/../datasets}"
case "$DATASET" in
  caltech101)   LATENT_DIR="caltech-101_processed_latents"; NC=101; FID="caltech-101-fid_stats.npz"; FDD="caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   LATENT_DIR="artbench-10_processed_latents"; NC=10;  FID="artbench-10_processed-fid_stats.npz"; FDD="artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       LATENT_DIR="cub-200-2011_processed_latents"; NC=200; FID="cub-200-2011_processed-fid_stats.npz"; FDD="cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      LATENT_DIR="food-101_processed_latents"; NC=101; FID="food-101_processed-fid_stats.npz"; FDD="food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) LATENT_DIR="stanford-cars_processed_latents"; NC=196; FID="stanford_cars_processed-fid_stats.npz"; FDD="stanford-cars-fd_dino-vitb14_stats.npz" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
DATA_ROOT="$DATA_ROOT_BASE/$LATENT_DIR"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$REPO/configs/${CONFIG_MODE}_config.yml" ]] || { echo "Config not found: $REPO/configs/${CONFIG_MODE}_config.yml" >&2; exit 3; }
[[ -d "$LOAD_FROM" ]] || { echo "Checkpoint directory not found: $LOAD_FROM" >&2; exit 3; }
[[ -d "$DATA_ROOT/train" ]] || { echo "Latent train dir not found: $DATA_ROOT/train" >&2; exit 3; }
if [[ -d "$WORKDIR" ]] && find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print -quit | grep -q .; then
  echo "Workdir already contains a checkpoint: $WORKDIR" >&2
  echo "Use a new workdir (CA-iMF does not resume from this launcher)." >&2
  exit 4
fi
mkdir -p "$WORKDIR"
CONFIG_SPEC="$REPO/configs/load_config.py:$CONFIG_MODE"

cd "$REPO"

# Optional full-state resume: continue a prior run to a larger step budget.
# RESUME_FROM = a periodic checkpoint_* dir (or workdir) with FULL train
# state; MAX_POSTTRAIN_BATCHES overrides the config step cap (e.g. 300000).
EXTRA_CFG_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -d "$RESUME_FROM" ]] || { echo "RESUME_FROM dir not found: $RESUME_FROM" >&2; exit 3; }
  EXTRA_CFG_ARGS+=(--config.caimf.resume_from="$RESUME_FROM")
  echo "RESUME (full state) from: $RESUME_FROM"
fi
if [[ -n "${MAX_POSTTRAIN_BATCHES:-}" ]]; then
  EXTRA_CFG_ARGS+=(--config.caimf.max_posttrain_batches="$MAX_POSTTRAIN_BATCHES")
  echo "max_posttrain_batches override: $MAX_POSTTRAIN_BATCHES"
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "CA-iMF-on-iMF workdir: $WORKDIR"
echo "Dataset: $DATASET  (root=$DATA_ROOT, NC=$NC)"
echo "iMF checkpoint: $LOAD_FROM"
echo "Config: $CONFIG_MODE  GPU: $GPU_LIST"

set +e
"$PYTHON_BIN" "$MAIN" \
  --config="$CONFIG_SPEC" \
  --config.load_from="$LOAD_FROM" \
  --config.dataset.name="${DATASET}_latent" \
  --config.dataset.root="$DATA_ROOT" \
  --config.dataset.class_mapping_root="" \
  --config.dataset.num_classes="$NC" \
  --config.model.num_classes="$NC" \
  --config.sampling.num_classes="$NC" \
  --config.fid.cache_ref="$REPO/files/fid_stats/$FID" \
  --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD" \
  "${EXTRA_CFG_ARGS[@]}" \
  --workdir="$WORKDIR"
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  echo "Training exited with status $TRAIN_STATUS; skipping final eval." >&2
  exit "$TRAIN_STATUS"
fi

if [[ "${RUN_FINAL_EVAL:-True}" == "True" ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS:-1 2}"
  echo "Training finished. Final best-FID eval at NFE: ${FINAL_EVAL_STEP_ARRAY[*]}"
  CONFIG_MODE="$CONFIG_MODE" PYTHON="$PYTHON_BIN" USE_WANDB=False \
    bash "$REPO/scripts/eval_best_fid_steps_plain_imf.sh" \
      "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
      --config.dataset.name="${DATASET}_latent" \
      --config.dataset.root="$DATA_ROOT" \
      --config.dataset.class_mapping_root="" \
      --config.dataset.num_classes="$NC" \
      --config.model.num_classes="$NC" \
      --config.sampling.num_classes="$NC" \
      --config.fid.cache_ref="$REPO/files/fid_stats/$FID" \
      --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD"
fi
