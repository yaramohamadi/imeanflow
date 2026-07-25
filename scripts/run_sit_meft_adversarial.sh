#!/usr/bin/env bash
# Run AFM or CA-iMF post-training on a SiT-DMF MeFT checkpoint.
#
# Usage:
#   bash scripts/run_sit_meft_adversarial.sh afm caltech101 <checkpoint> <workdir>
#   bash scripts/run_sit_meft_adversarial.sh caimf caltech101 <checkpoint> <workdir>
#
# The checkpoint must be a Flax checkpoint directory such as
# files/SiT/.../best_fid/checkpoint_15000.  Optimizer/discriminator state is
# intentionally reset; only the generator parameters are restored.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <afm|caimf> <dataset> <load_from> <workdir>" >&2
  exit 2
fi

METHOD="$1"
DATASET="$2"
LOAD_FROM="$3"
WORKDIR="$4"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON:-$REPO/.venv/bin/python}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"

DATA_ROOT="${DATA_ROOT:-$REPO/../datasets}"
case "$DATASET" in
  caltech101)   LATENT_DIR="caltech-101_processed_latents"; NC=101; FID="caltech-101-fid_stats.npz"; FDD="caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   LATENT_DIR="artbench-10_processed_latents"; NC=10; FID="artbench-10_processed-fid_stats.npz"; FDD="artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       LATENT_DIR="cub-200-2011_processed_latents"; NC=200; FID="cub-200-2011_processed-fid_stats.npz"; FDD="cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      LATENT_DIR="food-101_processed_latents"; NC=101; FID="food-101_processed-fid_stats.npz"; FDD="food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) LATENT_DIR="stanford-cars_processed_latents"; NC=196; FID="stanford_cars_processed-fid_stats.npz"; FDD="stanford-cars-fd_dino-vitb14_stats.npz" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
DATA_ROOT="$DATA_ROOT/$LATENT_DIR"

case "$METHOD" in
  afm)
    CONFIG_MODE="${CONFIG_MODE:-caltech_sit_meft_afm_posttrain}"
    MAIN="$REPO/main_afm_sit_meft.py"
    ;;
  caimf)
    CONFIG_MODE="${CONFIG_MODE:-caltech_sit_meft_caimf_posttrain}"
    MAIN="$REPO/main_caimf_sit_meft.py"
    ;;
  *) echo "METHOD must be afm or caimf" >&2; exit 2 ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$REPO/configs/${CONFIG_MODE}_config.yml" ]] || {
  echo "Config not found: $REPO/configs/${CONFIG_MODE}_config.yml" >&2
  exit 3
}
[[ -d "$LOAD_FROM" ]] || { echo "Checkpoint directory not found: $LOAD_FROM" >&2; exit 3; }
if [[ -d "$WORKDIR" ]] && find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print -quit | grep -q .; then
  echo "Workdir already contains a checkpoint: $WORKDIR" >&2
  echo "Use a new workdir (CA-iMF/AFM do not resume from this launcher)." >&2
  exit 4
fi
mkdir -p "$WORKDIR"
CONFIG_SPEC="$REPO/configs/load_config.py:$CONFIG_MODE"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

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
  --workdir="$WORKDIR"
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi

if [[ "${RUN_FINAL_EVAL:-True}" == "True" ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "${FINAL_EVAL_STEPS:-1 2}"
  FINAL_EVAL_ARGS=(
    "--config.dataset.name=${DATASET}_latent"
    "--config.dataset.root=$DATA_ROOT"
    "--config.dataset.class_mapping_root="
    "--config.dataset.num_classes=$NC"
    "--config.model.num_classes=$NC"
    "--config.sampling.num_classes=$NC"
    "--config.fid.cache_ref=$REPO/files/fid_stats/$FID"
    "--config.fd_dino.cache_ref=$REPO/files/fdd_stats/$FDD"
  )
  CONFIG_MODE="$CONFIG_MODE" PYTHON="$PYTHON_BIN" USE_WANDB=False \
    bash "$REPO/scripts/eval_best_fid_steps_sit_meft_adversarial.sh" \
      "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- "${FINAL_EVAL_ARGS[@]}"
fi
