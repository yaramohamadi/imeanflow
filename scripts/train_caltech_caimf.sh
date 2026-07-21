#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ens/Zdehghani/imeanflow"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
IMF_CHECKPOINT="${IMF_CHECKPOINT:-}"
EXPERIMENT="${EXPERIMENT:-2}"
ENTRY_MODE="${ENTRY_MODE:-target_ft}"
DATASET_ROOT="${DATASET_ROOT:-/home/ens/Zdehghani/datasets/caltech-101_processed_latents}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LAMBDA_OT="${LAMBDA_OT:-0.0}"
LAMBDA_CP="${LAMBDA_CP:-0.001}"
export CUDA_VISIBLE_DEVICES

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$IMF_CHECKPOINT" ]]; then
  echo "ERROR: set IMF_CHECKPOINT to the target-finetuned or ImageNet iMF checkpoint." >&2
  exit 2
fi
if [[ ! -e "$IMF_CHECKPOINT" ]]; then
  echo "ERROR: IMF_CHECKPOINT does not exist: $IMF_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: latent train directory does not exist: $DATASET_ROOT/train" >&2
  exit 2
fi

case "$EXPERIMENT" in
  1)
    LAMBDA_IMF=0.01
    LAMBDA_ADV=1.0
    DISCRIMINATOR_UPDATES=True
    ;;
  2)
    LAMBDA_IMF=1.0
    LAMBDA_ADV=1.0
    DISCRIMINATOR_UPDATES=True
    ;;
  3)
    LAMBDA_IMF=1.0
    LAMBDA_ADV=0.01
    DISCRIMINATOR_UPDATES=True
    ;;
  5|adv_only)
    EXPERIMENT=5
    LAMBDA_IMF=0.0
    LAMBDA_ADV=1.0
    DISCRIMINATOR_UPDATES=True
    ;;
  4|baseline|mf_only)
    EXPERIMENT=4
    LAMBDA_IMF=1.0
    LAMBDA_ADV=0.0
    DISCRIMINATOR_UPDATES=False
    ;;
  *)
    echo "ERROR: EXPERIMENT must be 1, 2, 3, 4/baseline, or 5/adv_only." >&2
    exit 2
    ;;
esac

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-$REPO_ROOT/files/logs/caimf/caltech_${ENTRY_MODE}_exp${EXPERIMENT}_${TIMESTAMP}}"
mkdir -p "$WORKDIR"

echo "CA-iMF workdir: $WORKDIR"
echo "Entry mode: $ENTRY_MODE"
echo "Initial iMF checkpoint: $IMF_CHECKPOINT"
echo "Dataset: $DATASET_ROOT"
echo "Experiment: $EXPERIMENT (lambda_imf=$LAMBDA_IMF, lambda_adv=$LAMBDA_ADV, lambda_ot=$LAMBDA_OT, lambda_cp=$LAMBDA_CP)"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

exec "$PYTHON" main_caimf.py \
  --config=configs/load_config.py:caltech_caimf_posttrain \
  --workdir="$WORKDIR" \
  --config.load_from="$IMF_CHECKPOINT" \
  --config.dataset.root="$DATASET_ROOT" \
  --config.caimf.lambda_imf="$LAMBDA_IMF" \
  --config.caimf.lambda_adv="$LAMBDA_ADV" \
  --config.caimf.lambda_ot="$LAMBDA_OT" \
  --config.caimf.lambda_cp="$LAMBDA_CP" \
  --config.caimf.discriminator_updates="$DISCRIMINATOR_UPDATES" \
  --config.logging.wandb_name="caltech_${ENTRY_MODE}_caimf_exp${EXPERIMENT}" \
  "$@"
