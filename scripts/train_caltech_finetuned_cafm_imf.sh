#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  CALTECH_IMF_CHECKPOINT=/path/to/caltech/imf/checkpoint \
    CUDA_VISIBLE_DEVICES=0,1 \
    bash scripts/train_caltech_finetuned_cafm_imf.sh <run_label> \
    [extra config overrides]
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

REPO=/home/ens/Zdehghani/imeanflow
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CALTECH_IMF_CHECKPOINT="${CALTECH_IMF_CHECKPOINT:-}"
DATASET_ROOT="${DATASET_ROOT:-/home/ens/Zdehghani/datasets/caltech-101_processed_latents}"
LOG_ROOT="${LOG_ROOT:-$REPO/files/logs/cafm_imf}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-$LOG_ROOT/caltech_original_cafm_${RUN_LABEL}_${STAMP}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$CALTECH_IMF_CHECKPOINT" || ! -e "$CALTECH_IMF_CHECKPOINT" ]]; then
  echo "ERROR: CALTECH_IMF_CHECKPOINT must point to a fine-tuned iMF checkpoint." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: Caltech latent directory missing: $DATASET_ROOT/train" >&2
  exit 2
fi

mkdir -p "$WORKDIR"
echo "Original CAFM-on-iMF workdir: $WORKDIR"
echo "Fine-tuned iMF checkpoint: $CALTECH_IMF_CHECKPOINT"
echo "Objective: JVP CAFM, lambda_adv=1, lambda_ot=0, lambda_cp=0.001"
echo "Schedule: 10000 D warm-up batches, then 16D:1G"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

cd "$REPO"
exec "$PYTHON" main_cafm_imf.py \
  --workdir="$WORKDIR" \
  --config=configs/load_config.py:caltech_finetuned_cafm_imf_posttrain \
  --config.load_from="$CALTECH_IMF_CHECKPOINT" \
  --config.dataset.root="$DATASET_ROOT" \
  --config.logging.wandb_name="caltech_original_cafm_${RUN_LABEL}" \
  "$@" \
  2>&1 | tee -a "$WORKDIR/output.log"
