#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  CALTECH_IMF_CHECKPOINT=/path/to/caltech/imf/checkpoint \
    bash scripts/train_caltech_finetuned_caimf.sh <run_label> [extra config overrides]

CAIMF_EXPERIMENT values:
  1  lambda_imf=0.01, lambda_adv=1.0  (recommended first run)
  2  lambda_imf=1.00, lambda_adv=1.0
  3  lambda_imf=1.00, lambda_adv=0.01
  4  lambda_imf=1.00, lambda_adv=0.0  (iMF-only control; no D)
  5  lambda_imf=0.00, lambda_adv=1.0  (adversarial-only CA-iMF)
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

REPO=/home/ens/Zdehghani/imeanflow
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CALTECH_IMF_CHECKPOINT="${CALTECH_IMF_CHECKPOINT:-}"
DATASET_ROOT="${DATASET_ROOT:-/home/ens/Zdehghani/datasets/caltech-101_processed_latents}"
CAIMF_EXPERIMENT="${CAIMF_EXPERIMENT:-1}"
LAMBDA_OT="${LAMBDA_OT:-0.0}"
LAMBDA_CP="${LAMBDA_CP:-0.001}"
LOG_ROOT="${LOG_ROOT:-$REPO/files/logs/caimf_target_ft}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$CALTECH_IMF_CHECKPOINT" || ! -e "$CALTECH_IMF_CHECKPOINT" ]]; then
  echo "ERROR: CALTECH_IMF_CHECKPOINT must point to the completed Caltech iMF fine-tuning checkpoint." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: Caltech latent directory missing: $DATASET_ROOT/train" >&2
  exit 2
fi

case "${CAIMF_EXPERIMENT,,}" in
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
  4|baseline|imf_only)
    CAIMF_EXPERIMENT=4
    LAMBDA_IMF=1.0
    LAMBDA_ADV=0.0
    LAMBDA_OT=0.0
    DISCRIMINATOR_UPDATES=False
    ;;
  5|adv_only)
    CAIMF_EXPERIMENT=5
    LAMBDA_IMF=0.0
    LAMBDA_ADV=1.0
    DISCRIMINATOR_UPDATES=True
    ;;
  *)
    echo "ERROR: CAIMF_EXPERIMENT must be 1, 2, 3, 4/baseline/imf_only, or 5/adv_only." >&2
    exit 2
    ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-$LOG_ROOT/caltech_ft_caimf_exp${CAIMF_EXPERIMENT}_${RUN_LABEL}_${STAMP}}"
mkdir -p "$WORKDIR"

echo "CA-iMF target-checkpoint workdir: $WORKDIR"
echo "Caltech-finetuned source checkpoint: $CALTECH_IMF_CHECKPOINT"
echo "Experiment: $CAIMF_EXPERIMENT"
echo "lambda_imf=$LAMBDA_IMF lambda_adv=$LAMBDA_ADV lambda_ot=$LAMBDA_OT lambda_cp=$LAMBDA_CP"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

cd "$REPO"
"$PYTHON" main_caimf.py \
  --workdir="$WORKDIR" \
  --config=configs/load_config.py:caltech_finetuned_caimf_posttrain \
  --config.load_from="$CALTECH_IMF_CHECKPOINT" \
  --config.dataset.root="$DATASET_ROOT" \
  --config.caimf.lambda_imf="$LAMBDA_IMF" \
  --config.caimf.lambda_adv="$LAMBDA_ADV" \
  --config.caimf.lambda_ot="$LAMBDA_OT" \
  --config.caimf.lambda_cp="$LAMBDA_CP" \
  --config.caimf.discriminator_updates="$DISCRIMINATOR_UPDATES" \
  --config.logging.wandb_name="caltech_ft_caimf_exp${CAIMF_EXPERIMENT}_${RUN_LABEL}" \
  "$@" \
  2>&1 | tee -a "$WORKDIR/output.log"

case "${RUN_FINAL_BEST_FID_EVAL,,}" in
  1|true|yes|y|on)
    read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
    echo "CA-iMF training finished. Evaluating best-FID checkpoint at steps: ${FINAL_EVAL_STEP_ARRAY[*]}"
    CONFIG_MODE=caltech_finetuned_caimf_posttrain \
      PYTHON="$PYTHON" \
      USE_WANDB=False \
      WANDB_NAME_PREFIX="caltech_ft_caimf_exp${CAIMF_EXPERIMENT}_${RUN_LABEL}" \
      bash scripts/eval_best_fid_steps_plain_imf.sh "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
      --config.dataset.root="$DATASET_ROOT" \
      --config.dataset.class_mapping_root="" \
      --config.fid.cache_ref="$REPO/files/fid_stats/caltech-101-fid_stats.npz" \
      --config.fd_dino.cache_ref="$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz" \
      --config.training.final_eval_write_images=True \
      --config.logging.use_wandb=False
    ;;
  0|false|no|n|off)
    echo "Skipping final best-FID evaluation."
    ;;
  *)
    echo "ERROR: RUN_FINAL_BEST_FID_EVAL must be boolean-like, got '$RUN_FINAL_BEST_FID_EVAL'." >&2
    exit 2
    ;;
esac
