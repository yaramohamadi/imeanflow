#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: TARGET_IMF_CHECKPOINT=/path/to/caltech/best_fid bash scripts/train_caltech_afm_posttrain.sh <run_label> [extra config overrides]

AFM_ABLATION values:
  A|target_imf  target iMF loss only
  B|afm_only    adversarial endpoint AFM only (principal method)
  C|imf_afm     iMF + endpoint AFM
  D|afm_anchor  endpoint AFM + decaying starting-checkpoint anchor
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
REPO=/home/ens/Zdehghani/imeanflow
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
TARGET_IMF_CHECKPOINT="${TARGET_IMF_CHECKPOINT:-}"
DATASET_ROOT="${DATASET_ROOT:-/home/ens/Zdehghani/datasets/caltech-101_processed_latents}"
CLASS_MAPPING_ROOT="${CLASS_MAPPING_ROOT:-/home/ens/Zdehghani/datasets/caltech-101_images/train}"
AFM_ABLATION="${AFM_ABLATION:-B}"
LOG_ROOT="${LOG_ROOT:-$REPO/files/logs/afm}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$TARGET_IMF_CHECKPOINT" || ! -e "$TARGET_IMF_CHECKPOINT" ]]; then
  echo "ERROR: TARGET_IMF_CHECKPOINT must point to the completed Caltech iMF checkpoint." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: target latent directory missing: $DATASET_ROOT/train" >&2
  exit 2
fi

case "${AFM_ABLATION,,}" in
  a|target_imf)
    ABLATION=target_imf
    LAMBDA_IMF=1.0
    LAMBDA_ADV=0.0
    LAMBDA_OT=0.0
    LAMBDA_ANCHOR=0.0
    DISCRIMINATOR_UPDATES=False
    ;;
  b|afm_only)
    ABLATION=afm_only
    LAMBDA_IMF=0.0
    LAMBDA_ADV=1.0
    LAMBDA_OT="${LAMBDA_OT:-0.0}"
    LAMBDA_ANCHOR=0.0
    DISCRIMINATOR_UPDATES=True
    ;;
  c|imf_afm)
    ABLATION=imf_afm
    LAMBDA_IMF=1.0
    LAMBDA_ADV=1.0
    LAMBDA_OT="${LAMBDA_OT:-0.0}"
    LAMBDA_ANCHOR=0.0
    DISCRIMINATOR_UPDATES=True
    ;;
  d|afm_anchor)
    ABLATION=afm_anchor
    LAMBDA_IMF=0.0
    LAMBDA_ADV=1.0
    LAMBDA_OT="${LAMBDA_OT:-0.0}"
    LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-0.1}"
    DISCRIMINATOR_UPDATES=True
    ;;
  *)
    echo "ERROR: AFM_ABLATION must be A, B, C, D or its named equivalent." >&2
    exit 2
    ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-$LOG_ROOT/caltech_${ABLATION}_${RUN_LABEL}_${STAMP}}"
mkdir -p "$WORKDIR"

echo "AFM workdir: $WORKDIR"
echo "Starting target iMF checkpoint: $TARGET_IMF_CHECKPOINT"
echo "Ablation: $ABLATION"
echo "lambda_imf=$LAMBDA_IMF lambda_adv=$LAMBDA_ADV lambda_ot=$LAMBDA_OT lambda_anchor=$LAMBDA_ANCHOR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

"$PYTHON" main_afm.py \
  --workdir="$WORKDIR" \
  --config=configs/load_config.py:caltech_afm_posttrain \
  --config.load_from="$TARGET_IMF_CHECKPOINT" \
  --config.dataset.root="$DATASET_ROOT" \
  --config.dataset.class_mapping_root="$CLASS_MAPPING_ROOT" \
  --config.afm.ablation="$ABLATION" \
  --config.afm.lambda_imf="$LAMBDA_IMF" \
  --config.afm.lambda_adv="$LAMBDA_ADV" \
  --config.afm.lambda_ot="$LAMBDA_OT" \
  --config.afm.lambda_anchor="$LAMBDA_ANCHOR" \
  --config.afm.discriminator_updates="$DISCRIMINATOR_UPDATES" \
  "$@" \
  2>&1 | tee -a "$WORKDIR/output.log"

case "${RUN_FINAL_BEST_FID_EVAL,,}" in
  1|true|yes|y|on)
    read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
    echo "AFM training finished. Evaluating best-FID checkpoint at steps: ${FINAL_EVAL_STEP_ARRAY[*]}"
    CONFIG_MODE=caltech_afm_posttrain \
      PYTHON="$PYTHON" \
      USE_WANDB=False \
      WANDB_NAME_PREFIX="caltech_afm_${ABLATION}_${RUN_LABEL}" \
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
