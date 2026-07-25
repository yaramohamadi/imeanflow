#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  DATASET_NAME=caltech101 TARGET_IMF_CHECKPOINT=/path/to/best_fid/checkpoint_N \
    bash scripts/train_finetuned_afm.sh <dataset_name> <run_label> [extra config overrides]

Known datasets: artbench10, caltech101, cub200, food101, stanfordcars

AFM_ABLATION values:
  A|target_imf  target iMF loss only
  B|afm_only    adversarial endpoint AFM only
  C|imf_afm     iMF + endpoint AFM
  D|afm_anchor  endpoint AFM + decaying starting-checkpoint anchor
EOF
  exit 1
fi

DATASET_NAME="$1"
RUN_LABEL="$2"
shift 2

REPO="${REPO:-/home/zahradt/links/projects/def-hadi87/zahradt/imeanflow}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
TARGET_IMF_CHECKPOINT="${TARGET_IMF_CHECKPOINT:-${IMF_CHECKPOINT:-}}"
AFM_ABLATION="${AFM_ABLATION:-B}"
LOG_ROOT="${LOG_ROOT:-$REPO/files/logs/afm_finetuned}"
USE_WANDB="${USE_WANDB:-False}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"

case "${DATASET_NAME}" in
  artbench10|artbench-10)
    DATASET_SLUG="artbench10"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/zahradt/datasets/artbench-10_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-$REPO/files/fid_stats/artbench-10_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$REPO/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
    ;;
  caltech101|caltech-101)
    DATASET_SLUG="caltech101"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/zahradt/datasets/caltech-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-$REPO/files/fid_stats/caltech-101-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
    ;;
  cub200|cub-200|cub-200-2011)
    DATASET_SLUG="cub200"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/zahradt/datasets/cub-200-2011_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-$REPO/files/fid_stats/cub-200-2011_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$REPO/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
    ;;
  food101|food-101)
    DATASET_SLUG="food101"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/zahradt/datasets/food-101_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-$REPO/files/fid_stats/food-101_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$REPO/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
    ;;
  stanfordcars|stanford-cars|cars)
    DATASET_SLUG="stanfordcars"
    DATASET_ROOT="${DATASET_ROOT:-/scratch/zahradt/datasets/stanford-cars_processed_latents}"
    FID_CACHE_REF="${FID_CACHE_REF:-$REPO/files/fid_stats/stanford_cars_processed-fid_stats.npz}"
    FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$REPO/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
    ;;
  *)
    echo "ERROR: unknown dataset '$DATASET_NAME'." >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$TARGET_IMF_CHECKPOINT" || ! -e "$TARGET_IMF_CHECKPOINT" ]]; then
  echo "ERROR: TARGET_IMF_CHECKPOINT/IMF_CHECKPOINT must point to the fine-tuned iMF checkpoint." >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/train" ]]; then
  echo "ERROR: latent train directory missing: $DATASET_ROOT/train" >&2
  exit 2
fi
for path in "$FID_CACHE_REF" "$FD_DINO_CACHE_REF"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing stats file: $path" >&2
    exit 2
  fi
done

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
    echo "ERROR: AFM_ABLATION must be A, B, C, D or a named equivalent." >&2
    exit 2
    ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-$LOG_ROOT/${DATASET_SLUG}_${ABLATION}_${RUN_LABEL}_${STAMP}}"
mkdir -p "$WORKDIR"

echo "AFM workdir: $WORKDIR"
echo "Dataset: $DATASET_SLUG"
echo "Dataset root: $DATASET_ROOT"
echo "Starting checkpoint: $TARGET_IMF_CHECKPOINT"
echo "FID stats: $FID_CACHE_REF"
echo "FD-DINO stats: $FD_DINO_CACHE_REF"
echo "Ablation: $ABLATION"
echo "lambda_imf=$LAMBDA_IMF lambda_adv=$LAMBDA_ADV lambda_ot=$LAMBDA_OT lambda_anchor=$LAMBDA_ANCHOR"
echo "RUN_FINAL_BEST_FID_EVAL: $RUN_FINAL_BEST_FID_EVAL"
echo "FINAL_EVAL_STEPS: $FINAL_EVAL_STEPS"

cd "$REPO"
COMMON_CONFIG_ARGS=(
  --config.load_from="$TARGET_IMF_CHECKPOINT"
  --config.dataset.root="$DATASET_ROOT"
  --config.dataset.class_mapping_root=""
  --config.dataset.num_classes_from_data=True
  --config.fid.cache_ref="$FID_CACHE_REF"
  --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF"
  --config.afm.ablation="$ABLATION"
  --config.afm.lambda_imf="$LAMBDA_IMF"
  --config.afm.lambda_adv="$LAMBDA_ADV"
  --config.afm.lambda_ot="$LAMBDA_OT"
  --config.afm.lambda_anchor="$LAMBDA_ANCHOR"
  --config.afm.discriminator_updates="$DISCRIMINATOR_UPDATES"
  --config.logging.use_wandb="$USE_WANDB"
  --config.logging.wandb_project="${WANDB_PROJECT:-afm_finetuned_imf}"
  --config.logging.wandb_name="${WANDB_NAME:-${DATASET_SLUG}_afm_${ABLATION}_${RUN_LABEL}}"
)

"$PYTHON" main_afm.py \
  --workdir="$WORKDIR" \
  --config=configs/load_config.py:finetuned_afm_posttrain \
  "${COMMON_CONFIG_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$WORKDIR/output.log"

case "${RUN_FINAL_BEST_FID_EVAL,,}" in
  1|true|yes|y|on)
    read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
    echo "AFM training finished. Evaluating best-FID checkpoint at steps: ${FINAL_EVAL_STEP_ARRAY[*]}"
    CONFIG_MODE=finetuned_afm_posttrain \
      PYTHON="$PYTHON" \
      USE_WANDB="$FINAL_EVAL_USE_WANDB" \
      WANDB_NAME_PREFIX="${DATASET_SLUG}_afm_${ABLATION}_${RUN_LABEL}" \
      bash scripts/eval_best_fid_steps_plain_imf.sh "$WORKDIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
      "${COMMON_CONFIG_ARGS[@]}" \
      "$@"
    ;;
  0|false|no|n|off)
    echo "Skipping final best-FID evaluation."
    ;;
  *)
    echo "ERROR: RUN_FINAL_BEST_FID_EVAL must be boolean-like, got '$RUN_FINAL_BEST_FID_EVAL'." >&2
    exit 2
    ;;
esac
