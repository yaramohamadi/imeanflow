#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: BACKBONE=sit bash scripts/sweep_plain_finetune_datasets.sh <sweep_label> [extra config overrides...]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=sit bash scripts/sweep_plain_finetune_datasets.sh baseline

  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=dit DATASETS="caltech101 food101" bash scripts/sweep_plain_finetune_datasets.sh ddpm_baseline \
    --config.training.max_train_steps=10000

Env knobs:
  BACKBONE=sit                       # sit or dit
  DATASETS="caltech101 artbench10 cub200 food101 stanfordcars"
  SWEEP_LOG_DIR=files/logs/sweeps/... # default: files/logs/sweeps/plain_<backbone>_<label>_<time>
  WANDB_PROJECT=plain_sit_finetune    # default depends on BACKBONE
  CONTINUE_ON_FAILURE=False           # set True to continue after a dataset fails

All extra args are forwarded to the underlying train script.
EOF
  exit 1
fi

SWEEP_LABEL="$1"
shift
EXTRA_ARGS=("$@")

BACKBONE="${BACKBONE:-sit}"
DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-False}"
NOW=$(date '+%Y%m%d_%H%M%S')

case "${BACKBONE,,}" in
  sit)
    TRAIN_SCRIPT="scripts/train_plain_sit_finetune.sh"
    BACKBONE_LABEL="sit"
    DEFAULT_WANDB_PROJECT="plain_sit_finetune"
    ;;
  dit)
    TRAIN_SCRIPT="scripts/train_plain_dit_finetune.sh"
    BACKBONE_LABEL="dit"
    DEFAULT_WANDB_PROJECT="plain_dit_finetune"
    ;;
  *)
    echo "ERROR: BACKBONE must be sit or dit, got '${BACKBONE}'." >&2
    exit 2
    ;;
esac

WANDB_PROJECT="${WANDB_PROJECT:-$DEFAULT_WANDB_PROJECT}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-files/logs/sweeps/plain_${BACKBONE_LABEL}_${SWEEP_LABEL}_${NOW}}"
mkdir -p "$SWEEP_LOG_DIR"

MANIFEST="$SWEEP_LOG_DIR/sweep_manifest.tsv"
printf "dataset\tbackbone\trun_label\twandb_project\twandb_name\tstatus\n" > "$MANIFEST"

cat <<EOF
Sweep label: $SWEEP_LABEL
Backbone: $BACKBONE_LABEL
Datasets: $DATASETS
Train script: $TRAIN_SCRIPT
Sweep log dir: $SWEEP_LOG_DIR
Wandb project: $WANDB_PROJECT
Extra args: ${EXTRA_ARGS[*]:-<none>}
EOF

for DATASET in $DATASETS; do
  RUN_LABEL="${SWEEP_LABEL}_${DATASET}"
  WANDB_NAME="${DATASET}_plain_${BACKBONE_LABEL}_${SWEEP_LABEL}"

  printf "\n=== Running %s on %s ===\n" "$BACKBONE_LABEL" "$DATASET"
  printf "%s\t%s\t%s\t%s\t%s\tstarted\n" \
    "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" >> "$MANIFEST"

  set +e
  DATASET_NAME="$DATASET" \
    LOG_DIR="$SWEEP_LOG_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$WANDB_NAME" \
    bash "$TRAIN_SCRIPT" "$RUN_LABEL" "${EXTRA_ARGS[@]}"
  STATUS=$?
  set -e

  if [[ "$STATUS" -eq 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tdone\n" \
      "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" >> "$MANIFEST"
  else
    printf "%s\t%s\t%s\t%s\t%s\tfailed_%s\n" \
      "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" "$STATUS" >> "$MANIFEST"
    case "${CONTINUE_ON_FAILURE,,}" in
      1|true|yes|y|on)
        echo "WARNING: dataset '$DATASET' failed with status $STATUS; continuing." >&2
        ;;
      *)
        echo "ERROR: dataset '$DATASET' failed with status $STATUS. Manifest: $MANIFEST" >&2
        exit "$STATUS"
        ;;
    esac
  fi
done

echo "Sweep complete. Manifest: $MANIFEST"
