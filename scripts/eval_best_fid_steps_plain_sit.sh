#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=plain_sit_finetune bash scripts/eval_best_fid_steps_plain_sit.sh <best_fid_dir_or_run_dir> [num_steps...]

Example:
  CONFIG_MODE=plain_sit_finetune bash scripts/eval_best_fid_steps_plain_sit.sh \
    files/logs/finetuning/plain_SiT_finetune_caltech101_baseline_20260418_123456_ab12cd 1 2 4

If the first argument points to the run directory, the script assumes the checkpoint is under <run_dir>/best_fid.
If the first argument points to the best_fid directory itself, it uses that directly.
EOF
  exit 1
fi

BEST_FID_ROOT="$1"
shift
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(1 2 4)
fi

CONFIG_MODE="${CONFIG_MODE:-plain_sit_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-True}"

if [[ -d "$BEST_FID_ROOT" && $(basename "$BEST_FID_ROOT") != "best_fid" ]]; then
  BEST_FID_DIR="$BEST_FID_ROOT/best_fid"
else
  BEST_FID_DIR="$BEST_FID_ROOT"
fi

if [[ ! -d "$BEST_FID_DIR" ]]; then
  echo "ERROR: best_fid directory not found: $BEST_FID_DIR" >&2
  exit 2
fi

mapfile -t CKPTS < <(find "$BEST_FID_DIR" -maxdepth 1 -type d -name 'checkpoint_*' | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoint_*/ directory found under $BEST_FID_DIR" >&2
  exit 3
fi
if [[ ${#CKPTS[@]} -gt 1 ]]; then
  echo "ERROR: more than one checkpoint_*/ directory found under $BEST_FID_DIR:" >&2
  printf '  %s\n' "${CKPTS[@]}" >&2
  exit 4
fi

CHECKPOINT_DIR="${CKPTS[0]}"
echo "Found checkpoint: $CHECKPOINT_DIR"

for NUM_STEPS in "${STEPS[@]}"; do
  if ! [[ "$NUM_STEPS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid num_steps value: $NUM_STEPS" >&2
    exit 5
  fi

  EVAL_WORKDIR="$(dirname "$BEST_FID_DIR")/eval_best_fid_${NUM_STEPS}steps"
  mkdir -p "$EVAL_WORKDIR"

  echo "=== Evaluating checkpoint with num_steps=$NUM_STEPS ==="
  echo "workdir: $EVAL_WORKDIR"
  echo "config mode: $CONFIG_MODE"

  TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
    XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
    XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} \
    PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
    $PYTHON \
      main_sit.py \
      --workdir="$EVAL_WORKDIR" \
      --config=configs/load_config.py:${CONFIG_MODE} \
      --config.eval_only=True \
      --config.partial_load=False \
      --config.load_from="$CHECKPOINT_DIR" \
      --config.sampling.num_steps=${NUM_STEPS} \
      --config.logging.use_wandb=${USE_WANDB} \
      2>&1 | tee -a "$EVAL_WORKDIR/output.log"
done
