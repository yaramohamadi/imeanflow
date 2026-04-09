#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/eval_and_sample_best_fid_sit_dmf_1step_cfg75.sh <run_dir_or_best_fid_dir>

Example:
  bash scripts/eval_and_sample_best_fid_sit_dmf_1step_cfg75.sh \
    files/logs/finetuning/caltech_SiT_DMF_no_train_guidance

This script:
  1) resolves the single checkpoint stored under <run_dir>/best_fid (or uses the provided best_fid dir)
  2) runs eval_only with the SiT-DMF config at 1 sampling step and omega=7.5
  3) writes a 1-step preview grid from the same checkpoint
EOF
  exit 1
fi

BEST_FID_ROOT="$1"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-True}"
EVAL_PLATFORM="${EVAL_PLATFORM:-gpu}"
SAMPLING_OMEGA="${SAMPLING_OMEGA:-7.5}"
NUM_STEPS="${NUM_STEPS:-1}"

case "$EVAL_PLATFORM" in
  gpu|cpu)
    ;;
  *)
    echo "ERROR: EVAL_PLATFORM must be 'gpu' or 'cpu', got: $EVAL_PLATFORM" >&2
    exit 2
    ;;
esac

if [[ -d "$BEST_FID_ROOT" && $(basename "$BEST_FID_ROOT") != "best_fid" ]]; then
  BEST_FID_DIR="$BEST_FID_ROOT/best_fid"
else
  BEST_FID_DIR="$BEST_FID_ROOT"
fi

if [[ ! -d "$BEST_FID_DIR" ]]; then
  echo "ERROR: best_fid directory not found: $BEST_FID_DIR" >&2
  exit 3
fi

mapfile -t CKPTS < <(find "$BEST_FID_DIR" -maxdepth 1 -type d -name 'checkpoint_*' | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoint_*/ directory found under $BEST_FID_DIR" >&2
  exit 4
fi
if [[ ${#CKPTS[@]} -gt 1 ]]; then
  echo "ERROR: more than one checkpoint_*/ directory found under $BEST_FID_DIR:" >&2
  printf '  %s\n' "${CKPTS[@]}" >&2
  exit 5
fi

CHECKPOINT_DIR="${CKPTS[0]}"
RUN_ROOT="$(dirname "$BEST_FID_DIR")"
EVAL_WORKDIR="$RUN_ROOT/eval_best_fid_1step_cfg75"
PREVIEW_DIR="$EVAL_WORKDIR/preview_1step_cfg75"
mkdir -p "$EVAL_WORKDIR"

echo "Found checkpoint: $CHECKPOINT_DIR"
echo "Eval workdir: $EVAL_WORKDIR"
echo "Preview dir: $PREVIEW_DIR"
echo "Config mode: $CONFIG_MODE"
echo "Sampling omega: $SAMPLING_OMEGA"
echo "Num steps: $NUM_STEPS"

if [[ "$EVAL_PLATFORM" == "cpu" ]]; then
  JAX_PLATFORM_NAME=cpu \
    JAX_NUM_DEVICES=1 \
    TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
    PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
    "$PYTHON" \
      main.py \
      --workdir="$EVAL_WORKDIR" \
      --config=configs/load_config.py:${CONFIG_MODE} \
      --config.eval_only=True \
      --config.training.use_ema=False \
      --config.load_from="$CHECKPOINT_DIR" \
      --config.sampling.num_steps=${NUM_STEPS} \
      --config.sampling.omega=${SAMPLING_OMEGA} \
      --config.logging.use_wandb=${USE_WANDB} \
      2>&1 | tee -a "$EVAL_WORKDIR/output.log"
else
  TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
    XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
    XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} \
    PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
    "$PYTHON" \
      main.py \
      --workdir="$EVAL_WORKDIR" \
      --config=configs/load_config.py:${CONFIG_MODE} \
      --config.eval_only=True \
      --config.training.use_ema=False \
      --config.load_from="$CHECKPOINT_DIR" \
      --config.sampling.num_steps=${NUM_STEPS} \
      --config.sampling.omega=${SAMPLING_OMEGA} \
      --config.logging.use_wandb=${USE_WANDB} \
      2>&1 | tee -a "$EVAL_WORKDIR/output.log"
fi

"$PYTHON" scripts/debug_sample_converted_path.py \
  --checkpoint="$CHECKPOINT_DIR" \
  --output-dir="$PREVIEW_DIR" \
  --config-mode="$CONFIG_MODE" \
  --sampling-family=meanflow_average_velocity \
  --omega="${SAMPLING_OMEGA}" \
  --num-steps "${NUM_STEPS}" \
  --decode-style=latent_manager \
  2>&1 | tee -a "$EVAL_WORKDIR/output.log"

echo "Finished eval + preview for $CHECKPOINT_DIR"
