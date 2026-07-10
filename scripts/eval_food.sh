#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Food-101 evaluation - DiT→SiT DDPM-v
# SET YOUR GPUs HERE:
export CUDA_VISIBLE_DEVICES=0,1
# =============================================================================

CHECKPOINT_DIR="/home/ens/AT74470/imeanflow/files/logs/finetuning/caltech_plain_SiT_DiTinit_taylor_ddpmv_20260606_204610_q2i8jv"
REPO_ROOT="$HOME/imeanflow"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATASET="food-101"

echo "=========================================="
echo "Evaluating: ${DATASET}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Step 1: Quick 4-step validation
echo "Step 1: Quick 4-step validation..."
EVAL_WORKDIR="${CHECKPOINT_DIR}/eval_multidomain_${DATASET}_4steps"
mkdir -p "$EVAL_WORKDIR"

TF_CPP_MIN_LOG_LEVEL=3 \
  XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONWARNINGS=ignore \
  "$PYTHON" \
    "${REPO_ROOT}/main_sit.py" \
    --workdir="$EVAL_WORKDIR" \
    --config="${REPO_ROOT}/configs/load_config.py:plain_sit_finetune" \
    --config.eval_only=True \
    --config.partial_load=False \
    --config.load_from="$CHECKPOINT_DIR/best_fid/checkpoint_"* \
    --config.sampling.num_steps=4 \
    --config.dataset.root="${HOME}/datasets/${DATASET}_processed_latents" \
    --config.dataset.name="${DATASET}" \
    --config.logging.use_wandb=False \
    2>&1 | tee "$EVAL_WORKDIR/output.log"

if [[ -f "$EVAL_WORKDIR/eval_metrics.csv" ]]; then
  echo "Quick validation done:"
  cat "$EVAL_WORKDIR/eval_metrics.csv"
fi

# Step 2: Full evaluation - 1, 2, 4, 250 steps
echo ""
echo "Step 2: Full evaluation sweep (1, 2, 4, 250 steps)..."
STEPS=(1 2 4 250)

for NUM_STEPS in "${STEPS[@]}"; do
  EVAL_WORKDIR="${CHECKPOINT_DIR}/eval_multidomain_${DATASET}_${NUM_STEPS}steps"

  if [[ -f "$EVAL_WORKDIR/eval_metrics.csv" ]]; then
    echo "  ${NUM_STEPS} steps: already done, skipping"
    continue
  fi

  mkdir -p "$EVAL_WORKDIR"
  echo "  Evaluating ${NUM_STEPS} steps..."

  TF_CPP_MIN_LOG_LEVEL=3 \
    XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONWARNINGS=ignore \
    "$PYTHON" \
      "${REPO_ROOT}/main_sit.py" \
      --workdir="$EVAL_WORKDIR" \
      --config="${REPO_ROOT}/configs/load_config.py:plain_sit_finetune" \
      --config.eval_only=True \
      --config.partial_load=False \
      --config.load_from="$CHECKPOINT_DIR/best_fid/checkpoint_"* \
      --config.sampling.num_steps=${NUM_STEPS} \
      --config.dataset.root="${HOME}/datasets/${DATASET}_processed_latents" \
      --config.dataset.name="${DATASET}" \
      --config.logging.use_wandb=False \
      2>&1 | tee "$EVAL_WORKDIR/output.log"
done

echo ""
echo "=========================================="
echo "Food-101 complete!"
echo "Results: ${CHECKPOINT_DIR}/eval_multidomain_food-101_*/"
echo "=========================================="
