#!/usr/bin/env bash
set -euo pipefail

# Stanford-Cars evaluation - DiT→SiT DDPM-v
# Phase 1A: Multi-domain evaluation

CHECKPOINT_DIR="/home/ens/AT74470/imeanflow/files/logs/finetuning/caltech_plain_SiT_DiTinit_taylor_ddpmv_20260606_204610_q2i8jv"
REPO_ROOT="$HOME/imeanflow"
PYTHON="${REPO_ROOT}/.venv/bin/python"

DATASET="stanford-cars"
export CUDA_VISIBLE_DEVICES=0,1

# Step 1: Quick 4-step validation
echo "=========================================="
echo "Step 1: Quick validation with 4 steps"
echo "=========================================="

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
  echo "Quick validation results:"
  cat "$EVAL_WORKDIR/eval_metrics.csv"
  echo ""
fi

# Step 2: Full evaluation sweep
echo "=========================================="
echo "Step 2: Full evaluation sweep"
echo "=========================================="

STEPS=(1 2 4 8 16 32 250)
for NUM_STEPS in "${STEPS[@]}"; do
  EVAL_WORKDIR="${CHECKPOINT_DIR}/eval_multidomain_${DATASET}_${NUM_STEPS}steps"

  # Skip if already done in step 1
  if [[ -f "$EVAL_WORKDIR/eval_metrics.csv" ]]; then
    echo "Skipping ${NUM_STEPS} steps (already evaluated)"
    continue
  fi

  mkdir -p "$EVAL_WORKDIR"
  echo "Evaluating ${DATASET} with ${NUM_STEPS} steps..."

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

echo "=========================================="
echo "Stanford-Cars evaluation complete!"
echo "Results in: ${CHECKPOINT_DIR}/eval_multidomain_stanford-cars_*steps/"
echo "=========================================="
