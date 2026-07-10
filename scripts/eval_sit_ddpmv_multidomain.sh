#!/usr/bin/env bash
set -euo pipefail

# Multi-domain evaluation for DiT→SiT DDPM-v model
# Phase 1A of MeFT experiment plan

CHECKPOINT_DIR="/home/ens/AT74470/imeanflow/files/logs/finetuning/caltech_plain_SiT_DiTinit_taylor_ddpmv_20260606_204610_q2i8jv"
REPO_ROOT="$HOME/imeanflow"
PYTHON="${REPO_ROOT}/.venv/bin/python"

# Domains to evaluate (excluding caltech which is already done)
DOMAINS=("artbench-10" "cub-200-2011" "food-101" "stanford-cars")
NUM_STEPS=16  # Same as original Caltech evaluation

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1

for DATASET in "${DOMAINS[@]}"; do
  echo "=========================================="
  echo "Evaluating on dataset: ${DATASET}"
  echo "=========================================="

  EVAL_WORKDIR="${CHECKPOINT_DIR}/eval_multidomain_${DATASET}_${NUM_STEPS}steps"
  mkdir -p "$EVAL_WORKDIR"

  # Run evaluation
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

  # Extract metrics
  if [[ -f "$EVAL_WORKDIR/eval_metrics.csv" ]]; then
    echo "Results for ${DATASET}:"
    cat "$EVAL_WORKDIR/eval_metrics.csv"
    echo ""
  fi
done

echo "=========================================="
echo "All evaluations complete!"
echo "Results saved in: ${CHECKPOINT_DIR}/eval_multidomain_*"
echo "=========================================="
