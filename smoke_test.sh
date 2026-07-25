#!/bin/bash
# Quick smoke test - CUB-200 (Birds) DDPM-v on GPU 0

cd /opt/dlami/nvme/meanflow/imeanflow
source .venv/bin/activate

echo "=========================================="
echo "Running smoke test: CUB-200 (Birds) DDPM-v"
echo "GPU: 0"
echo "This will train for a few steps to verify everything works"
echo "=========================================="

# Run with minimal training (you can ctrl+C after a few minutes once you see it's working)
CUDA_VISIBLE_DEVICES=0 USE_WANDB=False bash scripts/train_cub_ddpmv.sh

echo "=========================================="
echo "Smoke test complete!"
echo "=========================================="
