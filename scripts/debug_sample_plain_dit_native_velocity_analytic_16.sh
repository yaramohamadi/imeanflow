#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_PATH="${1:-$REPO_ROOT/files/weights/DiT-XL-2-256x256.pt}"
DATASET_ROOT="${DATASET_ROOT:-/projets/Ymohammadi/datasets/caltech-101_processed_latents}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_IMAGES="${NUM_IMAGES:-16}"
OMEGA="${OMEGA:-1.5}"
SEED="${SEED:-99}"
WORKDIR="${WORKDIR:-files/debug/plain_dit_native_velocity_16_imagenet1000_cfg_in_velocity_analytic_sigma1e5}"

if [[ ! -e "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint path does not exist: $CHECKPOINT_PATH" >&2
  exit 2
fi

python scripts/debug_sample_plain_dit_native_velocity.py \
  ${CHECKPOINT_PATH:+$CHECKPOINT_PATH} \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --num-images "$NUM_IMAGES" \
  --num-steps 16 \
  --method native_velocity \
  --native-velocity-cfg-space velocity \
  --native-velocity-derivative-mode analytic \
  --native-velocity-sigma-clamp 1e-5 \
  --dataset-root "$DATASET_ROOT" \
  --label-space imagenet1000 \
  --omega "$OMEGA" \
  --seed "$SEED" \
  --workdir "$WORKDIR"
