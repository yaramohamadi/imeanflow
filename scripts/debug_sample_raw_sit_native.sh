#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/debug_sample_raw_sit_native.sh <output_dir> [extra python args...]

Example:
  bash scripts/debug_sample_raw_sit_native.sh files/debug/raw_sit_preview \
    --num-steps 250 100 50 25 10 4 1 \
    --num-samples 16 \
    --cfg-scale 4.0

This script:
  1) loads the raw SiT-XL-2-256 PyTorch checkpoint in the original torch SiT model
  2) runs a native-style Euler CFG latent sampler at the requested NFEs
  3) decodes and saves preview grids without touching the Flax training path
EOF
  exit 1
fi

OUTPUT_DIR="$1"
shift

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/debug_sample_raw_sit_native.py \
  --output-dir="$OUTPUT_DIR" \
  "$@"
