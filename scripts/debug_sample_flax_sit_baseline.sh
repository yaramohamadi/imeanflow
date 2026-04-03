#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/debug_sample_flax_sit_baseline.sh <output_dir> [extra python args...]

Example:
  bash scripts/debug_sample_flax_sit_baseline.sh files/debug/flax_sit_baseline_preview \
    --num-steps 250 100 50 25 10 4 2 \
    --num-samples 16 \
    --cfg-scale 4.0

This script:
  1) loads the raw SiT-XL-2-256 checkpoint into the exact Flax SiT baseline
  2) runs fixed-step SiT-style velocity sampling with JAX/Flax
  3) decodes and saves preview grids using the official SiT VAE convention
EOF
  exit 1
fi

OUTPUT_DIR="$1"
shift

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/debug_sample_flax_sit_baseline.py \
  --output-dir="$OUTPUT_DIR" \
  "$@"
