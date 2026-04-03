#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: bash scripts/debug_sample_converted_path.sh <checkpoint_or_pt> <output_dir> [extra python args...]

Example:
  bash scripts/debug_sample_converted_path.sh \
    /home/ens/AT74470/imeanflow/files/weights/SiT-XL-2-256.pt \
    files/debug/converted_dmf_preview \
    --config-mode caltech_sit_dmf_finetune \
    --num-steps 250 100 50 25 10 4 1

This script:
  1) loads either a raw .pt checkpoint or a Flax checkpoint into the converted Flax path
  2) runs lightweight preview sampling without FID or DINO
  3) decodes and saves image grids for the requested NFEs
EOF
  exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"
shift 2

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/debug_sample_converted_path.py \
  --checkpoint="$CHECKPOINT_PATH" \
  --output-dir="$OUTPUT_DIR" \
  "$@"
