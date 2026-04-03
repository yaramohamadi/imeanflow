#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: bash scripts/debug_sample_converted_average_velocity.sh <checkpoint_or_pt> <output_dir> [extra python args...]

Example:
  bash scripts/debug_sample_converted_average_velocity.sh \
    /home/ens/AT74470/imeanflow/files/weights/SiT-XL-2-256.pt \
    files/debug/converted_dmf_average_velocity_preview \
    --config-mode caltech_sit_dmf_finetune \
    --label-space imagenet1000 \
    --omega 4.0 \
    --num-steps 250 100 50 25 10 4 1

This script:
  1) loads either a raw .pt checkpoint or a Flax checkpoint into the converted Flax path
  2) samples with interval-conditioned MeanFlow average velocity using t_i and r_i from the denoising grid
  3) decodes and saves preview grids for the requested NFEs
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
  --sampling-family=meanflow_average_velocity \
  "$@"
