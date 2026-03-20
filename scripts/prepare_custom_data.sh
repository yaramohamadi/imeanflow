#!/bin/bash

set -euo pipefail

# Configuration for custom class-folder dataset preparation.
# INPUT_ROOT should point to a dataset laid out like:
#   INPUT_ROOT/class_a/*.jpg
#   INPUT_ROOT/class_b/*.jpg
# The script wraps it into a temporary train/ directory so the existing
# latent-prep pipeline can be reused unchanged.

export INPUT_ROOT="${INPUT_ROOT:-YOUR_INPUT_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:-YOUR_OUTPUT_DIR}"
export LOG_ROOT="${LOG_ROOT:-YOUR_LOG_DIR}"

if [ "$INPUT_ROOT" = "YOUR_INPUT_ROOT" ] || [ "$OUTPUT_DIR" = "YOUR_OUTPUT_DIR" ] || [ "$LOG_ROOT" = "YOUR_LOG_DIR" ]; then
    echo "ERROR: Set INPUT_ROOT, OUTPUT_DIR, and LOG_ROOT before running."
    exit 1
fi

export BATCH_SIZE="${BATCH_SIZE:-128}"
export VAE_TYPE="${VAE_TYPE:-mse}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export COMPUTE_LATENT="${COMPUTE_LATENT:-True}"
export COMPUTE_FID="${COMPUTE_FID:-False}"
export OVERWRITE="${OVERWRITE:-False}"

if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

now=$(date '+%Y%m%d_%H%M%S')
salt=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
job_name=${1:-custom_data}
export JOBNAME="prepare_${job_name}_${now}_${salt}"
export LOG_DIR="$LOG_ROOT/$USER/$JOBNAME"

mkdir -p "$LOG_DIR"

tmp_root="/tmp/${JOBNAME}_wrapper"
rm -rf "$tmp_root"
mkdir -p "$tmp_root"
ln -s "$INPUT_ROOT" "$tmp_root/train"

cleanup() {
    rm -rf "$tmp_root"
}
trap cleanup EXIT

latent_size=$((IMAGE_SIZE / 8))

echo "=============================================="
echo "Custom Data Preparation Configuration"
echo "=============================================="
echo "Input Root: $INPUT_ROOT"
echo "Wrapped Root: $tmp_root"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "VAE Type: $VAE_TYPE"
echo "Image Size: $IMAGE_SIZE -> Latent Size: ${latent_size}x${latent_size}"
echo "Compute Latent: $COMPUTE_LATENT"
echo "Compute FID: $COMPUTE_FID"
echo "Overwrite: $OVERWRITE"
echo "=============================================="

"$PYTHON_BIN" prepare_dataset.py \
    --imagenet_root="$tmp_root" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size="$BATCH_SIZE" \
    --vae_type="$VAE_TYPE" \
    --image_size="$IMAGE_SIZE" \
    --compute_latent="$COMPUTE_LATENT" \
    --compute_fid="$COMPUTE_FID" \
    --overwrite="$OVERWRITE" \
    2>&1 | tee -a "$LOG_DIR/output.log"

echo "=============================================="
echo "Data preparation completed!"
echo "Check logs at: $LOG_DIR/output.log"
echo "Latent dataset saved to: $OUTPUT_DIR"
echo "=============================================="
