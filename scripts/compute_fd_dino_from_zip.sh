#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <dataset_zip> <output_dir> [image_size] [fd_dino_arch] [cleanup_extracted]"
    echo "Example: $0 /path/to/data.zip /path/to/output 256 vitb14 true"
    exit 1
fi

DATASET_ZIP="$1"
OUTPUT_DIR="$2"
IMAGE_SIZE="${3:-256}"
FD_DINO_ARCH="${4:-vitb14}"
CLEANUP_EXTRACTED="${5:-true}"
DATASET_BASENAME="$(basename "$DATASET_ZIP")"
DATASET_NAME="${DATASET_BASENAME%.zip}"
DEFAULT_FD_DINO_NAME="imagenet_${IMAGE_SIZE}_fd_dino_${FD_DINO_ARCH}_stats.npz"
TARGET_BASE_NAME="${DATASET_NAME%_processed}"
TARGET_BASE_NAME="${TARGET_BASE_NAME%-processed}"
TARGET_FD_DINO_NAME="${TARGET_BASE_NAME}-fd_dino-${FD_DINO_ARCH}_stats.npz"

if [ ! -f "$DATASET_ZIP" ]; then
    echo "Dataset zip not found: $DATASET_ZIP"
    exit 1
fi

WORK_ROOT="$(mktemp -d /tmp/imeanflow_fd_dino_XXXXXX)"
EXTRACT_DIR="$WORK_ROOT/extracted"
NORMALIZED_ROOT="$WORK_ROOT/normalized"
mkdir -p "$EXTRACT_DIR" "$NORMALIZED_ROOT" "$OUTPUT_DIR"

cleanup() {
    if [ "${CLEANUP_EXTRACTED}" = "true" ] || [ "${CLEANUP_EXTRACTED}" = "True" ]; then
        rm -rf "$WORK_ROOT"
    else
        echo "Keeping extracted data at: $WORK_ROOT"
    fi
}
trap cleanup EXIT

echo "Unzipping dataset to temporary directory..."
unzip -q "$DATASET_ZIP" -d "$EXTRACT_DIR"

find "$EXTRACT_DIR" -type f \
    \( -name '._*' -o -name '.DS_Store' -o -name 'Thumbs.db' \) \
    -delete

find "$EXTRACT_DIR" -type f ! \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -delete

SOURCE_ROOT="$EXTRACT_DIR"
ENTRY_COUNT="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
if [ "$ENTRY_COUNT" = "1" ]; then
    ONLY_ENTRY="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 | head -n 1)"
    if [ -d "$ONLY_ENTRY" ]; then
        SOURCE_ROOT="$ONLY_ENTRY"
    fi
fi

if [ -d "$SOURCE_ROOT/EuroSAT" ] && [ -d "$SOURCE_ROOT/EuroSATallBands" ]; then
    SOURCE_ROOT="$SOURCE_ROOT/EuroSAT"
fi

if [ -d "$SOURCE_ROOT/train" ]; then
    DATA_ROOT="$SOURCE_ROOT"
else
    mkdir -p "$NORMALIZED_ROOT/train"
    if find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
        find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -exec mv {} "$NORMALIZED_ROOT/train/" \;
    else
        mkdir -p "$NORMALIZED_ROOT/train/default"
        find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type f -exec mv {} "$NORMALIZED_ROOT/train/default/" \;
    fi
    DATA_ROOT="$NORMALIZED_ROOT"
fi

find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d -empty -delete

echo "Using dataset root: $DATA_ROOT"
echo "Saving FD-DINO stats into: $OUTPUT_DIR"

IMAGE_COUNT="$(find "$DATA_ROOT/train" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
echo "Kept $IMAGE_COUNT image files for FD-DINO stats computation"
echo "Per-class image counts:"
find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | sort | while read -r class_dir; do
    class_name="$(basename "$class_dir")"
    class_count="$(find "$class_dir" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
    echo "  $class_name: $class_count"
done

python3 prepare_dataset.py \
    --imagenet_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR" \
    --image_size="$IMAGE_SIZE" \
    --compute_latent=False \
    --compute_fid=False \
    --compute_fd_dino=True \
    --fd_dino_arch="$FD_DINO_ARCH" \
    --overwrite=True

if [ -f "$OUTPUT_DIR/$DEFAULT_FD_DINO_NAME" ]; then
    mv "$OUTPUT_DIR/$DEFAULT_FD_DINO_NAME" "$OUTPUT_DIR/$TARGET_FD_DINO_NAME"
fi

echo "FD-DINO stats written to: $OUTPUT_DIR/$TARGET_FD_DINO_NAME"
