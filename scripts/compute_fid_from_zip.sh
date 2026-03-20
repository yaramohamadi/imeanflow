#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <dataset_zip> <output_dir> [image_size] [cleanup_extracted]"
    echo "Example: $0 /path/to/data.zip /path/to/output 256 true"
    exit 1
fi

DATASET_ZIP="$1"
OUTPUT_DIR="$2"
IMAGE_SIZE="${3:-256}"
CLEANUP_EXTRACTED="${4:-true}"
DATASET_BASENAME="$(basename "$DATASET_ZIP")"
DATASET_NAME="${DATASET_BASENAME%.zip}"
DEFAULT_FID_NAME="imagenet_${IMAGE_SIZE}_fid_stats.npz"
TARGET_BASE_NAME="${DATASET_NAME%_processed}"
TARGET_BASE_NAME="${TARGET_BASE_NAME%-processed}"
TARGET_FID_NAME="${TARGET_BASE_NAME}-fid_stats.npz"

if [ ! -f "$DATASET_ZIP" ]; then
    echo "Dataset zip not found: $DATASET_ZIP"
    exit 1
fi

WORK_ROOT="$(mktemp -d /tmp/imeanflow_fid_XXXXXX)"
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

# Remove common archive metadata files and anything that's not a standard RGB image.
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

# Special-case EuroSAT archives that contain both RGB and multispectral trees.
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

# Remove empty class directories after filtering unsupported files.
find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d -empty -delete

echo "Using dataset root: $DATA_ROOT"
echo "Saving FID stats into: $OUTPUT_DIR"

IMAGE_COUNT="$(find "$DATA_ROOT/train" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
echo "Kept $IMAGE_COUNT image files for FID stats computation"
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
    --compute_fid=True \
    --overwrite=True

if [ -f "$OUTPUT_DIR/$DEFAULT_FID_NAME" ]; then
    mv "$OUTPUT_DIR/$DEFAULT_FID_NAME" "$OUTPUT_DIR/$TARGET_FID_NAME"
fi

echo "FID stats written to: $OUTPUT_DIR/$TARGET_FID_NAME"
