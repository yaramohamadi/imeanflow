#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <dataset_zip> <output_dir> [image_size] [fd_dino_arch] [cleanup_extracted] [batch_size]"
    echo "Example: $0 /path/to/data.zip /path/to/output 256 vitb14 true 16"
    exit 1
fi

DATASET_ZIP="$1"
OUTPUT_DIR="$2"
IMAGE_SIZE="${3:-256}"
FD_DINO_ARCH="${4:-vitb14}"
CLEANUP_EXTRACTED="${5:-true}"
BATCH_SIZE="${6:-${BATCH_SIZE:-32}}"
PYTHON="${PYTHON:-python3}"
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

if ! "$PYTHON" -c "import jax" >/dev/null 2>&1; then
    echo "ERROR: PYTHON='$PYTHON' cannot import jax." >&2
    echo "Run with: PYTHON=/home/ens/AT74470/imeanflow/.venv/bin/python" >&2
    exit 2
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
    mapfile -d '' CLASS_DIRS < <(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
    if [ "${#CLASS_DIRS[@]}" -gt 0 ]; then
        for class_dir in "${CLASS_DIRS[@]}"; do
            mv "$class_dir" "$NORMALIZED_ROOT/train/"
        done
    else
        mkdir -p "$NORMALIZED_ROOT/train/default"
        mapfile -d '' IMAGE_FILES < <(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type f -print0)
        for image_file in "${IMAGE_FILES[@]}"; do
            mv "$image_file" "$NORMALIZED_ROOT/train/default/"
        done
    fi
    DATA_ROOT="$NORMALIZED_ROOT"
fi

find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d -empty -delete

echo "Using dataset root: $DATA_ROOT"
echo "Saving FD-DINO stats into: $OUTPUT_DIR"
echo "FD-DINO batch size: $BATCH_SIZE"

IMAGE_COUNT="$(find "$DATA_ROOT/train" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
echo "Kept $IMAGE_COUNT image files for FD-DINO stats computation"
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: no image files found after extracting/normalizing $DATASET_ZIP" >&2
    echo "Debug by rerunning with CLEANUP_EXTRACTED=false to inspect $WORK_ROOT." >&2
    exit 3
fi
echo "Per-class image counts:"
find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | sort | while read -r class_dir; do
    class_name="$(basename "$class_dir")"
    class_count="$(find "$class_dir" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
    echo "  $class_name: $class_count"
done

"$PYTHON" prepare_dataset.py \
    --imagenet_root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR" \
    --image_size="$IMAGE_SIZE" \
    --batch_size="$BATCH_SIZE" \
    --compute_latent=False \
    --compute_fid=False \
    --compute_fd_dino=True \
    --fd_dino_arch="$FD_DINO_ARCH" \
    --overwrite=True

if [ -f "$OUTPUT_DIR/$DEFAULT_FD_DINO_NAME" ]; then
    mv "$OUTPUT_DIR/$DEFAULT_FD_DINO_NAME" "$OUTPUT_DIR/$TARGET_FD_DINO_NAME"
fi

echo "FD-DINO stats written to: $OUTPUT_DIR/$TARGET_FD_DINO_NAME"
