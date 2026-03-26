#!/bin/bash

set -euo pipefail

DATASET_DIR="${1:-/home/ens/AT74470/datasets}"
OUTPUT_DIR="${2:-/home/ens/AT74470/imeanflow/files/fdd_stats}"
IMAGE_SIZE="${3:-256}"
FD_DINO_ARCH="${4:-vitb14}"
CLEANUP_EXTRACTED="${5:-true}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory not found: $DATASET_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t DATASET_ZIPS < <(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.zip' | sort)

if [ "${#DATASET_ZIPS[@]}" -eq 0 ]; then
    echo "No dataset zip files found in: $DATASET_DIR"
    exit 1
fi

echo "Found ${#DATASET_ZIPS[@]} dataset zip files in: $DATASET_DIR"
echo "Saving FD-DINO stats into: $OUTPUT_DIR"
echo "Using DINOv2 architecture: $FD_DINO_ARCH"

for dataset_zip in "${DATASET_ZIPS[@]}"; do
    dataset_name="$(basename "$dataset_zip")"

    if [ "$dataset_name" = "caltech-101_processed.zip" ]; then
        echo
        echo "Skipping $dataset_name because Caltech FD-DINO stats were already generated."
        continue
    fi

    echo
    echo "=================================================="
    echo "Processing $dataset_name"
    echo "=================================================="

    bash /home/ens/AT74470/imeanflow/scripts/compute_fd_dino_from_zip.sh \
        "$dataset_zip" \
        "$OUTPUT_DIR" \
        "$IMAGE_SIZE" \
        "$FD_DINO_ARCH" \
        "$CLEANUP_EXTRACTED"
done

echo
echo "Finished FD-DINO generation for all remaining dataset zips."
