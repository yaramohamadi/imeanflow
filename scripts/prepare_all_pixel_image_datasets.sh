#!/bin/bash

set -euo pipefail

# Prepare all dataset zip archives into persistent JiT pixel-space ImageFolder
# roots:
#   <OUTPUT_PARENT>/<dataset-name>_images/train/<class>/*

DATASET_DIR="${DATASET_DIR:-/home/ens/AT74470/datasets}"
OUTPUT_PARENT="${OUTPUT_PARENT:-$DATASET_DIR}"
OVERWRITE="${OVERWRITE:-False}"
PYTHON="${PYTHON:-.venv/bin/python}"
SKIP_DATASETS="${SKIP_DATASETS:-eurosat-dataset ffhq256}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

mapfile -t DATASET_ZIPS < <(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.zip' ! -name '*_latents.zip' | sort)

if [ "${#DATASET_ZIPS[@]}" -eq 0 ]; then
    echo "No non-latent dataset zip files found in: $DATASET_DIR"
    exit 1
fi

echo "Found ${#DATASET_ZIPS[@]} dataset zip files in: $DATASET_DIR"
echo "Output parent: $OUTPUT_PARENT"
echo "Overwrite: $OVERWRITE"
echo "Skip datasets: $SKIP_DATASETS"

for dataset_zip in "${DATASET_ZIPS[@]}"; do
    dataset_name="$(basename "$dataset_zip" .zip)"
    dataset_name="${dataset_name%_processed}"
    dataset_name="${dataset_name%-processed}"
    skip_dataset=False
    for skip_name in $SKIP_DATASETS; do
        if [ "$dataset_name" = "$skip_name" ]; then
            skip_dataset=True
            break
        fi
    done
    if [ "$skip_dataset" = "True" ]; then
        echo "Skipping $dataset_name due to SKIP_DATASETS."
        continue
    fi
    output_root="$OUTPUT_PARENT/${dataset_name}_images"

    if [ -d "$output_root" ] && [ "$OVERWRITE" != "True" ] && [ "$OVERWRITE" != "true" ]; then
        echo "Skipping existing output: $output_root"
        continue
    fi

    echo "=============================================="
    echo "Preparing $dataset_zip"
    echo "Output: $output_root"
    echo "=============================================="

    args=("$dataset_zip" "$output_root")
    if [ "$OVERWRITE" = "True" ] || [ "$OVERWRITE" = "true" ]; then
        args+=(--overwrite)
    fi
    "$PYTHON" scripts/prepare_pixel_image_dataset.py "${args[@]}"
done

echo "Finished preparing pixel-space image datasets."
