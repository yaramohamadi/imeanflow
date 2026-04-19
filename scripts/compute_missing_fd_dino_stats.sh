#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/home/ens/AT74470/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ens/AT74470/imeanflow/files/fdd_stats}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
FD_DINO_ARCH="${FD_DINO_ARCH:-vitb14}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CLEANUP_EXTRACTED="${CLEANUP_EXTRACTED:-true}"
OVERWRITE="${OVERWRITE:-False}"
DRY_RUN="${DRY_RUN:-False}"
DATASETS="${DATASETS:-cub200 food101 stanfordcars}"

mkdir -p "$OUTPUT_DIR"

dataset_zip_and_output() {
  case "$1" in
    caltech101|caltech-101)
      echo "$DATASET_DIR/caltech-101_processed.zip $OUTPUT_DIR/caltech-101-fd_dino-${FD_DINO_ARCH}_stats.npz"
      ;;
    artbench10|artbench-10)
      echo "$DATASET_DIR/artbench-10_processed.zip $OUTPUT_DIR/artbench-10-fd_dino-${FD_DINO_ARCH}_stats.npz"
      ;;
    cub200|cub-200|cub-200-2011)
      echo "$DATASET_DIR/cub-200-2011_processed.zip $OUTPUT_DIR/cub-200-2011-fd_dino-${FD_DINO_ARCH}_stats.npz"
      ;;
    food101|food-101)
      echo "$DATASET_DIR/food-101_processed.zip $OUTPUT_DIR/food-101-fd_dino-${FD_DINO_ARCH}_stats.npz"
      ;;
    stanfordcars|stanford-cars|cars)
      echo "$DATASET_DIR/stanford-cars_processed.zip $OUTPUT_DIR/stanford-cars-fd_dino-${FD_DINO_ARCH}_stats.npz"
      ;;
    *)
      echo "ERROR: unknown dataset '$1'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
      return 2
      ;;
  esac
}

echo "Dataset dir: $DATASET_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Image size: $IMAGE_SIZE"
echo "FD-DINO arch: $FD_DINO_ARCH"
echo "Batch size: $BATCH_SIZE"
echo "Datasets: $DATASETS"
echo "Overwrite: $OVERWRITE"
echo "Dry run: $DRY_RUN"

for dataset in $DATASETS; do
  read -r dataset_zip target_path < <(dataset_zip_and_output "$dataset")

  if [[ ! -f "$dataset_zip" ]]; then
    echo "ERROR: dataset zip not found for $dataset: $dataset_zip" >&2
    exit 2
  fi

  if [[ -f "$target_path" ]]; then
    case "${OVERWRITE,,}" in
      1|true|yes|y|on)
        echo "OVERWRITE=True, recomputing existing stats: $target_path"
        rm -f "$target_path"
        ;;
      *)
        echo "Skipping $dataset; stats already exist: $target_path"
        continue
        ;;
    esac
  fi

  echo
  echo "=================================================="
  echo "Computing FD-DINO stats for $dataset"
  echo "Zip: $dataset_zip"
  echo "Target: $target_path"
  echo "=================================================="

  case "${DRY_RUN,,}" in
    1|true|yes|y|on)
      echo "DRY_RUN=True, not computing."
      continue
      ;;
  esac

  bash scripts/compute_fd_dino_from_zip.sh \
    "$dataset_zip" \
    "$OUTPUT_DIR" \
    "$IMAGE_SIZE" \
    "$FD_DINO_ARCH" \
    "$CLEANUP_EXTRACTED" \
    "$BATCH_SIZE"

  if [[ ! -f "$target_path" ]]; then
    echo "ERROR: expected stats were not created: $target_path" >&2
    exit 3
  fi
done

echo
echo "Done. FD-DINO stats in: $OUTPUT_DIR"
