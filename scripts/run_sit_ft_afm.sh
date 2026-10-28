#!/usr/bin/env bash
# Tab. 2 "FT + AFM" (SiT-XL/2): official fixed-NFE AFM post-training of a target-finetuned PLAIN SiT,
# then 10K-sample final eval of the best_fid checkpoint at 4 Euler steps with CFG (omega 1.5, "4x2")
# and without CFG (omega 1, 4 NFE). One GPU (CUDA_VISIBLE_DEVICES from the caller).
# Usage: bash scripts/run_sit_ft_afm.sh <dataset> <FT best_fid/checkpoint_* dir> <workdir>
set -euo pipefail
if [[ $# -ne 3 ]]; then echo "Usage: $0 <dataset> <load_from> <workdir>" >&2; exit 2; fi
DATASET="$1"; LOAD_FROM="$2"; WORKDIR="$3"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON:-/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python}"
CONFIG_MODE="${CONFIG_MODE:-sit_ft_afm_tab2}"
DATA_ROOT="${DATA_ROOT:-$REPO/../datasets}"
case "$DATASET" in
  caltech101)   LATENT_DIR="caltech-101_processed_latents"; NC=101; FID="caltech-101-fid_stats.npz"; FDD="caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   LATENT_DIR="artbench-10_processed_latents"; NC=10; FID="artbench-10_processed-fid_stats.npz"; FDD="artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       LATENT_DIR="cub-200-2011_processed_latents"; NC=200; FID="cub-200-2011_processed-fid_stats.npz"; FDD="cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      LATENT_DIR="food-101_processed_latents"; NC=101; FID="food-101_processed-fid_stats.npz"; FDD="food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) LATENT_DIR="stanford-cars_processed_latents"; NC=196; FID="stanford_cars_processed-fid_stats.npz"; FDD="stanford-cars-fd_dino-vitb14_stats.npz" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
DATA_ROOT="$DATA_ROOT/$LATENT_DIR"
[[ -d "$LOAD_FROM" ]] || { echo "Checkpoint directory not found: $LOAD_FROM" >&2; exit 3; }
[[ -d "$DATA_ROOT/train" ]] || { echo "Latent train dir not found: $DATA_ROOT/train" >&2; exit 3; }
if [[ -d "$WORKDIR" ]] && find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print -quit | grep -q .; then
  echo "Workdir already contains a checkpoint: $WORKDIR (use a new workdir)" >&2; exit 4
fi
mkdir -p "$WORKDIR"
cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
ARGS=(
  --config="$REPO/configs/load_config.py:$CONFIG_MODE"
  --config.dataset.name="${DATASET}_latent"
  --config.dataset.root="$DATA_ROOT"
  --config.dataset.class_mapping_root=""
  --config.dataset.num_classes="$NC"
  --config.model.num_classes="$NC"
  --config.sampling.num_classes="$NC"
  --config.fid.cache_ref="$REPO/files/fid_stats/$FID"
  --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD"
  --workdir="$WORKDIR"
)
echo "SiT-FT AFM: $DATASET  ckpt=$LOAD_FROM  workdir=$WORKDIR  config=$CONFIG_MODE"
"$PYTHON_BIN" main_afm_sit_ft.py "${ARGS[@]}" --config.load_from="$LOAD_FROM"
if [[ "${RUN_FINAL_EVAL:-True}" == "True" ]]; then
  BEST=$(ls -d "$WORKDIR"/best_fid/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [[ -n "$BEST" ]] || { echo "No best_fid checkpoint in $WORKDIR" >&2; exit 5; }
  echo "Final eval of $BEST (4 Euler steps, omega 1.5 CFG and omega 1 no CFG)"
  "$PYTHON_BIN" main_afm_sit_ft.py "${ARGS[@]}" --config.load_from="$LOAD_FROM" --final_eval_ckpt="$BEST"
fi
