#!/usr/bin/env bash
# Run CA-iMF (CAMF) post-training on a JiT-DMF MeFT checkpoint (pixel space).
# Usage: bash scripts/run_jit_meft_caimf.sh <dataset> <load_from> <workdir>
# Optional env: SMOKE=1 (short run), EXTRA_ARGS="--config.x=y ...",
#   FINAL_EVAL_STEPS="1 2", RUN_FINAL_EVAL=True
set -uo pipefail
if [[ $# -ne 3 ]]; then echo "Usage: $0 <dataset> <load_from> <workdir>" >&2; exit 2; fi
DATASET="$1"; LOAD_FROM="$2"; WORKDIR="$3"
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY="${PYTHON:-$MAIN/.venv/bin/python}"
DATA=/opt/dlami/nvme/meanflow/datasets
CONFIG_MODE=caltech_jit_meft_caimf_posttrain

case "$DATASET" in
  artbench10)   IMG=artbench-10_images;  NC=10;  FID=artbench-10_processed-fid_stats.npz; FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  caltech101)   IMG=caltech-101_images;  NC=101; FID=caltech-101-fid_stats.npz;          FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       IMG=cub-200-2011_images; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  food101)      IMG=food-101_images;     NC=101; FID=food-101_processed-fid_stats.npz;   FDD=food-101-fd_dino-vitb14_stats.npz ;;
  stanfordcars) IMG=stanford-cars_images; NC=196; FID=stanford_cars_processed-fid_stats.npz; FDD=stanford-cars-fd_dino-vitb14_stats.npz ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
ROOT="$DATA/$IMG"
[[ -d "$ROOT/train" ]] || { echo "Missing $ROOT/train" >&2; exit 3; }
[[ -d "$LOAD_FROM" ]] || { echo "Missing ckpt $LOAD_FROM" >&2; exit 3; }
[[ -e "$MAIN/files/fid_stats/$FID" ]] || { echo "Missing fid $FID" >&2; exit 3; }
[[ -e "$MAIN/files/fdd_stats/$FDD" ]] || { echo "Missing fdd $FDD" >&2; exit 3; }
mkdir -p "$WORKDIR"

SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  # Short GPU feasibility run: 200 D-warmup + a little G, one eval, tiny samples.
  # force_fid_per_step (scalar) overrides the nested fid_schedule list, which
  # absl config_flags cannot address by index.
  SMOKE_ARGS=(
    --config.caimf.discriminator_warmup_batches=200
    --config.caimf.max_posttrain_batches=800
    --config.training.force_fid_per_step=400
    --config.fid.num_samples=200
  )
fi

cd "$ADV"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

set +e
"$PY" main_caimf_jit_meft.py \
  --config="$ADV/configs/load_config.py:$CONFIG_MODE" \
  --config.load_from="$LOAD_FROM" \
  --config.dataset.root="$ROOT" \
  --config.dataset.num_classes="$NC" \
  --config.model.num_classes="$NC" \
  --config.sampling.num_classes="$NC" \
  --config.fid.cache_ref="$MAIN/files/fid_stats/$FID" \
  --config.fd_dino.cache_ref="$MAIN/files/fdd_stats/$FDD" \
  --config.logging.use_wandb=False \
  "${SMOKE_ARGS[@]}" ${EXTRA_ARGS:-} \
  --workdir="$WORKDIR"
STATUS=$?
set -e
[[ "$STATUS" -ne 0 ]] && exit "$STATUS"

if [[ "${SMOKE:-0}" != "1" && "${RUN_FINAL_EVAL:-True}" == "True" ]]; then
  # The generic MeFT-adversarial eval script is architecture-agnostic: it runs
  # main.py in eval_only mode with our config mode, which builds imfJiT_DMF_H_16.
  read -r -a STEP_ARR <<< "${FINAL_EVAL_STEPS:-1 2}"
  CONFIG_MODE="$CONFIG_MODE" PYTHON="$PY" USE_WANDB=False \
    bash "$ADV/scripts/eval_best_fid_steps_sit_meft_adversarial.sh" \
      "$WORKDIR" "${STEP_ARR[@]}" -- \
      "--config.dataset.root=$ROOT" \
      "--config.dataset.num_classes=$NC" \
      "--config.model.num_classes=$NC" \
      "--config.sampling.num_classes=$NC" \
      "--config.fid.cache_ref=$MAIN/files/fid_stats/$FID" \
      "--config.fd_dino.cache_ref=$MAIN/files/fdd_stats/$FDD"
fi
