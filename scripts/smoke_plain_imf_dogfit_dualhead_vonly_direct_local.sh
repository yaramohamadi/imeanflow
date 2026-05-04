#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/smoke_plain_imf_dogfit_dualhead_vonly_direct_local.sh [run_label]

Direct local smoke test for:
  - native iMF start
  - dual-head trainable model
  - DogFit + EMA v_c
  - v-only frozen teacher/source copies

Useful env overrides:
  PYTHON=.venv/bin/python
  DATASET_ROOT=/home/ens/AT74470/datasets/caltech-101_processed_latents
  FID_CACHE_REF=/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz
  FD_DINO_CACHE_REF=
  LOAD_FROM=/home/ens/AT74470/imeanflow/files/weights/iMF-XL-2-full
  LOG_DIR=files/logs/smoke
EOF
  exit 1
fi

RUN_LABEL="${1:-smoke_dualhead_vonly_direct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-False}"
LOG_DIR="${LOG_DIR:-files/logs/smoke}"
DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/iMF-XL-2-full}"

NOW=$(date '+%Y%m%d_%H%M%S')
SALT=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
JOBNAME="plain_iMF_DogFit_dualhead_vonly_smoke_${RUN_LABEL}_${NOW}_${SALT}"
WORKDIR="${LOG_DIR}/finetuning/${JOBNAME}"
mkdir -p "$WORKDIR"

echo "Smoke test workdir: $WORKDIR"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "FID_CACHE_REF: $FID_CACHE_REF"
echo "FD_DINO_CACHE_REF: $FD_DINO_CACHE_REF"
echo "LOAD_FROM: $LOAD_FROM"

CONFIG_ARGS=(
  --workdir="$WORKDIR"
  --config=configs/load_config.py:plain_imf_dogfit_finetune
  --config.dataset.root="$DATASET_ROOT"
  --config.dataset.num_classes=101
  --config.dataset.num_classes_from_data=False
  --config.model.model_str=imfDiT_XL_2
  --config.model.num_classes=101
  --config.model.use_auxiliary_v_head=True
  --config.model.use_dogfit=True
  --config.model.use_ema_vc=True
  --config.model.use_v_only_teacher_source_copies=True
  --config.model.target_use_null_class=False
  --config.model.source_num_classes=1000
  --config.model.guidance_scale_strategy=fixed
  --config.model.fixed_guidance_scale=1.5
  --config.model.training_guidance_interval_strategy=fixed
  --config.model.training_guidance_t_min=0.0
  --config.model.training_guidance_t_max=1.0
  --config.model.training_guidance_start_step=0
  --config.training.use_ema=True
  --config.training.capture_source_from_load=True
  --config.training.fid_use_online_only=True
  --config.training.max_train_steps=2
  --config.training.num_epochs=1
  --config.training.batch_size=1
  --config.training.grad_accum_steps=1
  --config.training.log_per_step=1
  --config.training.sample_per_step=0
  --config.training.fid_per_step=0
  --config.training.preview_at_step_zero=False
  --config.training.debug_log_during_train=False
  --config.fid.cache_ref="$FID_CACHE_REF"
  --config.fid.device_batch_size=1
  --config.fid.sample_device_batch_size=1
  --config.fid.num_images_to_log=0
  --config.logging.use_wandb="$USE_WANDB"
  --config.load_from="$LOAD_FROM"
)

if [[ -n "$FD_DINO_CACHE_REF" ]]; then
  CONFIG_ARGS+=(--config.fd_dino.cache_ref="$FD_DINO_CACHE_REF")
fi

TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
"$PYTHON" main.py "${CONFIG_ARGS[@]}" 2>&1 | tee -a "$WORKDIR/output.log"
