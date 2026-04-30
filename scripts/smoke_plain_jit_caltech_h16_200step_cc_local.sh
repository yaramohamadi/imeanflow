#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/scratch/$USER}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_DIR}"
RUN_LABEL="${RUN_LABEL:-caltech_h16_lr3e6_ema9998_200step_cc_local_smoke}"
RUN_ID="${RUN_ID:-plain_JiT_finetune_caltech101_${RUN_LABEL}_$(date '+%Y%m%d_%H%M%S')}"

SCRATCH_RUN_DIR="${WORKDIR:-$SCRATCH_ROOT/imeanflow/files/logs/finetuning/$RUN_ID}"
PROJECT_RUN_DIR="$PROJECT_ROOT/files/logs/finetuning/$RUN_ID"

DATASET_NAME="${DATASET_NAME:-caltech101}"
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH_ROOT/datasets/caltech-101_images}"
FID_CACHE_REF="${FID_CACHE_REF:-$SCRATCH_ROOT/fid_stats/caltech-101-fid_stats.npz}"
FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-$SCRATCH_ROOT/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
LOAD_FROM="${LOAD_FROM:-$SCRATCH_ROOT/weights/JiT-H-16-256.pth}"

USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
WANDB_NAME="${WANDB_NAME:-caltech101_plain_jit_${RUN_LABEL}_EMA}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-$USE_WANDB}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"

sync_run_dir() {
  mkdir -p "$PROJECT_RUN_DIR"
  if [[ -d "$SCRATCH_RUN_DIR" ]]; then
    rsync -a "$SCRATCH_RUN_DIR/" "$PROJECT_RUN_DIR/"
  fi
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 2
  fi
}

trap sync_run_dir EXIT

require_path "python" "$PYTHON"
require_path "dataset root" "$DATASET_ROOT"
require_path "FID stats" "$FID_CACHE_REF"
require_path "FD-DINO stats" "$FD_DINO_CACHE_REF"
require_path "JiT weights" "$LOAD_FROM"
mkdir -p "$SCRATCH_RUN_DIR" "$PROJECT_RUN_DIR"

if [[ "${REQUIRE_GPU:-True}" =~ ^([Tt]rue|1|[Yy]es|[Yy]|[Oo]n)$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    sys.exit("ERROR: no JAX GPU device is visible. Run inside an interactive GPU allocation or set REQUIRE_GPU=False intentionally.")
PY
fi

cat <<EOF
Local JiT Caltech smoke
repo: $REPO_DIR
scratch root: $SCRATCH_ROOT
scratch run dir: $SCRATCH_RUN_DIR
project run dir: $PROJECT_RUN_DIR
dataset root: $DATASET_ROOT
FID stats: $FID_CACHE_REF
FD-DINO stats: $FD_DINO_CACHE_REF
weights: $LOAD_FROM
train steps: 200
train microbatch/effective batch: $TRAIN_BATCH_SIZE/$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))
grad accumulation steps: $GRAD_ACCUM_STEPS
preview/FID every: 100
final eval steps: $FINAL_EVAL_STEPS
EOF

WORKDIR="$SCRATCH_RUN_DIR" \
RUN_FINAL_BEST_FID_EVAL=False \
bash scripts/train_plain_jit_finetune.sh "$RUN_LABEL" \
  --config.model.model_str=flaxJiT_H_16 \
  --config.training.batch_size="$TRAIN_BATCH_SIZE" \
  --config.training.grad_accum_steps="$GRAD_ACCUM_STEPS" \
  --config.training.max_train_steps=200 \
  --config.training.learning_rate=3e-6 \
  --config.training.use_ema=True \
  --config.training.ema_type=const \
  --config.training.ema_val=0.9998 \
  --config.training.optimizer=adamw \
  --config.training.optimizer_mu_dtype=float16 \
  --config.training.half_precision=True \
  --config.training.half_precision_dtype=float16 \
  --config.training.sample_per_step=100 \
  --config.training.fid_per_step=0 \
  --config.training.force_fid_per_step=100 \
  --config.training.force_metric_num_steps=4 \
  --config.training.preview_num_steps='(4,)' \
  --config.training.preview_guidance_scales='(2.2,)' \
  --config.training.preview_at_step0=True \
  --config.training.save_best_fid_only=True \
  --config.training.save_best_fid_eval_state_only=True \
  --config.training.save_eval_checkpoint_per_fid=False \
  --config.training.eval_checkpoint_dir=latest_eval \
  --config.training.best_fid_checkpoint_dir=best_fid \
  --config.sampling.num_steps=4 \
  --config.sampling.method=heun \
  --config.sampling.half_precision=True \
  --config.sampling.half_precision_dtype=float16 \
  --config.sampling.omega=2.2 \
  --config.sampling.cfg_scale=2.2 \
  --config.sampling.t_min=0.1 \
  --config.sampling.t_max=1.0 \
  --config.model.proj_drop=0.2 \
  --config.model.P_mean=-0.8 \
  --config.model.P_std=0.8 \
  --config.model.t_eps=0.05 \
  --config.model.noise_scale=1.0 \
  --config.model.class_dropout_prob=0.1 \
  --config.model.target_use_null_class=True \
  --config.fid.sample_device_batch_size="$SAMPLE_DEVICE_BATCH_SIZE" \
  --config.fid.num_samples="$NUM_SAMPLES" \
  --config.fid.num_images_to_log=16 \
  --config.dataset.root="$DATASET_ROOT" \
  --config.fid.cache_ref="$FID_CACHE_REF" \
  --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
  --config.load_from="$LOAD_FROM" \
  --config.logging.use_wandb="$USE_WANDB" \
  --config.logging.wandb_project="$WANDB_PROJECT" \
  --config.logging.wandb_name="$WANDB_NAME" \
  "$@"

sync_run_dir

read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
CONFIG_MODE=plain_jit_finetune \
PYTHON="$PYTHON" \
USE_WANDB="$FINAL_EVAL_USE_WANDB" \
WANDB_NAME_PREFIX="$WANDB_NAME" \
bash scripts/eval_best_fid_steps_plain_jit.sh "$SCRATCH_RUN_DIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
  --config.model.model_str=flaxJiT_H_16 \
  --config.training.use_ema=True \
  --config.dataset.root="$DATASET_ROOT" \
  --config.fid.cache_ref="$FID_CACHE_REF" \
  --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
  --config.fid.sample_device_batch_size="$SAMPLE_DEVICE_BATCH_SIZE" \
  --config.fid.num_samples="$NUM_SAMPLES" \
  --config.sampling.method=heun \
  --config.sampling.half_precision=True \
  --config.sampling.half_precision_dtype=float16 \
  --config.sampling.omega=2.2 \
  --config.sampling.cfg_scale=2.2 \
  --config.sampling.t_min=0.1 \
  --config.sampling.t_max=1.0 \
  --config.logging.wandb_project="$WANDB_PROJECT"

sync_run_dir
echo "Synced smoke run to: $PROJECT_RUN_DIR"
