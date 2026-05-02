#!/usr/bin/env bash
#SBATCH --job-name=plain_jit_cc
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [[ -n "${REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/train_plain_jit_compute_canada.sbatch" ]]; then
  REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_DIR"

module load python/3.10.13 cuda/12.2
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  cat >&2 <<EOF
ERROR: Python virtualenv not found at '$VENV_DIR'.
Create it on the cluster first with:
  cd "$REPO_DIR"
  module load python/3.10.13 cuda/12.2
  VENV_DIR="$VENV_DIR" bash scripts/install.sh
EOF
  exit 4
fi
source "$VENV_DIR/bin/activate"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-/scratch/$USER}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_DIR}"

dataset_assets() {
  local dataset="$1"
  case "$dataset" in
    caltech101|caltech-101)
      DATASET_NAME_OUT="caltech101"
      DATASET_DIR_OUT="caltech-101_images"
      FID_FILE_OUT="caltech-101-fid_stats.npz"
      FD_DINO_FILE_OUT="caltech-101-fd_dino-vitb14_stats.npz"
      ;;
    artbench10|artbench-10)
      DATASET_NAME_OUT="artbench10"
      DATASET_DIR_OUT="artbench-10_images"
      FID_FILE_OUT="artbench-10_processed-fid_stats.npz"
      FD_DINO_FILE_OUT="artbench-10-fd_dino-vitb14_stats.npz"
      ;;
    cub200|cub-200|cub-200-2011)
      DATASET_NAME_OUT="cub200"
      DATASET_DIR_OUT="cub-200-2011_images"
      FID_FILE_OUT="cub-200-2011_processed-fid_stats.npz"
      FD_DINO_FILE_OUT="cub-200-2011-fd_dino-vitb14_stats.npz"
      ;;
    food101|food-101)
      DATASET_NAME_OUT="food101"
      DATASET_DIR_OUT="food-101_images"
      FID_FILE_OUT="food-101_processed-fid_stats.npz"
      FD_DINO_FILE_OUT="food-101-fd_dino-vitb14_stats.npz"
      ;;
    stanfordcars|stanford-cars|cars)
      DATASET_NAME_OUT="stanfordcars"
      DATASET_DIR_OUT="stanford-cars_images"
      FID_FILE_OUT="stanford_cars_processed-fid_stats.npz"
      FD_DINO_FILE_OUT="stanford-cars-fd_dino-vitb14_stats.npz"
      ;;
    *)
      echo "ERROR: unknown dataset '$dataset'." >&2
      exit 2
      ;;
  esac
}

pick_existing_path() {
  local primary="$1"
  local secondary="$2"
  if [[ -e "$primary" ]]; then
    printf '%s\n' "$primary"
    return 0
  fi
  if [[ -e "$secondary" ]]; then
    printf '%s\n' "$secondary"
    return 0
  fi
  return 1
}

require_existing_path() {
  local label="$1"
  local primary="$2"
  local secondary="$3"
  local resolved=""
  if ! resolved="$(pick_existing_path "$primary" "$secondary")"; then
    echo "ERROR: could not find $label at either '$primary' or '$secondary'." >&2
    exit 3
  fi
  printf '%s\n' "$resolved"
}

sync_run_dir() {
  mkdir -p "$PROJECT_RUN_DIR"
  rsync -a "$SCRATCH_RUN_DIR/" "$PROJECT_RUN_DIR/"
}

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME in sbatch export.}"
RUN_LABEL="${RUN_LABEL:-${DATASET_NAME}_jit_h16_lr3e6_ema9998_40k_fid2500}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
USE_WANDB="${USE_WANDB:-True}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.2}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
EMA_VAL="${EMA_VAL:-0.9998}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
FID_EVERY_STEPS="${FID_EVERY_STEPS:-2500}"
PREVIEW_EVERY_STEPS="${PREVIEW_EVERY_STEPS:-2500}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
OPTIMIZER="${OPTIMIZER:-adamw}"
OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"
HALF_PRECISION="${HALF_PRECISION:-True}"
SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SAVE_BEST_FID_EVAL_STATE_ONLY="${SAVE_BEST_FID_EVAL_STATE_ONLY:-True}"
RUN_ID="${RUN_ID:-plain_JiT_finetune_${DATASET_NAME}_${RUN_LABEL}_${SLURM_JOB_ID}}"

dataset_assets "$DATASET_NAME"

DATASET_ROOT="$(require_existing_path "dataset root" \
  "$SCRATCH_ROOT/datasets/$DATASET_DIR_OUT" \
  "$PROJECT_ROOT/datasets/$DATASET_DIR_OUT")"
FID_CACHE_REF="$(require_existing_path "FID stats" \
  "$SCRATCH_ROOT/fid_stats/$FID_FILE_OUT" \
  "$PROJECT_ROOT/fid_stats/$FID_FILE_OUT")"
FD_DINO_CACHE_REF="$(require_existing_path "FD-DINO stats" \
  "$SCRATCH_ROOT/fdd_stats/$FD_DINO_FILE_OUT" \
  "$PROJECT_ROOT/fdd_stats/$FD_DINO_FILE_OUT")"
LOAD_FROM="$(require_existing_path "JiT weights" \
  "$SCRATCH_ROOT/weights/JiT-H-16-256.pth" \
  "$PROJECT_ROOT/weights/JiT-H-16-256.pth")"

SCRATCH_RUN_DIR="$SCRATCH_ROOT/imeanflow/files/logs/finetuning/$RUN_ID"
PROJECT_RUN_DIR="$PROJECT_ROOT/files/logs/finetuning/$RUN_ID"
mkdir -p "$SCRATCH_RUN_DIR" "$PROJECT_RUN_DIR"

trap sync_run_dir EXIT

cat <<EOF
Repo dir: $REPO_DIR
Scratch root: $SCRATCH_ROOT
Project root: $PROJECT_ROOT
Dataset: $DATASET_NAME_OUT
Dataset root: $DATASET_ROOT
FID stats: $FID_CACHE_REF
FD-DINO stats: $FD_DINO_CACHE_REF
Weights: $LOAD_FROM
Scratch run dir: $SCRATCH_RUN_DIR
Project run dir: $PROJECT_RUN_DIR
Learning rate: $LEARNING_RATE
EMA value: $EMA_VAL
Max train steps: $MAX_TRAIN_STEPS
Train microbatch/effective batch: $TRAIN_BATCH_SIZE/$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))
Grad accumulation steps: $GRAD_ACCUM_STEPS
FID every steps: $FID_EVERY_STEPS
Preview every steps: $PREVIEW_EVERY_STEPS
Final eval steps: $FINAL_EVAL_STEPS
EOF

bash scripts/train_plain_jit_finetune.sh "$RUN_LABEL" \
  --config.model.model_str=flaxJiT_H_16 \
  --config.training.learning_rate="$LEARNING_RATE" \
  --config.training.max_train_steps="$MAX_TRAIN_STEPS" \
  --config.training.sample_per_step="$PREVIEW_EVERY_STEPS" \
  --config.training.fid_per_step=0 \
  --config.training.force_fid_per_step="$FID_EVERY_STEPS" \
  --config.training.force_metric_num_steps=4 \
  --config.training.use_ema=True \
  --config.training.ema_type=const \
  --config.training.ema_val="$EMA_VAL" \
  --config.training.optimizer="$OPTIMIZER" \
  --config.training.optimizer_mu_dtype="$OPTIMIZER_MU_DTYPE" \
  --config.training.batch_size="$TRAIN_BATCH_SIZE" \
  --config.training.grad_accum_steps="$GRAD_ACCUM_STEPS" \
  --config.training.half_precision="$HALF_PRECISION" \
  --config.training.half_precision_dtype=float16 \
  --config.sampling.half_precision="$SAMPLING_HALF_PRECISION" \
  --config.sampling.half_precision_dtype=float16 \
  --config.fid.sample_device_batch_size="$SAMPLE_DEVICE_BATCH_SIZE" \
  --config.fid.num_samples="$NUM_SAMPLES" \
  --config.sampling.num_steps=4 \
  --config.sampling.method=heun \
  --config.sampling.omega="$GUIDANCE_SCALE" \
  --config.sampling.cfg_scale="$GUIDANCE_SCALE" \
  --config.sampling.t_min=0.1 \
  --config.sampling.t_max=1.0 \
  --config.model.proj_drop=0.2 \
  --config.model.P_mean=-0.8 \
  --config.model.P_std=0.8 \
  --config.model.t_eps=0.05 \
  --config.model.noise_scale=1.0 \
  --config.model.class_dropout_prob=0.1 \
  --config.model.target_use_null_class=True \
  --config.training.save_best_fid_only=True \
  --config.training.save_best_fid_eval_state_only="$SAVE_BEST_FID_EVAL_STATE_ONLY" \
  --config.training.save_eval_checkpoint_per_fid=False \
  --config.training.eval_checkpoint_dir=latest_eval \
  --config.training.best_fid_checkpoint_dir=best_fid \
  --config.logging.wandb_project="$WANDB_PROJECT" \
  --config.logging.wandb_name="${DATASET_NAME_OUT}_plain_jit_${RUN_LABEL}" \
  --config.dataset.root="$DATASET_ROOT" \
  --config.fid.cache_ref="$FID_CACHE_REF" \
  --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
  --config.load_from="$LOAD_FROM" \
  "$@"

sync_run_dir

if [[ "${RUN_FINAL_BEST_FID_EVAL,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  read -r -a FINAL_EVAL_STEP_ARRAY <<< "$FINAL_EVAL_STEPS"
  bash scripts/eval_best_fid_steps_plain_jit.sh "$SCRATCH_RUN_DIR" "${FINAL_EVAL_STEP_ARRAY[@]}" -- \
    --config.model.model_str=flaxJiT_H_16 \
    --config.dataset.root="$DATASET_ROOT" \
    --config.fid.cache_ref="$FID_CACHE_REF" \
    --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
    --config.fid.sample_device_batch_size="$SAMPLE_DEVICE_BATCH_SIZE" \
    --config.fid.num_samples="$NUM_SAMPLES" \
    --config.sampling.method=heun \
    --config.sampling.omega="$GUIDANCE_SCALE" \
    --config.sampling.cfg_scale="$GUIDANCE_SCALE" \
    --config.sampling.t_min=0.1 \
    --config.sampling.t_max=1.0 \
    --config.training.use_ema=True \
    --config.logging.wandb_project="$WANDB_PROJECT" \
    --config.logging.wandb_name="${DATASET_NAME_OUT}_plain_jit_${RUN_LABEL}_best" \
    --config.logging.use_wandb="$USE_WANDB"
  sync_run_dir
fi
