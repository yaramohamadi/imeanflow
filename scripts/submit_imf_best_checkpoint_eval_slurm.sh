#!/usr/bin/env bash
set -euo pipefail

cat_usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_imf_best_checkpoint_eval_slurm.sh [extra eval config overrides...]

Submits one SLURM job per dataset to evaluate the existing iMF best_fid checkpoint.
Each job runs scripts/eval_best_fid_steps_plain_imf.sh for FINAL_EVAL_STEPS.

Env knobs:
  DATASETS="caltech101 artbench10 cub200 food101 stanfordcars"
  SEARCH_ROOTS="files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923 files/logs/sweeps/plain_imf_plain_4step_best_imf_remaining_b16_20260421_124033"
  FINAL_EVAL_STEPS="1 2 250"
  FID_NUM_SAMPLES=10000
  SAMPLE_DEVICE_BATCH_SIZE=32
  SUBMIT_SLURM=True                  # True submits; DryRun writes sbatch files only
  EVAL_SWEEP_DIR=files/logs/sweeps/imf_best_checkpoint_eval_<timestamp>
  SLURM_ACCOUNT=def-hadi87
  SLURM_GRES=gpu:h100:1
  SLURM_MEM=64G
  SLURM_CPUS_PER_TASK=8
  SLURM_TIME=12:00:00
  SLURM_MAIL_USER=yara.mohammadi-bahram@livia.etsmtl.ca
  SLURM_MAIL_TYPE=BEGIN,END,FAIL
  PYTHON_MODULE=python/3.10.13
  CUDA_MODULE=cuda/12.2
  PYTHON=.venv/bin/python
  USE_WANDB=False

Examples:
  SUBMIT_SLURM=DryRun bash scripts/submit_imf_best_checkpoint_eval_slurm.sh
  DATASETS="stanfordcars cub200" FINAL_EVAL_STEPS="250" bash scripts/submit_imf_best_checkpoint_eval_slurm.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat_usage
  exit 0
fi

EXTRA_ARGS=("$@")

DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}"
SEARCH_ROOTS="${SEARCH_ROOTS:-files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923 files/logs/sweeps/plain_imf_plain_4step_best_imf_remaining_b16_20260421_124033}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}"
SUBMIT_SLURM="${SUBMIT_SLURM:-True}"
USE_WANDB="${USE_WANDB:-False}"
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG_MODE="${CONFIG_MODE:-plain_imf_finetune}"

NOW="$(date '+%Y%m%d_%H%M%S')"
EVAL_SWEEP_DIR="${EVAL_SWEEP_DIR:-files/logs/sweeps/imf_best_checkpoint_eval_${NOW}}"
SLURM_SCRIPT_DIR="$EVAL_SWEEP_DIR/slurm"
mkdir -p "$SLURM_SCRIPT_DIR"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-def-hadi87}"
SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-yara.mohammadi-bahram@livia.etsmtl.ca}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
REPO_ROOT="$(pwd)"

MANIFEST="$EVAL_SWEEP_DIR/manifest.tsv"
printf "dataset\trun_dir\tcheckpoint\tsteps\tstatus\n" > "$MANIFEST"

quote_args() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s %q' "$out" "$arg"
  done
  printf '%s' "$out"
}

set_dataset_overrides() {
  local dataset="$1"
  case "$dataset" in
    caltech101|caltech-101)
      DATASET_LABEL="caltech101"
      DATASET_ROOT_FOR_JOB="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_processed_latents}"
      FID_CACHE_REF_FOR_JOB="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_JOB="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
      ;;
    artbench10|artbench-10)
      DATASET_LABEL="artbench10"
      DATASET_ROOT_FOR_JOB="${DATASET_ROOT:-/scratch/ymbahram/datasets/artbench-10_processed_latents}"
      FID_CACHE_REF_FOR_JOB="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/artbench-10_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_JOB="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
      ;;
    cub200|cub-200|cub-200-2011)
      DATASET_LABEL="cub200"
      DATASET_ROOT_FOR_JOB="${DATASET_ROOT:-/scratch/ymbahram/datasets/cub-200-2011_processed_latents}"
      FID_CACHE_REF_FOR_JOB="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/cub-200-2011_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_JOB="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
      ;;
    food101|food-101)
      DATASET_LABEL="food101"
      DATASET_ROOT_FOR_JOB="${DATASET_ROOT:-/scratch/ymbahram/datasets/food-101_processed_latents}"
      FID_CACHE_REF_FOR_JOB="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/food-101_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_JOB="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
      ;;
    stanfordcars|stanford-cars|cars)
      DATASET_LABEL="stanfordcars"
      DATASET_ROOT_FOR_JOB="${DATASET_ROOT:-/scratch/ymbahram/datasets/stanford-cars_processed_latents}"
      FID_CACHE_REF_FOR_JOB="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/stanford_cars_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_JOB="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
      ;;
    *)
      echo "ERROR: unknown dataset '$dataset'." >&2
      exit 2
      ;;
  esac
}

find_latest_run_for_dataset() {
  local dataset="$1"
  local root
  for root in $SEARCH_ROOTS; do
    [[ -d "$root" ]] || continue
    find "$root" -path "*/finetuning/*${dataset}*imf*" -type d \
      -exec test -d "{}/best_fid" ";" -print
  done | sort | tail -n 1
}

check_dataset_root_or_zip() {
  local root="$1"
  [[ -d "$root" || -f "${root}.zip" ]]
}

cat <<EOF
iMF best-checkpoint eval SLURM submitter
Datasets: $DATASETS
Search roots: $SEARCH_ROOTS
Final eval steps: $FINAL_EVAL_STEPS
Eval sweep dir: $EVAL_SWEEP_DIR
Submit SLURM: $SUBMIT_SLURM
Use W&B: $USE_WANDB
EOF

for DATASET in $DATASETS; do
  set_dataset_overrides "$DATASET"
  RUN_DIR="$(find_latest_run_for_dataset "$DATASET_LABEL")"
  if [[ -z "$RUN_DIR" ]]; then
    echo "WARNING: no run with best_fid found for $DATASET_LABEL; skipping." >&2
    printf "%s\t\t\t%s\tskipped_missing_run\n" "$DATASET_LABEL" "$FINAL_EVAL_STEPS" >> "$MANIFEST"
    continue
  fi

  mapfile -t CKPTS < <(find "$RUN_DIR/best_fid" -maxdepth 1 -type d -name 'checkpoint_*' | sort)
  if [[ ${#CKPTS[@]} -ne 1 ]]; then
    echo "WARNING: expected one checkpoint under $RUN_DIR/best_fid, found ${#CKPTS[@]}; skipping." >&2
    printf "%s\t%s\t\t%s\tskipped_bad_checkpoint_count\n" "$DATASET_LABEL" "$RUN_DIR" "$FINAL_EVAL_STEPS" >> "$MANIFEST"
    continue
  fi

  if ! check_dataset_root_or_zip "$DATASET_ROOT_FOR_JOB"; then
    echo "WARNING: missing dataset root/zip for $DATASET_LABEL: $DATASET_ROOT_FOR_JOB; job may fail." >&2
  fi

  EVAL_ARGS=(
    --config.dataset.root="$DATASET_ROOT_FOR_JOB"
    --config.fid.cache_ref="$FID_CACHE_REF_FOR_JOB"
    --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF_FOR_JOB"
  )
  if [[ -n "${FID_NUM_SAMPLES:-}" ]]; then
    EVAL_ARGS+=(--config.fid.num_samples="$FID_NUM_SAMPLES")
  fi
  if [[ -n "${SAMPLE_DEVICE_BATCH_SIZE:-}" ]]; then
    EVAL_ARGS+=(--config.fid.sample_device_batch_size="$SAMPLE_DEVICE_BATCH_SIZE")
  fi
  EVAL_ARGS+=("${EXTRA_ARGS[@]}")

  JOB_NAME="imf_eval_${DATASET_LABEL}"
  SBATCH_FILE="$SLURM_SCRIPT_DIR/${JOB_NAME}.sbatch"
  OUT_FILE="$SLURM_SCRIPT_DIR/${JOB_NAME}_%j.out"
  ERR_FILE="$SLURM_SCRIPT_DIR/${JOB_NAME}_%j.err"
  QUOTED_EVAL_ARGS="$(quote_args "${EVAL_ARGS[@]}")"

  cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME
#SBATCH --gres=$SLURM_GRES
#SBATCH --output=$OUT_FILE
#SBATCH --error=$ERR_FILE
#SBATCH --mail-user=$SLURM_MAIL_USER
#SBATCH --mail-type=$SLURM_MAIL_TYPE

set -euo pipefail

cd "$REPO_ROOT"
module load "$PYTHON_MODULE" "$CUDA_MODULE"
source .venv/bin/activate

export MPLCONFIGDIR="\${MPLCONFIGDIR:-/tmp/\$USER-matplotlib}"
export PYTHON="$PYTHON"
export CONFIG_MODE="$CONFIG_MODE"
export USE_WANDB="$USE_WANDB"
export WANDB_NAME_PREFIX="${DATASET_LABEL}_plain_imf_best_checkpoint_eval"

echo "Dataset: $DATASET_LABEL"
echo "Run dir: $RUN_DIR"
echo "Checkpoint: ${CKPTS[0]}"
echo "Final eval steps: $FINAL_EVAL_STEPS"

CONFIG_MODE="$CONFIG_MODE" PYTHON="$PYTHON" USE_WANDB="$USE_WANDB" \\
  bash scripts/eval_best_fid_steps_plain_imf.sh "$RUN_DIR" $FINAL_EVAL_STEPS --$QUOTED_EVAL_ARGS
EOF

  printf "%s\t%s\t%s\t%s\twritten\n" "$DATASET_LABEL" "$RUN_DIR" "${CKPTS[0]}" "$FINAL_EVAL_STEPS" >> "$MANIFEST"

  case "${SUBMIT_SLURM,,}" in
    1|true|yes|y|on)
      sbatch "$SBATCH_FILE"
      ;;
    dryrun|dry-run|0|false|no|n|off)
      echo "DryRun: wrote $SBATCH_FILE"
      ;;
    *)
      echo "ERROR: SUBMIT_SLURM must be True or DryRun/False, got '$SUBMIT_SLURM'." >&2
      exit 2
      ;;
  esac
done

echo "Manifest: $MANIFEST"
