#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: BACKBONES="sit dit" bash scripts/sweep_plain_finetune_datasets.sh <sweep_label> [extra config overrides...]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=sit bash scripts/sweep_plain_finetune_datasets.sh baseline

  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=dit DATASETS="caltech101 food101" bash scripts/sweep_plain_finetune_datasets.sh ddpm_baseline \
    --config.training.max_train_steps=10000

  SUBMIT_SLURM=True BACKBONES="sit dit" DATASETS="caltech101 food101" \
    bash scripts/sweep_plain_finetune_datasets.sh baseline

Env knobs:
  BACKBONES="sit"                    # one or more: sit dit
  BACKBONE=sit                       # legacy alias used only when BACKBONES is unset
  DATASETS="caltech101 artbench10 cub200 food101 stanfordcars"
  SWEEP_LOG_DIR=files/logs/sweeps/... # default: files/logs/sweeps/plain_<backbones>_<label>_<time>
  WANDB_PROJECT=...                   # optional single project for all backbones
  WANDB_PROJECT_SIT=plain_sit_finetune
  WANDB_PROJECT_DIT=plain_dit_finetune
  CONTINUE_ON_FAILURE=False           # set True to continue after a dataset fails
  ASSET_CHECK=True                    # verify dataset/stat/weight paths before running or submitting
  SUBMIT_SLURM=False                  # True submits; DryRun writes sbatch files without submitting; False runs locally
  SLURM_ACCOUNT=def-hadi87
  SLURM_GRES=gpu:h100:1               # override if your cluster uses a different H100 GRES name
  SLURM_MEM=48G
  SLURM_CPUS_PER_TASK=8
  SLURM_MAIL_USER=yara.mohammadi-bahram@livia.etsmtl.ca
  SLURM_MAIL_TYPE=BEGIN,END,FAIL
  SLURM_TIME=...                     # optional single time for all jobs
  SLURM_TIME_SIT=11:00:00
  SLURM_TIME_DIT=08:00:00
  PYTHON_MODULE=python/3.10.13
  CUDA_MODULE=cuda/12.2

All extra args are forwarded to the underlying train script.
EOF
  exit 1
fi

SWEEP_LABEL="$1"
shift
EXTRA_ARGS=("$@")

BACKBONES="${BACKBONES:-${BACKBONE:-sit}}"
DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-False}"
ASSET_CHECK="${ASSET_CHECK:-True}"
SUBMIT_SLURM="${SUBMIT_SLURM:-False}"
NOW=$(date '+%Y%m%d_%H%M%S')
REPO_ROOT="$(pwd)"

<<<<<<< HEAD
BACKBONES_LABEL="${BACKBONES// /_}"
=======
case "${BACKBONE,,}" in
  sit)
    TRAIN_SCRIPT="scripts/train_plain_sit_finetune.sh"
    BACKBONE_LABEL="sit"
    DEFAULT_WANDB_PROJECT="plain_sit_finetune"
    DEFAULT_SLURM_TIME="60:00:00"
    ;;
  dit)
    TRAIN_SCRIPT="scripts/train_plain_dit_finetune.sh"
    BACKBONE_LABEL="dit"
    DEFAULT_WANDB_PROJECT="plain_dit_finetune"
    DEFAULT_SLURM_TIME="45:00:00"
    ;;
  imf)
    TRAIN_SCRIPT="scripts/train_plain_imf_finetune.sh"
    BACKBONE_LABEL="imf"
    DEFAULT_WANDB_PROJECT="plain_imf_finetune"
    ;;
  *)
    echo "ERROR: BACKBONE must be sit, dit, or imf, got '${BACKBONE}'." >&2
    exit 2
    ;;
esac
>>>>>>> b3fb1390dad8051fe72f432ed0215435fffa0a10

SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-files/logs/sweeps/plain_${BACKBONES_LABEL}_${SWEEP_LABEL}_${NOW}}"
mkdir -p "$SWEEP_LOG_DIR"

MANIFEST="$SWEEP_LOG_DIR/sweep_manifest.tsv"
printf "dataset\tbackbone\trun_label\twandb_project\twandb_name\tslurm_time\tstatus\n" > "$MANIFEST"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-def-hadi87}"
SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
SLURM_MEM="${SLURM_MEM:-48G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-yara.mohammadi-bahram@livia.etsmtl.ca}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
SLURM_SCRIPT_DIR="$SWEEP_LOG_DIR/slurm"

cat <<EOF
Sweep label: $SWEEP_LABEL
Backbones: $BACKBONES
Datasets: $DATASETS
Sweep log dir: $SWEEP_LOG_DIR
Submit Slurm: $SUBMIT_SLURM
Asset check: $ASSET_CHECK
Extra args: ${EXTRA_ARGS[*]:-<none>}
EOF

set_backbone_defaults() {
  local backbone="$1"
  case "${backbone,,}" in
    sit)
      TRAIN_SCRIPT="scripts/train_plain_sit_finetune.sh"
      BACKBONE_LABEL="sit"
      WANDB_PROJECT_FOR_JOB="${WANDB_PROJECT:-${WANDB_PROJECT_SIT:-plain_sit_finetune}}"
      SLURM_TIME_FOR_JOB="${SLURM_TIME:-${SLURM_TIME_SIT:-11:00:00}}"
      ;;
    dit)
      TRAIN_SCRIPT="scripts/train_plain_dit_finetune.sh"
      BACKBONE_LABEL="dit"
      WANDB_PROJECT_FOR_JOB="${WANDB_PROJECT:-${WANDB_PROJECT_DIT:-plain_dit_finetune}}"
      SLURM_TIME_FOR_JOB="${SLURM_TIME:-${SLURM_TIME_DIT:-08:00:00}}"
      ;;
    *)
      echo "ERROR: BACKBONES entries must be sit or dit, got '${backbone}'." >&2
      exit 2
      ;;
  esac
}

set_dataset_assets_for_check() {
  local dataset="$1"
  case "${dataset}" in
    caltech101|caltech-101)
      DATASET_ROOT_FOR_CHECK="${DATASET_ROOT:-/scratch/ymbahram/datasets/caltech-101_processed_latents}"
      FID_CACHE_REF_FOR_CHECK="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_CHECK="${FD_DINO_CACHE_REF-/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
      ;;
    artbench10|artbench-10)
      DATASET_ROOT_FOR_CHECK="${DATASET_ROOT:-/scratch/ymbahram/datasets/artbench-10_processed_latents}"
      FID_CACHE_REF_FOR_CHECK="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/artbench-10_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_CHECK="${FD_DINO_CACHE_REF-/scratch/ymbahram/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz}"
      ;;
    cub200|cub-200|cub-200-2011)
      DATASET_ROOT_FOR_CHECK="${DATASET_ROOT:-/scratch/ymbahram/datasets/cub-200-2011_processed_latents}"
      FID_CACHE_REF_FOR_CHECK="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/cub-200-2011_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_CHECK="${FD_DINO_CACHE_REF-/scratch/ymbahram/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
      ;;
    food101|food-101)
      DATASET_ROOT_FOR_CHECK="${DATASET_ROOT:-/scratch/ymbahram/datasets/food-101_processed_latents}"
      FID_CACHE_REF_FOR_CHECK="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/food-101_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_CHECK="${FD_DINO_CACHE_REF-/scratch/ymbahram/fdd_stats/food-101-fd_dino-vitb14_stats.npz}"
      ;;
    stanfordcars|stanford-cars|cars)
      DATASET_ROOT_FOR_CHECK="${DATASET_ROOT:-/scratch/ymbahram/datasets/stanford-cars_processed_latents}"
      FID_CACHE_REF_FOR_CHECK="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/stanford_cars_processed-fid_stats.npz}"
      FD_DINO_CACHE_REF_FOR_CHECK="${FD_DINO_CACHE_REF-/scratch/ymbahram/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz}"
      ;;
    *)
      echo "ERROR: unknown DATASET_NAME='$dataset'. Known: caltech101, artbench10, cub200, food101, stanfordcars." >&2
      exit 2
      ;;
  esac
}

check_required_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    return 1
  fi
}

check_dataset_root_or_zip() {
  local dataset="$1"
  local root="$2"
  local zip_path="${root}.zip"
  if [[ -d "$root" || -f "$zip_path" ]]; then
    return 0
  fi
  echo "ERROR: missing dataset root or zip for $dataset: $root or $zip_path" >&2
  return 1
}

check_assets_for_job() {
  local dataset="$1"
  local backbone="$2"
  local missing=0

  case "${ASSET_CHECK,,}" in
    0|false|no|n|off)
      return 0
      ;;
  esac

  set_dataset_assets_for_check "$dataset"
  check_dataset_root_or_zip "$dataset" "$DATASET_ROOT_FOR_CHECK" || missing=1
  check_required_path "FID stats for $dataset" "$FID_CACHE_REF_FOR_CHECK" || missing=1
  if [[ -n "${FD_DINO_CACHE_REF_FOR_CHECK:-}" ]]; then
    check_required_path "FD-DINO stats for $dataset" "$FD_DINO_CACHE_REF_FOR_CHECK" || missing=1
  fi

  case "${backbone,,}" in
    sit)
      LOAD_FROM_FOR_CHECK="${LOAD_FROM:-/scratch/ymbahram/weights/SiT-XL-2-256.pt}"
      ;;
    dit)
      LOAD_FROM_FOR_CHECK="${LOAD_FROM:-/scratch/ymbahram/weights/DiT-XL-2-256x256.pt}"
      ;;
  esac
  check_required_path "$backbone initial checkpoint" "$LOAD_FROM_FOR_CHECK" || missing=1

  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: asset check failed for dataset=$dataset backbone=$backbone. Set ASSET_CHECK=False to bypass intentionally." >&2
    exit 6
  fi
}

append_export_if_set() {
  local var_name="$1"
  if [[ -v "$var_name" ]]; then
    printf 'export %s=%q\n' "$var_name" "${!var_name}"
  fi
}

quote_args() {
  local quoted=""
  local arg
  for arg in "$@"; do
    quoted+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$quoted"
}

write_dataset_job() {
  local dataset="$1"
  local run_label="$2"
  local wandb_name="$3"
  local job_name="plain_${BACKBONE_LABEL}_${dataset}_${SWEEP_LABEL}"
  local job_script="$SLURM_SCRIPT_DIR/${job_name}.sbatch"
  local slurm_stdout="$SLURM_SCRIPT_DIR/${job_name}_%j.out"
  local slurm_stderr="$SLURM_SCRIPT_DIR/${job_name}_%j.err"
  local extra_args_quoted
  extra_args_quoted="$(quote_args "${EXTRA_ARGS[@]}")"
  set_dataset_assets_for_check "$dataset"

  case "${BACKBONE_LABEL}" in
    sit)
      LOAD_FROM_FOR_CHECK="${LOAD_FROM:-/scratch/ymbahram/weights/SiT-XL-2-256.pt}"
      ;;
    dit)
      LOAD_FROM_FOR_CHECK="${LOAD_FROM:-/scratch/ymbahram/weights/DiT-XL-2-256x256.pt}"
      ;;
  esac

  mkdir -p "$SLURM_SCRIPT_DIR"

  {
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=%s\n' "$job_name"
    printf '#SBATCH --account=%s\n' "$SLURM_ACCOUNT"
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --cpus-per-task=%s\n' "$SLURM_CPUS_PER_TASK"
    printf '#SBATCH --mem=%s\n' "$SLURM_MEM"
    printf '#SBATCH --time=%s\n' "$SLURM_TIME_FOR_JOB"
    printf '#SBATCH --gres=%s\n' "$SLURM_GRES"
    printf '#SBATCH --output=%s\n' "$slurm_stdout"
    printf '#SBATCH --error=%s\n' "$slurm_stderr"
    if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
      printf '#SBATCH --mail-user=%s\n' "$SLURM_MAIL_USER"
      printf '#SBATCH --mail-type=%s\n' "$SLURM_MAIL_TYPE"
    fi
    printf '\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_ROOT"
    printf 'module load %q %q\n' "$PYTHON_MODULE" "$CUDA_MODULE"
    printf 'source .venv/bin/activate\n'
    printf 'export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/$USER-matplotlib}\n'
    printf 'export PYTHON=%q\n' ".venv/bin/python"
    printf 'export BACKBONE=%q\n' "$BACKBONE_LABEL"
    printf 'export DATASET_NAME=%q\n' "$dataset"
    printf 'export DATASET_ROOT=%q\n' "$DATASET_ROOT_FOR_CHECK"
    printf 'export FID_CACHE_REF=%q\n' "$FID_CACHE_REF_FOR_CHECK"
    printf 'export FD_DINO_CACHE_REF=%q\n' "$FD_DINO_CACHE_REF_FOR_CHECK"
    printf 'export LOAD_FROM=%q\n' "$LOAD_FROM_FOR_CHECK"
    printf 'export LOG_DIR=%q\n' "$SWEEP_LOG_DIR"
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT_FOR_JOB"
    printf 'export WANDB_NAME=%q\n' "$wandb_name"
    printf 'if [[ ! -d "$DATASET_ROOT" && -f "${DATASET_ROOT}.zip" ]]; then\n'
    printf '  EXTRACT_LOCK="${DATASET_ROOT}.extract.lock"\n'
    printf '  if mkdir "$EXTRACT_LOCK" 2>/dev/null; then\n'
    printf '    echo "Extracting ${DATASET_ROOT}.zip into $DATASET_ROOT"\n'
    printf '    EXTRACT_TMP="${DATASET_ROOT}.extracting.$$"\n'
    printf '    mkdir -p "$EXTRACT_TMP"\n'
    printf '    unzip -q "${DATASET_ROOT}.zip" -d "$EXTRACT_TMP"\n'
    printf '    EXTRACTED_ROOT="$(find "$EXTRACT_TMP" -type d -name "$(basename "$DATASET_ROOT")" | head -n 1)"\n'
    printf '    if [[ -z "$EXTRACTED_ROOT" ]]; then\n'
    printf '      echo "ERROR: could not find $(basename "$DATASET_ROOT") inside ${DATASET_ROOT}.zip" >&2\n'
    printf '      rmdir "$EXTRACT_LOCK" 2>/dev/null || true\n'
    printf '      exit 7\n'
    printf '    fi\n'
    printf '    mv "$EXTRACTED_ROOT" "$DATASET_ROOT"\n'
    printf '    rmdir "$EXTRACT_LOCK" 2>/dev/null || true\n'
    printf '  else\n'
    printf '    echo "Waiting for another job to finish extracting $DATASET_ROOT"\n'
    printf '    while [[ ! -d "$DATASET_ROOT" && -d "$EXTRACT_LOCK" ]]; do sleep 30; done\n'
    printf '  fi\n'
    printf 'fi\n'
    printf 'if [[ ! -d "$DATASET_ROOT" ]]; then\n'
    printf '  echo "ERROR: DATASET_ROOT missing after extraction attempt: $DATASET_ROOT" >&2\n'
    printf '  exit 7\n'
    printf 'fi\n'
    append_export_if_set USE_WANDB
    append_export_if_set RUN_FINAL_BEST_FID_EVAL
    append_export_if_set FINAL_EVAL_STEPS
    append_export_if_set FINAL_EVAL_USE_WANDB
    append_export_if_set SAMPLE_DEVICE_BATCH_SIZE
    append_export_if_set SAMPLE_LOG_EVERY
    append_export_if_set HALF_PRECISION
    append_export_if_set HALF_PRECISION_DTYPE
    append_export_if_set SAMPLING_HALF_PRECISION
    append_export_if_set SAMPLING_HALF_PRECISION_DTYPE
    append_export_if_set MODEL_STR
    append_export_if_set CONFIG_MODE
    append_export_if_set FORCE_FID_PER_STEP
    append_export_if_set METRIC_NUM_STEPS
    printf 'bash %q %q%s\n' "$TRAIN_SCRIPT" "$run_label" "$extra_args_quoted"
  } > "$job_script"

  printf '%s' "$job_script"
}

submit_dataset_job() {
  local dataset="$1"
  local run_label="$2"
  local wandb_name="$3"
  local job_script
  job_script="$(write_dataset_job "$dataset" "$run_label" "$wandb_name")"

  case "${SUBMIT_SLURM,,}" in
    dryrun|dry-run|write|write-only|generate|generate-only)
      printf 'Wrote %s job for %s. Script: %s\n' \
        "$BACKBONE_LABEL" "$dataset" "$job_script" >&2
      printf 'dryrun:%s' "$job_script"
      ;;
    *)
      local job_id
      job_id="$(sbatch --parsable "$job_script")"
      printf 'Submitted %s job for %s as %s. Script: %s\n' \
        "$BACKBONE_LABEL" "$dataset" "$job_id" "$job_script" >&2
      printf '%s' "$job_id"
      ;;
  esac
}

for BACKBONE in $BACKBONES; do
  set_backbone_defaults "$BACKBONE"

  printf "\n--- Backbone: %s ---\n" "$BACKBONE_LABEL"
  printf "Train script: %s\n" "$TRAIN_SCRIPT"
  printf "Wandb project: %s\n" "$WANDB_PROJECT_FOR_JOB"
  printf "Slurm time: %s\n" "$SLURM_TIME_FOR_JOB"

  for DATASET in $DATASETS; do
    check_assets_for_job "$DATASET" "$BACKBONE_LABEL"

    RUN_LABEL="${SWEEP_LABEL}_${DATASET}_${BACKBONE_LABEL}"
    WANDB_NAME="${DATASET}_plain_${BACKBONE_LABEL}_${SWEEP_LABEL}"

    printf "\n=== Running %s on %s ===\n" "$BACKBONE_LABEL" "$DATASET"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tstarted\n" \
      "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT_FOR_JOB" "$WANDB_NAME" "$SLURM_TIME_FOR_JOB" >> "$MANIFEST"

    case "${SUBMIT_SLURM,,}" in
      1|true|yes|y|on|dryrun|dry-run|write|write-only|generate|generate-only)
        JOB_ID="$(submit_dataset_job "$DATASET" "$RUN_LABEL" "$WANDB_NAME")"
        printf "%s\t%s\t%s\t%s\t%s\t%s\tsubmitted_%s\n" \
          "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT_FOR_JOB" "$WANDB_NAME" "$SLURM_TIME_FOR_JOB" "$JOB_ID" >> "$MANIFEST"
        continue
        ;;
    esac

    set +e
    BACKBONE="$BACKBONE_LABEL" \
      DATASET_NAME="$DATASET" \
      LOG_DIR="$SWEEP_LOG_DIR" \
      WANDB_PROJECT="$WANDB_PROJECT_FOR_JOB" \
      WANDB_NAME="$WANDB_NAME" \
      bash "$TRAIN_SCRIPT" "$RUN_LABEL" "${EXTRA_ARGS[@]}"
    STATUS=$?
    set -e

    if [[ "$STATUS" -eq 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\tdone\n" \
        "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT_FOR_JOB" "$WANDB_NAME" "$SLURM_TIME_FOR_JOB" >> "$MANIFEST"
    else
      printf "%s\t%s\t%s\t%s\t%s\t%s\tfailed_%s\n" \
        "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT_FOR_JOB" "$WANDB_NAME" "$SLURM_TIME_FOR_JOB" "$STATUS" >> "$MANIFEST"
      case "${CONTINUE_ON_FAILURE,,}" in
        1|true|yes|y|on)
          echo "WARNING: dataset '$DATASET' failed with status $STATUS; continuing." >&2
          ;;
        *)
          echo "ERROR: dataset '$DATASET' failed with status $STATUS. Manifest: $MANIFEST" >&2
          exit "$STATUS"
          ;;
      esac
    fi
  done
done

echo "Sweep complete. Manifest: $MANIFEST"
