#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: BACKBONE=sit bash scripts/sweep_plain_finetune_datasets.sh <sweep_label> [extra config overrides...]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=sit bash scripts/sweep_plain_finetune_datasets.sh baseline

  CUDA_VISIBLE_DEVICES=0,1 PYTHON=.venv/bin/python USE_WANDB=True \
    BACKBONE=dit DATASETS="caltech101 food101" bash scripts/sweep_plain_finetune_datasets.sh ddpm_baseline \
    --config.training.max_train_steps=10000

  SUBMIT_SLURM=True BACKBONE=sit DATASETS="caltech101 food101" \
    bash scripts/sweep_plain_finetune_datasets.sh baseline

Env knobs:
  BACKBONE=sit                       # sit, dit, or imf
  DATASETS="caltech101 artbench10 cub200 food101 stanfordcars"
  SWEEP_LOG_DIR=files/logs/sweeps/... # default: files/logs/sweeps/plain_<backbone>_<label>_<time>
  WANDB_PROJECT=plain_sit_finetune    # default depends on BACKBONE
  CONTINUE_ON_FAILURE=False           # set True to continue after a dataset fails
  SUBMIT_SLURM=False                  # set True to submit one Slurm job per dataset
  SLURM_ACCOUNT=def-hadi87
  SLURM_GRES=gpu:h100:1               # override if your cluster uses a different H100 GRES name
  SLURM_MEM=48G
  SLURM_CPUS_PER_TASK=8
  SLURM_MAIL_USER=yara.mohammadi-bahram@livia.etsmtl.ca
  SLURM_MAIL_TYPE=BEGIN,END,FAIL
  SLURM_TIME=60:00:00                # default: 60h for SiT, 45h for DiT
  PYTHON_MODULE=python/3.10.13
  CUDA_MODULE=cuda/12.2

All extra args are forwarded to the underlying train script.
EOF
  exit 1
fi

SWEEP_LABEL="$1"
shift
EXTRA_ARGS=("$@")

BACKBONE="${BACKBONE:-sit}"
DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-False}"
SUBMIT_SLURM="${SUBMIT_SLURM:-False}"
NOW=$(date '+%Y%m%d_%H%M%S')
REPO_ROOT="$(pwd)"

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

WANDB_PROJECT="${WANDB_PROJECT:-$DEFAULT_WANDB_PROJECT}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-files/logs/sweeps/plain_${BACKBONE_LABEL}_${SWEEP_LABEL}_${NOW}}"
mkdir -p "$SWEEP_LOG_DIR"

MANIFEST="$SWEEP_LOG_DIR/sweep_manifest.tsv"
printf "dataset\tbackbone\trun_label\twandb_project\twandb_name\tstatus\n" > "$MANIFEST"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-def-hadi87}"
SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
SLURM_MEM="${SLURM_MEM:-48G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_TIME="${SLURM_TIME:-$DEFAULT_SLURM_TIME}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-yara.mohammadi-bahram@livia.etsmtl.ca}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
SLURM_SCRIPT_DIR="$SWEEP_LOG_DIR/slurm"

cat <<EOF
Sweep label: $SWEEP_LABEL
Backbone: $BACKBONE_LABEL
Datasets: $DATASETS
Train script: $TRAIN_SCRIPT
Sweep log dir: $SWEEP_LOG_DIR
Wandb project: $WANDB_PROJECT
Submit Slurm: $SUBMIT_SLURM
Extra args: ${EXTRA_ARGS[*]:-<none>}
EOF

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

submit_dataset_job() {
  local dataset="$1"
  local run_label="$2"
  local wandb_name="$3"
  local job_name="plain_${BACKBONE_LABEL}_${dataset}_${SWEEP_LABEL}"
  local job_script="$SLURM_SCRIPT_DIR/${job_name}.sbatch"
  local slurm_stdout="$SLURM_SCRIPT_DIR/${job_name}_%j.out"
  local slurm_stderr="$SLURM_SCRIPT_DIR/${job_name}_%j.err"
  local extra_args_quoted
  extra_args_quoted="$(quote_args "${EXTRA_ARGS[@]}")"

  mkdir -p "$SLURM_SCRIPT_DIR"

  {
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=%s\n' "$job_name"
    printf '#SBATCH --account=%s\n' "$SLURM_ACCOUNT"
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --cpus-per-task=%s\n' "$SLURM_CPUS_PER_TASK"
    printf '#SBATCH --mem=%s\n' "$SLURM_MEM"
    printf '#SBATCH --time=%s\n' "$SLURM_TIME"
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
    printf 'export LOG_DIR=%q\n' "$SWEEP_LOG_DIR"
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    printf 'export WANDB_NAME=%q\n' "$wandb_name"
    append_export_if_set USE_WANDB
    append_export_if_set DATASET_ROOT
    append_export_if_set FID_CACHE_REF
    append_export_if_set FD_DINO_CACHE_REF
    append_export_if_set LOAD_FROM
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

  local job_id
  job_id="$(sbatch --parsable "$job_script")"
  printf 'Submitted %s job for %s as %s. Script: %s\n' \
    "$BACKBONE_LABEL" "$dataset" "$job_id" "$job_script" >&2
  printf '%s' "$job_id"
}

for DATASET in $DATASETS; do
  RUN_LABEL="${SWEEP_LABEL}_${DATASET}"
  WANDB_NAME="${DATASET}_plain_${BACKBONE_LABEL}_${SWEEP_LABEL}"

  printf "\n=== Running %s on %s ===\n" "$BACKBONE_LABEL" "$DATASET"
  printf "%s\t%s\t%s\t%s\t%s\tstarted\n" \
    "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" >> "$MANIFEST"

  case "${SUBMIT_SLURM,,}" in
    1|true|yes|y|on)
      JOB_ID="$(submit_dataset_job "$DATASET" "$RUN_LABEL" "$WANDB_NAME")"
      printf "%s\t%s\t%s\t%s\t%s\tsubmitted_%s\n" \
        "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" "$JOB_ID" >> "$MANIFEST"
      continue
      ;;
  esac

  set +e
  DATASET_NAME="$DATASET" \
    LOG_DIR="$SWEEP_LOG_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$WANDB_NAME" \
    bash "$TRAIN_SCRIPT" "$RUN_LABEL" "${EXTRA_ARGS[@]}"
  STATUS=$?
  set -e

  if [[ "$STATUS" -eq 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tdone\n" \
      "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" >> "$MANIFEST"
  else
    printf "%s\t%s\t%s\t%s\t%s\tfailed_%s\n" \
      "$DATASET" "$BACKBONE_LABEL" "$RUN_LABEL" "$WANDB_PROJECT" "$WANDB_NAME" "$STATUS" >> "$MANIFEST"
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

echo "Sweep complete. Manifest: $MANIFEST"
