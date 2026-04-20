#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/sweep_caltech_imf_finetune_slurm.sh <sweep_label> [extra main.py config overrides...]

Example:
  USE_WANDB=True bash scripts/sweep_caltech_imf_finetune_slurm.sh lr_seed \
    --config.training.max_train_steps=30000

Dry-run without submitting:
  SUBMIT_SLURM=False bash scripts/sweep_caltech_imf_finetune_slurm.sh test

Default sweep grid:
  LR_VALUES="0.0001 0.00005 0.00002"
  SEEDS="42 43 44"
  GUIDANCE_VALUES="7.5"

Useful env knobs:
  SUBMIT_SLURM=True
  SWEEP_LOG_DIR=files/logs/sweeps/caltech_imf_<label>_<time>
  CONFIG_MODE=caltech_sit_dmf_finetune
  TRAIN_SCRIPT=scripts/train_caltech_sit_dmf_finetune.sh
  SLURM_ACCOUNT=def-hadi87
  SLURM_TIME=80:00:00
  SLURM_GRES=gpu:h100:1
  SLURM_MEM=64G
  SLURM_CPUS_PER_TASK=8
  PYTHON_MODULE=python/3.10.13
  CUDA_MODULE=cuda/12.2
  WANDB_PROJECT=imfa_init
  USE_WANDB=True

All extra args are forwarded to scripts/train_caltech_sit_dmf_finetune.sh.
EOF
  exit 1
fi

SWEEP_LABEL="$1"
shift
EXTRA_ARGS=("$@")

NOW="$(date '+%Y%m%d_%H%M%S')"
REPO_ROOT="$(pwd)"

SUBMIT_SLURM="${SUBMIT_SLURM:-True}"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_caltech_sit_dmf_finetune.sh}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-imfa_init}"

LR_VALUES="${LR_VALUES:-0.0001 0.00005 0.00002}"
SEEDS="${SEEDS:-42 43 44}"
GUIDANCE_VALUES="${GUIDANCE_VALUES:-7.5}"

SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-files/logs/sweeps/caltech_imf_${SWEEP_LABEL}_${NOW}}"
SLURM_SCRIPT_DIR="$SWEEP_LOG_DIR/slurm"
MANIFEST="$SWEEP_LOG_DIR/sweep_manifest.tsv"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-def-hadi87}"
SLURM_TIME="${SLURM_TIME:-80:00:00}"
SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-END,FAIL}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"

mkdir -p "$SLURM_SCRIPT_DIR"
printf "run_label\tjob_name\tlearning_rate\tseed\tguidance_scale\twandb_name\tstatus\tjob_id\tsbatch_script\n" > "$MANIFEST"

quote_args() {
  local quoted=""
  local arg
  for arg in "$@"; do
    quoted+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$quoted"
}

append_export_if_set() {
  local var_name="$1"
  if [[ -v "$var_name" ]]; then
    printf 'export %s=%q\n' "$var_name" "${!var_name}"
  fi
}

submit_point() {
  local lr="$1"
  local seed="$2"
  local guidance="$3"
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  local guidance_tag="${guidance//./p}"
  guidance_tag="${guidance_tag//-/m}"
  local run_label="${SWEEP_LABEL}_lr${lr_tag}_seed${seed}_g${guidance_tag}"
  local job_name="caltech_imf_${run_label}"
  local wandb_name="caltech_imf_${run_label}"
  local job_script="$SLURM_SCRIPT_DIR/${job_name}.sbatch"
  local slurm_stdout="$SLURM_SCRIPT_DIR/${job_name}_%j.out"
  local slurm_stderr="$SLURM_SCRIPT_DIR/${job_name}_%j.err"
  local extra_args_quoted
  extra_args_quoted="$(quote_args "${EXTRA_ARGS[@]}")"

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
    if [[ -n "$SLURM_MAIL_USER" ]]; then
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
    printf 'export CONFIG_MODE=%q\n' "$CONFIG_MODE"
    printf 'export LOG_DIR=%q\n' "$SWEEP_LOG_DIR"
    printf 'export USE_WANDB=%q\n' "$USE_WANDB"
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    printf 'export WANDB_NAME=%q\n' "$wandb_name"
    append_export_if_set DATASET_ROOT
    append_export_if_set FID_CACHE_REF
    append_export_if_set FD_DINO_CACHE_REF
    append_export_if_set LOAD_FROM
    append_export_if_set SAMPLE_DEVICE_BATCH_SIZE
    append_export_if_set SAMPLE_LOG_EVERY
    append_export_if_set HALF_PRECISION
    append_export_if_set HALF_PRECISION_DTYPE
    append_export_if_set SAMPLING_HALF_PRECISION
    append_export_if_set SAMPLING_HALF_PRECISION_DTYPE
    append_export_if_set FORCE_FID_PER_STEP
    append_export_if_set METRIC_NUM_STEPS
    printf 'bash %q %q \\\n' "$TRAIN_SCRIPT" "$run_label"
    printf '  --config.training.learning_rate=%q \\\n' "$lr"
    printf '  --config.training.seed=%q \\\n' "$seed"
    printf '  --config.model.fixed_guidance_scale=%q \\\n' "$guidance"
    printf '  --config.sampling.omega=%q \\\n' "$guidance"
    printf '  --config.logging.wandb_project=%q \\\n' "$WANDB_PROJECT"
    printf '  --config.logging.wandb_name=%q%s\n' "$wandb_name" "$extra_args_quoted"
  } > "$job_script"

  local status="dry_run"
  local job_id=""
  case "${SUBMIT_SLURM,,}" in
    1|true|yes|y|on)
      job_id="$(sbatch --parsable "$job_script")"
      status="submitted"
      printf 'Submitted %s as job %s\n' "$job_name" "$job_id" >&2
      ;;
    *)
      printf 'Wrote %s\n' "$job_script" >&2
      ;;
  esac

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_label" "$job_name" "$lr" "$seed" "$guidance" "$wandb_name" "$status" "$job_id" "$job_script" >> "$MANIFEST"
}

cat <<EOF
Caltech iMF fine-tuning sweep
  config: $CONFIG_MODE -> configs/${CONFIG_MODE}_config.yml
  train script: $TRAIN_SCRIPT
  sweep log dir: $SWEEP_LOG_DIR
  slurm time: $SLURM_TIME
  lrs: $LR_VALUES
  seeds: $SEEDS
  guidance scales: $GUIDANCE_VALUES
  submit slurm: $SUBMIT_SLURM
  extra args: ${EXTRA_ARGS[*]:-<none>}
EOF

for lr in $LR_VALUES; do
  for seed in $SEEDS; do
    for guidance in $GUIDANCE_VALUES; do
      submit_point "$lr" "$seed" "$guidance"
    done
  done
done

echo "Sweep manifest: $MANIFEST"
