#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  SUBMIT_SLURM=DryRun METHODS="afm caimf" DATASETS="artbench10 caltech101 cub200 food101 stanfordcars" \
    bash scripts/submit_finetuned_afm_caimf_slurm.sh <run_label> [extra config overrides]

Set SUBMIT_SLURM=True to actually submit with sbatch. DryRun only writes sbatch files.

Common knobs:
  METHODS="afm caimf"
  DATASETS="artbench10 caltech101 cub200 food101 stanfordcars"
  SOURCE_MODE=target_ft             # target_ft or original_imf
  ORIGINAL_IMF_CHECKPOINT=/path/to/iMF-XL-2
  AFM_ABLATION=D
  LAMBDA_ANCHOR=0.1
  AFM_MAX_POSTTRAIN_BATCHES=40000
  CAIMF_EXPERIMENT=5
  CAIMF_MAX_POSTTRAIN_BATCHES=120000
  RUN_FINAL_BEST_FID_EVAL=True
  FINAL_EVAL_STEPS="1 2"
  USE_WANDB=True
  SLURM_ACCOUNT=rrg-josedolz
  SLURM_GRES=gpu:h100:1
  SLURM_TIME_AFM=10:00:00
  SLURM_TIME_CAIMF=18:00:00
EOF
  exit 1
fi

RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

REPO="${REPO:-/home/zahradt/projects/def-hadi87/zahradt/imeanflow}"
METHODS="${METHODS:-afm caimf}"
DATASETS="${DATASETS:-artbench10 caltech101 cub200 food101 stanfordcars}"
SUBMIT_SLURM="${SUBMIT_SLURM:-DryRun}"
SOURCE_MODE="${SOURCE_MODE:-target_ft}"
ORIGINAL_IMF_CHECKPOINT="${ORIGINAL_IMF_CHECKPOINT:-$REPO/files/weights/iMF-XL-2}"
case "${SOURCE_MODE,,}" in
  target_ft|target_finetuned)
    SOURCE_LOAD_GENERATOR_EMA=False
    ;;
  original_imf|imagenet_imf|imagenet)
    SOURCE_LOAD_GENERATOR_EMA=True
    ;;
  *)
    echo "ERROR: SOURCE_MODE must be target_ft or original_imf, got '$SOURCE_MODE'." >&2
    exit 2
    ;;
esac
NOW="$(date '+%Y%m%d_%H%M%S')"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-$REPO/files/logs/sweeps/finetuned_afm_caimf_${RUN_LABEL}_${NOW}}"
SLURM_SCRIPT_DIR="$SWEEP_LOG_DIR/slurm"
MANIFEST="$SWEEP_LOG_DIR/manifest.tsv"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-rrg-josedolz}"
SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
SLURM_MEM="${SLURM_MEM:-96G}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
SLURM_TIME_AFM="${SLURM_TIME_AFM:-10:00:00}"
SLURM_TIME_CAIMF="${SLURM_TIME_CAIMF:-18:00:00}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-zahra.dehghani.t@gmail.com}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-END,FAIL}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
USE_WANDB="${USE_WANDB:-False}"
AFM_ABLATION="${AFM_ABLATION:-D}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-0.1}"
AFM_MAX_POSTTRAIN_BATCHES="${AFM_MAX_POSTTRAIN_BATCHES:-40000}"
CAIMF_EXPERIMENT="${CAIMF_EXPERIMENT:-5}"
CAIMF_MAX_POSTTRAIN_BATCHES="${CAIMF_MAX_POSTTRAIN_BATCHES:-120000}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}"

mkdir -p "$SLURM_SCRIPT_DIR"
printf "dataset\tmethod\tcheckpoint\tdataset_root\tfid_stats\tfd_dino_stats\tslurm_script\tstatus\n" > "$MANIFEST"

quote_args() {
  local quoted=""
  local arg
  for arg in "$@"; do
    quoted+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$quoted"
}

dataset_assets() {
  case "$1" in
    artbench10|artbench-10)
      DATASET_SLUG="artbench10"
      CHECKPOINT="$REPO/files/iMF/plain_iMF_finetune_artbench10_h100_20260722_225848_jt6v93/best_fid/checkpoint_7500"
      DATASET_ROOT="/scratch/zahradt/datasets/artbench-10_processed_latents"
      FID_CACHE_REF="$REPO/files/fid_stats/artbench-10_processed-fid_stats.npz"
      FD_DINO_CACHE_REF="$REPO/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz"
      ;;
    caltech101|caltech-101)
      DATASET_SLUG="caltech101"
      CHECKPOINT="$REPO/files/iMF/plain_iMF_finetune_caltech101_h100_20260722_222718_cwbunp/best_fid/checkpoint_32500"
      DATASET_ROOT="/scratch/zahradt/datasets/caltech-101_processed_latents"
      FID_CACHE_REF="$REPO/files/fid_stats/caltech-101-fid_stats.npz"
      FD_DINO_CACHE_REF="$REPO/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz"
      ;;
    cub200|cub-200|cub-200-2011)
      DATASET_SLUG="cub200"
      CHECKPOINT="$REPO/files/iMF/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/best_fid/checkpoint_35000"
      DATASET_ROOT="/scratch/zahradt/datasets/cub-200-2011_processed_latents"
      FID_CACHE_REF="$REPO/files/fid_stats/cub-200-2011_processed-fid_stats.npz"
      FD_DINO_CACHE_REF="$REPO/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz"
      ;;
    food101|food-101)
      DATASET_SLUG="food101"
      CHECKPOINT="$REPO/files/iMF/plain_iMF_finetune_food101_h100_20260722_231935_s9moo7/best_fid/checkpoint_17500"
      DATASET_ROOT="/scratch/zahradt/datasets/food-101_processed_latents"
      FID_CACHE_REF="$REPO/files/fid_stats/food-101_processed-fid_stats.npz"
      FD_DINO_CACHE_REF="$REPO/files/fdd_stats/food-101-fd_dino-vitb14_stats.npz"
      ;;
    stanfordcars|stanford-cars|cars)
      DATASET_SLUG="stanfordcars"
      CHECKPOINT="$REPO/files/iMF/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh/best_fid/checkpoint_12500"
      DATASET_ROOT="/scratch/zahradt/datasets/stanford-cars_processed_latents"
      FID_CACHE_REF="$REPO/files/fid_stats/stanford_cars_processed-fid_stats.npz"
      FD_DINO_CACHE_REF="$REPO/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz"
      ;;
    *)
      echo "ERROR: unknown dataset '$1'." >&2
      exit 2
      ;;
  esac

  if [[ "$SOURCE_LOAD_GENERATOR_EMA" == "True" ]]; then
    CHECKPOINT="$ORIGINAL_IMF_CHECKPOINT"
  fi
}

check_assets() {
  local missing=0
  [[ -x "$REPO/.venv/bin/python" ]] || { echo "ERROR: missing env python: $REPO/.venv/bin/python" >&2; missing=1; }
  [[ -e "$CHECKPOINT" ]] || { echo "ERROR: missing checkpoint: $CHECKPOINT" >&2; missing=1; }
  [[ -d "$DATASET_ROOT/train" ]] || { echo "ERROR: missing latent train root: $DATASET_ROOT/train" >&2; missing=1; }
  [[ -f "$FID_CACHE_REF" ]] || { echo "ERROR: missing FID stats: $FID_CACHE_REF" >&2; missing=1; }
  [[ -f "$FD_DINO_CACHE_REF" ]] || { echo "ERROR: missing FD-DINO stats: $FD_DINO_CACHE_REF" >&2; missing=1; }
  [[ "$missing" -eq 0 ]]
}

write_job() {
  local method="$1"
  local extra_args_quoted
  extra_args_quoted="$(quote_args "${EXTRA_ARGS[@]}")"

  local method_lower="${method,,}"
  local train_script
  local job_time
  local job_name
  local job_script
  local method_env=""

  case "$method_lower" in
    afm)
      train_script="scripts/train_finetuned_afm.sh"
      job_time="$SLURM_TIME_AFM"
      method_env="export AFM_ABLATION=$(printf '%q' "$AFM_ABLATION")
export LAMBDA_ANCHOR=$(printf '%q' "$LAMBDA_ANCHOR")
export AFM_MAX_POSTTRAIN_BATCHES=$(printf '%q' "$AFM_MAX_POSTTRAIN_BATCHES")
export AFM_LOAD_GENERATOR_EMA=$(printf '%q' "$SOURCE_LOAD_GENERATOR_EMA")"
      ;;
    caimf)
      train_script="scripts/train_finetuned_caimf.sh"
      job_time="$SLURM_TIME_CAIMF"
      method_env="export CAIMF_EXPERIMENT=$(printf '%q' "$CAIMF_EXPERIMENT")
export CAIMF_MAX_POSTTRAIN_BATCHES=$(printf '%q' "$CAIMF_MAX_POSTTRAIN_BATCHES")
export CAIMF_LOAD_GENERATOR_EMA=$(printf '%q' "$SOURCE_LOAD_GENERATOR_EMA")"
      ;;
    *)
      echo "ERROR: unknown method '$method'. Known: afm, caimf." >&2
      exit 2
      ;;
  esac

  job_name="${method_lower}_${DATASET_SLUG}_${RUN_LABEL}"
  job_script="$SLURM_SCRIPT_DIR/${job_name}.sbatch"

  {
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=%s\n' "$job_name"
    printf '#SBATCH --account=%s\n' "$SLURM_ACCOUNT"
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --ntasks=1\n'
    printf '#SBATCH --cpus-per-task=%s\n' "$SLURM_CPUS_PER_TASK"
    printf '#SBATCH --mem=%s\n' "$SLURM_MEM"
    printf '#SBATCH --time=%s\n' "$job_time"
    printf '#SBATCH --gres=%s\n' "$SLURM_GRES"
    printf '#SBATCH --output=%s/%s_%%j.out\n' "$SLURM_SCRIPT_DIR" "$job_name"
    printf '#SBATCH --error=%s/%s_%%j.err\n' "$SLURM_SCRIPT_DIR" "$job_name"
    if [[ -n "$SLURM_MAIL_USER" ]]; then
      printf '#SBATCH --mail-user=%s\n' "$SLURM_MAIL_USER"
      printf '#SBATCH --mail-type=%s\n' "$SLURM_MAIL_TYPE"
    fi
    printf '\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO"
    printf 'module load %q %q\n' "$PYTHON_MODULE" "$CUDA_MODULE"
    printf 'source .venv/bin/activate\n'
    printf 'export REPO=%q\n' "$REPO"
    printf 'export PYTHON=%q\n' "$REPO/.venv/bin/python"
    printf 'export DATASET_ROOT=%q\n' "$DATASET_ROOT"
    printf 'export FID_CACHE_REF=%q\n' "$FID_CACHE_REF"
    printf 'export FD_DINO_CACHE_REF=%q\n' "$FD_DINO_CACHE_REF"
    printf 'export IMF_CHECKPOINT=%q\n' "$CHECKPOINT"
    printf 'export TARGET_IMF_CHECKPOINT=%q\n' "$CHECKPOINT"
    printf 'export USE_WANDB=%q\n' "$USE_WANDB"
    printf 'export RUN_FINAL_BEST_FID_EVAL=%q\n' "$RUN_FINAL_BEST_FID_EVAL"
    printf 'export FINAL_EVAL_STEPS=%q\n' "$FINAL_EVAL_STEPS"
    printf 'export FINAL_EVAL_USE_WANDB=%q\n' "$FINAL_EVAL_USE_WANDB"
    printf 'export WANDB_PROJECT=%q\n' "${WANDB_PROJECT:-finetuned_afm_caimf}"
    printf 'export WANDB_NAME=%q\n' "${DATASET_SLUG}_${method_lower}_${RUN_LABEL}"
    printf 'export LOG_ROOT=%q\n' "$SWEEP_LOG_DIR/${method_lower}"
    printf '%s\n' "$method_env"
    printf 'export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}\n'
    printf 'bash %q %q %q%s\n' "$train_script" "$DATASET_SLUG" "$RUN_LABEL" "$extra_args_quoted"
  } > "$job_script"

  printf '%s' "$job_script"
}

echo "Repo: $REPO"
echo "Run label: $RUN_LABEL"
echo "Source mode: $SOURCE_MODE"
echo "Methods: $METHODS"
echo "Datasets: $DATASETS"
echo "Submit Slurm: $SUBMIT_SLURM"
echo "Sweep log dir: $SWEEP_LOG_DIR"

for dataset in $DATASETS; do
  dataset_assets "$dataset"
  check_assets

  for method in $METHODS; do
    job_script="$(write_job "$method")"
    status="written"
    case "${SUBMIT_SLURM,,}" in
      true|1|yes|y)
        job_id="$(sbatch --parsable "$job_script")"
        status="submitted:$job_id"
        echo "Submitted $method for $DATASET_SLUG as $job_id"
        ;;
      *)
        echo "DryRun wrote $method for $DATASET_SLUG: $job_script"
        ;;
    esac
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$DATASET_SLUG" "$method" "$CHECKPOINT" "$DATASET_ROOT" "$FID_CACHE_REF" \
      "$FD_DINO_CACHE_REF" "$job_script" "$status" >> "$MANIFEST"
  done
done

echo "Manifest: $MANIFEST"
