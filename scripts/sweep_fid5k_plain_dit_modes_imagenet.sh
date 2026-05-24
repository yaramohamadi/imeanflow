#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CONFIG_MODE="${CONFIG_MODE:-plain_dit_finetune}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/files/weights/DiT-XL-2-256x256.pt}"
FID_CACHE_REF="${FID_CACHE_REF:-$REPO_ROOT/files/fid_stats/imagenet_256_fid_stats.npz}"
WORKDIR_ROOT="${WORKDIR_ROOT:-$REPO_ROOT/files/debug/fid5k_plain_dit_modes_imagenet}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-16}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_dit_imagenet_fid5k_sweeps}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-fid5k_plain_dit}"
IMF_FID_DEVICE="${IMF_FID_DEVICE:-cpu}"
MODES_ENV="${MODES:-p_sample native_velocity_analytic native_velocity_data}"

STEPS=("${@:-1 2 4 16 32 64 250}")
if [[ $# -eq 0 ]]; then
  STEPS=(1 2 4 16 32 64 250)
fi

read -r -a MODES <<< "$MODES_ENV"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 2
fi

if [[ ! -f "$FID_CACHE_REF" ]]; then
  echo "FID reference stats not found: $FID_CACHE_REF" >&2
  exit 3
fi

mkdir -p "$WORKDIR_ROOT"
SUMMARY_CSV="$WORKDIR_ROOT/summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "mode,num_steps,fid,is,eval_metrics_csv,workdir,checkpoint_path,fid_cache_ref" > "$SUMMARY_CSV"
fi

run_eval() {
  local mode="$1"
  local num_steps="$2"
  local workdir="$WORKDIR_ROOT/${mode}_${num_steps}steps"
  local eval_csv="$workdir/eval_metrics.csv"
  local wandb_name="${WANDB_NAME_PREFIX}_${mode}_${num_steps}steps"
  local extra_args=()

  case "$mode" in
    p_sample)
      extra_args=(
        --config.sampling.method=p_sample
      )
      ;;
    native_velocity_analytic)
      extra_args=(
        --config.sampling.method=native_velocity
        --config.sampling.native_velocity_cfg_space=velocity
        --config.sampling.native_velocity_derivative_mode=analytic
        --config.sampling.native_velocity_sigma_clamp=1e-5
      )
      ;;
    native_velocity_data)
      extra_args=(
        --config.sampling.method=native_velocity
        --config.sampling.native_velocity_cfg_space=velocity
        --config.sampling.native_velocity_derivative_mode=finite_difference
      )
      ;;
    transport_velocity_aligned)
      extra_args=(
        --config.sampling.method=transport_velocity
        --config.sampling.transport_velocity_cfg_space=velocity
        --config.sampling.transport_velocity_time_map=noise_ratio
        --config.sampling.transport_velocity_eps=1e-3
        --config.sampling.transport_velocity_scale_input=True
      )
      ;;
    transport_velocity_diff2flow)
      extra_args=(
        --config.sampling.method=transport_velocity
        --config.sampling.transport_velocity_cfg_space=velocity
        --config.sampling.transport_velocity_time_map=diff2flow
        --config.sampling.transport_velocity_eps=1e-3
        --config.sampling.transport_velocity_scale_input=True
      )
      ;;
    transport_velocity_noalign|transport_velocity_flipped_linear)
      extra_args=(
        --config.sampling.method=transport_velocity
        --config.sampling.transport_velocity_cfg_space=velocity
        --config.sampling.transport_velocity_time_map=flipped_linear
        --config.sampling.transport_velocity_eps=1e-3
        --config.sampling.transport_velocity_scale_input=True
      )
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 4
      ;;
  esac

  mkdir -p "$workdir"
  echo "=== mode=$mode num_steps=$num_steps ==="
  echo "workdir: $workdir"
  echo "IMF_FID_DEVICE: $IMF_FID_DEVICE"
  echo "DEVICE_BATCH_SIZE: $DEVICE_BATCH_SIZE"
  echo "SAMPLE_DEVICE_BATCH_SIZE: $SAMPLE_DEVICE_BATCH_SIZE"

  IMF_FID_DEVICE="$IMF_FID_DEVICE" \
  TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
    XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
    XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} \
    PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
    "$PYTHON" \
      main_dit.py \
      --workdir="$workdir" \
      --config="configs/load_config.py:${CONFIG_MODE}" \
      --config.eval_only=True \
      --config.partial_load=False \
      --config.load_from="$CHECKPOINT_PATH" \
      --config.logging.use_wandb="${USE_WANDB}" \
      --config.logging.wandb_project="${WANDB_PROJECT}" \
      --config.logging.wandb_name="${wandb_name}" \
      --config.logging.wandb_entity="${WANDB_ENTITY}" \
      --config.dataset.num_classes_from_data=False \
      --config.dataset.num_classes=1000 \
      --config.model.num_classes=1000 \
      --config.sampling.num_classes=1000 \
      --config.fid.cache_ref="$FID_CACHE_REF" \
      --config.fd_dino.cache_ref='' \
      --config.fid.num_samples="${NUM_SAMPLES}" \
      --config.fid.num_images_to_log=16 \
      --config.fid.device_batch_size="${DEVICE_BATCH_SIZE}" \
      --config.fid.sample_device_batch_size="${SAMPLE_DEVICE_BATCH_SIZE}" \
      --config.sampling.num_steps="${num_steps}" \
      "${extra_args[@]}" \
      2>&1 | tee "$workdir/output.log"

  if [[ ! -f "$eval_csv" ]]; then
    echo "Missing eval metrics CSV: $eval_csv" >&2
    exit 5
  fi

  local metrics_line
  metrics_line="$(tail -n 1 "$eval_csv")"
  local fid
  local is_score
  fid="$(echo "$metrics_line" | cut -d, -f8)"
  is_score="$(echo "$metrics_line" | cut -d, -f9)"
  echo "${mode},${num_steps},${fid},${is_score},${eval_csv},${workdir},${CHECKPOINT_PATH},${FID_CACHE_REF}" >> "$SUMMARY_CSV"
}

for mode in "${MODES[@]}"; do
  case "$mode" in
    p_sample|native_velocity_analytic|native_velocity_data|transport_velocity_aligned|transport_velocity_diff2flow|transport_velocity_noalign|transport_velocity_flipped_linear)
      ;;
    *)
      echo "Invalid mode in MODES: $mode" >&2
      exit 7
      ;;
  esac
done

for num_steps in "${STEPS[@]}"; do
  if ! [[ "$num_steps" =~ ^[0-9]+$ ]]; then
    echo "Invalid num_steps: $num_steps" >&2
    exit 6
  fi
  for mode in "${MODES[@]}"; do
    run_eval "$mode" "$num_steps"
  done
done

echo "Sweep summary: $SUMMARY_CSV"
