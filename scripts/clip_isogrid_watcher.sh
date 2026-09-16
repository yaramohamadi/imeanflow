#!/usr/bin/env bash
# CUB-200 grad-clip isolation grid.
# 2 methods (SiT plain diffusion, MF-T meanflow) x 3 objectives (ddpme, ddpmv,
# transportv) x 2 clip settings (0.0 off, 1.0 on) = 12 training runs.
# SiT is evaluated at NFE 4 and 250. MF-T is evaluated at NFE 4 only.
# One GPU per run. The watcher launches a queued cell whenever a GPU is free,
# skipping GPUs that are busy (CP-ablation or anything else). Idempotent: a cell
# whose final-eval CSV already exists is treated as done and never relaunched.
set -uo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
PY=$REPO/.venv/bin/python
LAT=$DATA/cub-200-2011_processed_latents
FID=$REPO/files/fid_stats/cub-200-2011_processed-fid_stats.npz
FDD=$REPO/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz
DIT=$REPO/files/weights/DiT-XL-2-256x256.pt
NC=200
STAMP=20260730
STATE=$REPO/files/logs/isogrid_${STAMP}
mkdir -p "$STATE"
cd "$REPO"

# cell = method:objective:clip
# method in {sit, mft}; objective in {ddpme, ddpmv, transportv}; clip in {0.0, 1.0}
CELLS=(
  sit:ddpme:0.0     sit:ddpme:1.0
  sit:ddpmv:0.0     sit:ddpmv:1.0
  sit:transportv:0.0 sit:transportv:1.0
  mft:ddpme:0.0     mft:ddpme:1.0
  mft:ddpmv:0.0     mft:ddpmv:1.0
  mft:transportv:0.0 mft:transportv:1.0
)

tag() { # method obj clip -> filesystem-safe run tag
  local m=$1 o=$2 c=$3; local ct=off; [ "$c" = "1.0" ] && ct=on
  echo "iso_${m}_${o}_clip${ct}_${STAMP}"
}

# Where a cell's final-eval CSV should land once done (idempotency probe).
done_probe() { # method tag -> path glob that exists only when the cell is fully evaluated
  local m=$1 t=$2
  if [ "$m" = sit ]; then
    # SiT: NFE 250 is the last eval step, so its CSV is the completion marker.
    echo "$STATE/${t}/run/eval_best_fid_250steps/eval_metrics.csv"
  else
    echo "$STATE/${t}/run/eval_best_fid_4steps/eval_metrics.csv"
  fi
}

is_done() { local p; p=$(done_probe "$1" "$2"); ls $p >/dev/null 2>&1; }
is_running() { screen -ls 2>/dev/null | grep -q "\.$1\b"; }

free_gpus() {
  # A GPU is free if it has no compute apps and <2GB used.
  local i
  for i in 0 1 2 3 4 5 6 7; do
    local used procs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null | grep -c . )
    [ -z "$used" ] && continue
    if [ "$used" -lt 2000 ] && [ "$procs" -eq 0 ]; then echo "$i"; fi
  done
}

launch_sit() { # obj clip gpu tag
  local obj=$1 clip=$2 gpu=$3 t=$4
  local mode="caltech_plain_sit_${obj}"
  local wd="$STATE/${t}/run"; mkdir -p "$wd"
  local sh="$STATE/${t}/launch.sh"
  cat > "$sh" <<EOF
#!/bin/bash
set -uo pipefail
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer="
# ---- train ----
$PY $REPO/main_sit.py \\
  --workdir=$wd \\
  --config=$REPO/configs/load_config.py:$mode \\
  --config.training.batch_size=16 \\
  --config.training.grad_accum_steps=2 \\
  --config.training.grad_clip_norm=$clip \\
  --config.training.max_train_steps=30000 \\
  --config.load_from=$DIT \\
  --config.dataset.root=$LAT \\
  --config.dataset.name=cub-200-2011 \\
  --config.dataset.num_workers=0 \\
  --config.dataset.num_classes=$NC \\
  --config.model.num_classes=$NC \\
  --config.fid.cache_ref=$FID \\
  --config.fd_dino.cache_ref=$FDD \\
  --config.logging.use_wandb=False 2>&1 | tee -a $wd/train.log
# ---- final eval: NFE 4 and 250 ----
CONFIG_MODE=$mode PYTHON=$PY USE_WANDB=False \\
  bash $REPO/scripts/eval_best_fid_steps_plain_sit.sh "$wd" 4 250 \\
    -- --config.load_from=$DIT \\
       --config.dataset.num_classes=$NC \\
       --config.model.num_classes=$NC \\
       --config.dataset.name=cub-200-2011 \\
       --config.dataset.root=$LAT \\
       --config.fid.cache_ref=$FID \\
       --config.fd_dino.cache_ref=$FDD 2>&1 | tee -a $wd/eval.log
echo "CELL_DONE $t" | tee -a $wd/status.log
EOF
  chmod +x "$sh"
  screen -dmS "$t" bash -c "bash $sh"
}

launch_mft() { # obj clip gpu tag
  local obj=$1 clip=$2 gpu=$3 t=$4
  local wrapper="$REPO/scripts/run_caltech_dit_dmf_${obj}_taylor.sh"
  local wd_parent="$STATE/${t}"; mkdir -p "$wd_parent"
  local sh="$STATE/${t}/launch.sh"
  # The MF-T wrapper creates its own timestamped workdir under files/logs/finetuning
  # and self-runs final eval. We point LOG_DIR at our state dir so the run and its
  # eval CSV are captured under $STATE/$t, and force NFE-4-only final eval.
  cat > "$sh" <<EOF
#!/bin/bash
set -uo pipefail
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export DATASET_NAME=cub200
export DATASET_ROOT=$LAT
export DATASET_NUM_CLASSES=$NC
export FID_CACHE_REF=$FID
export FD_DINO_CACHE_REF=$FDD
export PYTHON=$PY
export ENABLE_DOGFIT=False
export USE_WANDB=False
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='4' FINAL_EVAL_USE_WANDB=False
export LOG_DIR=$wd_parent
bash $wrapper $t \\
  --config.load_from=$DIT \\
  --config.training.grad_clip_norm=$clip \\
  --config.training.max_train_steps=30000 \\
  --config.sampling.num_steps=4 \\
  --config.training.debug_log_during_train=False 2>&1 | tee -a $wd_parent/train.log
echo "CELL_DONE $t" | tee -a $wd_parent/status.log
EOF
  chmod +x "$sh"
  screen -dmS "$t" bash -c "bash $sh"
}

# The MF-T wrapper puts its run dir under $LOG_DIR/finetuning/<jobname>. Our
# done_probe for mft looks under $STATE/$t/run, so symlink after launch is messy;
# instead, for mft we probe the wrapper's actual output location.
mft_probe() { # tag -> csv path (glob)
  local t=$1
  echo "$STATE/${t}/finetuning/"*"${t}"*"/eval_best_fid_4steps/eval_metrics.csv"
}
is_done_mft() { ls $(mft_probe "$1") >/dev/null 2>&1; }

echo "[isogrid] start $(date '+%F %T')  state=$STATE"
while :; do
  pending=0
  claimed=""   # GPUs handed out this cycle, excluded until they claim memory
  for cell in "${CELLS[@]}"; do
    IFS=':' read -r m o c <<< "$cell"
    t=$(tag "$m" "$o" "$c")
    if [ "$m" = mft ]; then
      is_done_mft "$t" && continue
    else
      is_done "$m" "$t" && continue
    fi
    is_running "$t" && { pending=1; continue; }
    pending=1
    # first free gpu not already claimed this cycle
    g=""
    for cand in $(free_gpus); do
      case " $claimed " in *" $cand "*) continue;; esac
      g=$cand; break
    done
    [ -z "$g" ] && break
    claimed="$claimed $g"
    echo "[isogrid] launch $t on GPU $g  $(date '+%F %T')"
    if [ "$m" = sit ]; then launch_sit "$o" "$c" "$g" "$t"; else launch_mft "$o" "$c" "$g" "$t"; fi
    sleep 90   # let it claim GPU memory before the next free-gpu scan
  done
  [ "$pending" -eq 0 ] && { echo "[isogrid] ALL 12 CELLS DONE $(date '+%F %T')"; break; }
  sleep 120
done
