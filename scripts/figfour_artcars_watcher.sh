#!/usr/bin/env bash
# Figure-4 completion grid: ArtBench + Cars, clip-on only.
# 2 methods (SiT plain diffusion, MF-T meanflow) x 2 datasets (artbench10,
# stanfordcars) x 3 objectives (ddpme, ddpmv, transportv) = 12 runs, all with
# grad clip on (norm 1.0). SiT is evaluated at NFE 4 and 250, MF-T at NFE 4 only.
# One GPU per run, launched when a GPU frees up. Idempotent: a cell whose final
# eval CSV already exists is treated as done and never relaunched.
set -uo pipefail

REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
PY=$REPO/.venv/bin/python
DIT=$REPO/files/weights/DiT-XL-2-256x256.pt
STAMP=20260731
CLIP=1.0
STATE=$REPO/files/logs/figfour_artcars_${STAMP}
mkdir -p "$STATE"
cd "$REPO"

# dataset -> latents, fid stats, fdd stats, num_classes, wrapper dataset name
ds_root() { case "$1" in
  artbench10) echo "$DATA/artbench-10_processed_latents";;
  stanfordcars) echo "$DATA/stanford-cars_processed_latents";; esac; }
ds_fid() { case "$1" in
  artbench10) echo "$REPO/files/fid_stats/artbench-10_processed-fid_stats.npz";;
  stanfordcars) echo "$REPO/files/fid_stats/stanford_cars_processed-fid_stats.npz";; esac; }
ds_fdd() { case "$1" in
  artbench10) echo "$REPO/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz";;
  stanfordcars) echo "$REPO/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz";; esac; }
ds_nc() { case "$1" in artbench10) echo 10;; stanfordcars) echo 196;; esac; }
ds_wrapname() { case "$1" in artbench10) echo artbench10;; stanfordcars) echo stanfordcars;; esac; }

# cell = method:dataset:objective
CELLS=(
  sit:artbench10:ddpme   sit:artbench10:ddpmv   sit:artbench10:transportv
  sit:stanfordcars:ddpme sit:stanfordcars:ddpmv sit:stanfordcars:transportv
  mft:artbench10:ddpme   mft:artbench10:ddpmv   mft:artbench10:transportv
  mft:stanfordcars:ddpme mft:stanfordcars:ddpmv mft:stanfordcars:transportv
)

tag() { echo "ff_${1}_${2}_${3}_clipon_${STAMP}"; }

is_running() { screen -ls 2>/dev/null | grep -q "\.$1\b"; }

# done probes
is_done_sit() { ls "$STATE/$1/run/eval_best_fid_250steps/eval_metrics.csv" >/dev/null 2>&1; }
is_done_mft() { ls "$STATE/$1/finetuning/"*"$1"*"/eval_best_fid_4steps/eval_metrics.csv" >/dev/null 2>&1; }

free_gpus() {
  local i used procs
  for i in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null | grep -c .)
    [ -z "$used" ] && continue
    if [ "$used" -lt 2000 ] && [ "$procs" -eq 0 ]; then echo "$i"; fi
  done
}

launch_sit() { # dataset objective gpu tag
  local ds=$1 obj=$2 gpu=$3 t=$4
  local mode="caltech_plain_sit_${obj}"
  local wd="$STATE/${t}/run"; mkdir -p "$wd"
  local root fid fdd nc; root=$(ds_root "$ds"); fid=$(ds_fid "$ds"); fdd=$(ds_fdd "$ds"); nc=$(ds_nc "$ds")
  local sh="$STATE/${t}/launch.sh"
  cat > "$sh" <<EOF
#!/bin/bash
set -uo pipefail
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_command_buffer="
$PY $REPO/main_sit.py \\
  --workdir=$wd \\
  --config=$REPO/configs/load_config.py:$mode \\
  --config.training.batch_size=16 \\
  --config.training.grad_accum_steps=2 \\
  --config.training.grad_clip_norm=$CLIP \\
  --config.training.max_train_steps=30000 \\
  --config.load_from=$DIT \\
  --config.dataset.root=$root \\
  --config.dataset.name=$ds \\
  --config.dataset.num_workers=0 \\
  --config.dataset.num_classes=$nc \\
  --config.model.num_classes=$nc \\
  --config.fid.cache_ref=$fid \\
  --config.fd_dino.cache_ref=$fdd \\
  --config.logging.use_wandb=False 2>&1 | tee -a $wd/train.log
CONFIG_MODE=$mode PYTHON=$PY USE_WANDB=False \\
  bash $REPO/scripts/eval_best_fid_steps_plain_sit.sh "$wd" 4 250 \\
    -- --config.load_from=$DIT \\
       --config.dataset.num_classes=$nc \\
       --config.model.num_classes=$nc \\
       --config.dataset.name=$ds \\
       --config.dataset.root=$root \\
       --config.fid.cache_ref=$fid \\
       --config.fd_dino.cache_ref=$fdd 2>&1 | tee -a $wd/eval.log
echo "CELL_DONE $t" | tee -a $wd/status.log
EOF
  chmod +x "$sh"; screen -dmS "$t" bash -c "bash $sh"
}

launch_mft() { # dataset objective gpu tag
  local ds=$1 obj=$2 gpu=$3 t=$4
  local wrapper="$REPO/scripts/run_caltech_dit_dmf_${obj}_taylor.sh"
  local wd_parent="$STATE/${t}"; mkdir -p "$wd_parent"
  local root fid fdd nc wn; root=$(ds_root "$ds"); fid=$(ds_fid "$ds"); fdd=$(ds_fdd "$ds"); nc=$(ds_nc "$ds"); wn=$(ds_wrapname "$ds")
  local sh="$STATE/${t}/launch.sh"
  cat > "$sh" <<EOF
#!/bin/bash
set -uo pipefail
cd $REPO
export CUDA_VISIBLE_DEVICES=$gpu
export DATASET_NAME=$wn
export DATASET_ROOT=$root
export DATASET_NUM_CLASSES=$nc
export FID_CACHE_REF=$fid
export FD_DINO_CACHE_REF=$fdd
export PYTHON=$PY
export ENABLE_DOGFIT=False
export USE_WANDB=False
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='4' FINAL_EVAL_USE_WANDB=False
export LOG_DIR=$wd_parent
bash $wrapper $t \\
  --config.load_from=$DIT \\
  --config.training.grad_clip_norm=$CLIP \\
  --config.training.max_train_steps=30000 \\
  --config.sampling.num_steps=4 \\
  --config.training.debug_log_during_train=False 2>&1 | tee -a $wd_parent/train.log
echo "CELL_DONE $t" | tee -a $wd_parent/status.log
EOF
  chmod +x "$sh"; screen -dmS "$t" bash -c "bash $sh"
}

RESERVE=1   # always leave this many GPUs idle for other work
echo "[figfour] start $(date '+%F %T')  state=$STATE  reserve=$RESERVE"
while :; do
  pending=0
  claimed=""
  # snapshot free GPUs once per cycle, drop RESERVE of them from the usable pool
  usable=""
  n=0
  for cand in $(free_gpus); do
    n=$((n+1))
    [ "$n" -le "$RESERVE" ] && continue   # hold the first RESERVE free GPUs back
    usable="$usable $cand"
  done
  for cell in "${CELLS[@]}"; do
    IFS=':' read -r m ds o <<< "$cell"
    t=$(tag "$m" "$ds" "$o")
    if [ "$m" = mft ]; then is_done_mft "$t" && continue
    else is_done_sit "$t" && continue; fi
    is_running "$t" && { pending=1; continue; }
    pending=1
    g=""
    for cand in $usable; do
      case " $claimed " in *" $cand "*) continue;; esac
      g=$cand; break
    done
    [ -z "$g" ] && break
    claimed="$claimed $g"
    echo "[figfour] launch $t on GPU $g  $(date '+%F %T')"
    if [ "$m" = sit ]; then launch_sit "$ds" "$o" "$g" "$t"; else launch_mft "$ds" "$o" "$g" "$t"; fi
    sleep 90
  done
  [ "$pending" -eq 0 ] && { echo "[figfour] ALL 12 CELLS DONE $(date '+%F %T')"; break; }
  sleep 120
done
