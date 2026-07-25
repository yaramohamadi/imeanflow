#!/usr/bin/env bash
# =============================================================================
# Launch all 8 SiT fine-tuning experiments, one per GPU, each in its own screen.
#   4 datasets (cub, food, stanford, artbench) x 2 objectives (ddpmv, transportv)
#
# Each experiment runs in a detached screen named "sit_<gpu>_<exp>" so it
# survives disconnects. Env: single GPU + platform allocator + wandb on.
#
# Usage:
#   bash launch_all_8.sh          # launch all 8
#   screen -ls                    # list running screens
#   screen -r sit_gpu0_cub_ddpmv  # attach to one (Ctrl-A D to detach)
# =============================================================================
set -euo pipefail

REPO_ROOT="/opt/dlami/nvme/meanflow/imeanflow"
cd "${REPO_ROOT}"

# --- Ensure `screen` is available -------------------------------------------
if ! command -v screen >/dev/null 2>&1; then
  echo "screen not found; installing via apt..."
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y screen >/dev/null 2>&1 || true
fi
if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: could not install 'screen'. Aborting." >&2
  exit 1
fi

# GPU -> training script  (index = GPU id)
# NOTE: no trailing comments on these lines — a "#" glued to the value (no
# space) is NOT a comment in a bash array; it becomes part of / splits the
# element. Keep them bare.
SCRIPTS=(
  "train_cub_ddpmv.sh"
  "train_cub_transportv.sh"
  "train_food_ddpmv.sh"
  "train_food_transportv.sh"
  "train_stanford_ddpmv.sh"
  "train_stanford_transportv.sh"
  "train_artbench_ddpmv.sh"
  "train_artbench_transportv.sh"
)

# --- Guard: the array must have exactly 8 entries (one per GPU) --------------
if [[ "${#SCRIPTS[@]}" -ne 8 ]]; then
  echo "ERROR: expected 8 scripts, got ${#SCRIPTS[@]}: ${SCRIPTS[*]}" >&2
  exit 1
fi
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${NGPU}" -lt 8 ]]; then
  echo "ERROR: need 8 GPUs, found ${NGPU}." >&2
  exit 1
fi

# --- Safety: clear any stray training processes (e.g. the earlier canary) ----
echo "Killing any existing main_sit.py processes..."
pkill -9 -f main_sit.py 2>/dev/null || true
sleep 3
REMAIN=$(pgrep -c -f main_sit.py || true)
echo "Remaining main_sit.py procs: ${REMAIN:-0}"

# --- Launch one detached screen per GPU --------------------------------------
for GPU in "${!SCRIPTS[@]}"; do
  SCRIPT="${SCRIPTS[$GPU]}"
  EXP="${SCRIPT#train_}"; EXP="${EXP%.sh}"          # e.g. cub_ddpmv
  SCREEN_NAME="sit_gpu${GPU}_${EXP}"
  LOG="${REPO_ROOT}/files/logs/${SCREEN_NAME}.out"

  # Wipe any dead screen of the same name from a previous run.
  screen -S "${SCREEN_NAME}" -X quit 2>/dev/null || true

  echo "Launching ${SCREEN_NAME} on GPU ${GPU} -> ${SCRIPT}"

  # -dm: start detached. Inside the screen we set the env and run the trainer,
  # tee'ing to a log so output is captured even without attaching.
  screen -dmS "${SCREEN_NAME}" bash -c "\
    cd '${REPO_ROOT}' && \
    CUDA_VISIBLE_DEVICES=${GPU} \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    USE_WANDB=True \
    bash scripts/${SCRIPT} 2>&1 | tee '${LOG}'; \
    ec=\$?; echo \"=== ${SCREEN_NAME} EXITED (code \$ec) ===\"; \
    exec bash"
  sleep 2
done

echo ""
echo "=========================================="
echo "All 8 experiments launched. Screens:"
screen -ls | grep "sit_gpu" || true
echo ""
echo "Attach:  screen -r sit_gpu0_cub_ddpmv   (Ctrl-A then D to detach)"
echo "Logs:    ${REPO_ROOT}/files/logs/sit_gpu*.out"
echo "GPUs:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=========================================="
