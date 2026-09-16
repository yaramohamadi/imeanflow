#!/usr/bin/env bash
# Denoising Resampling Forcing: the whole test plan as one detached chain.
#
# Order is by priority, not by cost, so that if the chain is cut short the
# results that survive are the ones the primary question needs:
#   smoke      2 steps on the trajectory path, and the measured steps/second
#   diagA/B    10 logged steps each, no training readout -- the zero-training
#              diagnostic: velocity error and target gap on trajectory states
#              (A) against the analytic interpolant at the same schedule times
#              (B). Paired: same seed, same schedule, same index draw.
#   EXP-045    the proposal exactly as written (target fm_velocity)
#   EXP-047    the time-matched control (state schedule, target fm_velocity):
#              cheap, and it is what makes EXP-045 vs EXP-039 interpretable
#   EXP-046    the same trajectory states with the state-consistent chord
#   EXP-048    the trajectory-portion ablation (index_min=7)
#
# Every arm is 1000 steps, which puts its FID/IS/FD-DINO eval at step 1000 on
# the same schedule as EXP-025/036/039. Kill THIS script's pid to stop the
# chain; never `screen -X quit`, which takes the running trainer with it.
set -u
cd /home/ymohammadibahram/meanflow/imeanflow

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID (EXP-039) $(date) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "=== pid $WAIT_PID gone $(date) ==="
fi
while pgrep -f "[m]ain[_]sit[.]py" >/dev/null 2>&1; do sleep 120; done
echo "=== gpu free $(date) ==="

# The trajectory code only exists on origin, and the reset is safe now that no
# trainer is reading the tree.
git fetch -q origin gt-on-policy && git reset -q --hard origin/gt-on-policy
git log --oneline -1

TRAJ=( --config.model.sit_gt_on_state=trajectory
       --config.model.sit_gt_on_target=fm_velocity
       --config.model.sit_gt_on_lambda=0.0
       --config.model.sit_gt_on_traj_steps=16
       --config.model.sit_gt_on_traj_solver=heun
       --config.model.sit_gt_on_traj_omega=1.5 )

run () {   # run <tag> <extra flags...>
  local tag="$1"; shift
  echo "=== $tag starting $(date) ==="
  USE_WANDB=True RUN_FINAL_BEST_FID_EVAL=False \
    bash scripts/run_caltech_plain_sit_gton_wsl.sh "$tag" \
    --config.training.log_per_step=10 \
    --config.training.max_train_steps=1000 \
    "$@" > ~/"$tag".log 2>&1
  echo "=== $tag exited $? at $(date) ==="
  sleep 60
}

diag () {  # diag <tag> <extra flags...>  -- 10 logged steps, no eval, no wandb
  local tag="$1"; shift
  echo "=== $tag starting $(date) ==="
  USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=False \
    bash scripts/run_caltech_plain_sit_gton_wsl.sh "$tag" \
    --config.training.log_per_step=1 \
    --config.training.max_train_steps=10 \
    "$@" > ~/"$tag".log 2>&1
  echo "=== $tag exited $? at $(date) ==="
  sleep 30
}

# --- gate ------------------------------------------------------------------
# 2 steps is enough to compile the unrolled 12-step rollout and to print
# steps_per_second, which is what the rest of the chain's cost rides on.
echo "=== smoke_traj starting $(date) ==="
USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=False \
  bash scripts/run_caltech_plain_sit_gton_wsl.sh smoke_traj \
  --config.training.log_per_step=1 \
  --config.training.max_train_steps=2 \
  "${TRAJ[@]}" > ~/smoke_traj.log 2>&1
SMOKE_RC=$?
echo "=== smoke_traj exited $SMOKE_RC at $(date) ==="
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "=== TRAJECTORY PATH FAILS TO RUN -- chain stops, nothing else launched ==="
  exit 1
fi
grep -o "steps_per_second=[0-9.]*" ~/smoke_traj.log | tail -2

# --- zero-training diagnostic (EXP-044) ------------------------------------
diag diag_traj "${TRAJ[@]}"
diag diag_sched --config.model.sit_gt_on_state=schedule \
                --config.model.sit_gt_on_target=fm_velocity \
                --config.model.sit_gt_on_lambda=0.0 \
                --config.model.sit_gt_on_traj_steps=16

# --- the arms --------------------------------------------------------------
run traj_fmvel "${TRAJ[@]}"                                    # EXP-045
run sched_ctl  --config.model.sit_gt_on_state=schedule \
               --config.model.sit_gt_on_target=fm_velocity \
               --config.model.sit_gt_on_lambda=0.0 \
               --config.model.sit_gt_on_traj_steps=16          # EXP-047
run traj_chord --config.model.sit_gt_on_state=trajectory \
               --config.model.sit_gt_on_target=data \
               --config.model.sit_gt_on_lambda=0.0 \
               --config.model.sit_gt_on_traj_steps=16 \
               --config.model.sit_gt_on_traj_solver=heun \
               --config.model.sit_gt_on_traj_omega=1.5         # EXP-046
run traj_late  "${TRAJ[@]}" \
               --config.model.sit_gt_on_traj_index_min=7       # EXP-048

echo "=== chain_traj done $(date) ==="
