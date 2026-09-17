#!/usr/bin/env bash
# GT-anchored rollout states: Phase 1 and Phase 2 as one detached chain.
#
# The correction to the trajectory arms (EXP-045/046/048): the state is built
# from the TRUE interpolant at t_start = t' - K*dt, which carries the paired x1,
# and then pushed K *inference* steps (Heun + CFG 1.5 at dt = 1/16) along the
# model's own dynamics. The target stays the plain FM target x1 - eps. So the
# endpoint coupling FM relies on is preserved and only the state is model-induced.
#
#   smoke        2 steps on the K=1 inference-rollout path; prints steps/second,
#                which is what the rest of the chain's cost estimate rides on
#   EXP-052      Phase 1 A: K=1, Heun+CFG, dt=1/16, target x1 - eps
#   EXP-053      Phase 1 A-control: the same run with the rollout removed
#                (state = the true interpolant at the same t'), same target
#   EXP-054      Phase 2: K=2
#   EXP-055      Phase 2: K=4
#
# Phase 2 is GATED on EXP-052 not collapsing: if arm A lands anywhere near
# EXP-045's 281.8 the state construction is broken and sweeping the rollout
# distance would only measure the breakage more finely, so the chain stops and
# says so instead of burning 4 GPU-hours.
#
# Every arm is 1000 steps, putting its FID-5k/IS/FD-DINO eval at step 1000 on the
# same schedule as EXP-039/045/047. Kill THIS script's pid to stop the chain;
# never `screen -X quit`, which takes the running trainer with it.
set -u
cd /home/ymohammadibahram/meanflow/imeanflow

# FID above this is "collapsed", not "degraded". The two recorded collapses are
# 281.8 and 284.8 and the two recorded controls are 29.1 and 29.5, so anything in
# between is a wide, uncontroversial line.
COLLAPSE_FID=100

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID (EXP-048 / chain_traj) $(date) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "=== pid $WAIT_PID gone $(date) ==="
fi
while pgrep -f "[m]ain[_]sit[.]py" >/dev/null 2>&1; do sleep 120; done
echo "=== gpu free $(date) ==="

# The rollout solver/omega flags and the `interp` control only exist on origin,
# and the reset is safe now that no trainer is reading the tree.
git fetch -q origin gt-on-policy && git reset -q --hard origin/gt-on-policy
git log --oneline -1

# The rollout step IS the inference step: Heun, CFG 1.5, dt = 1/16, matching
# sampling.{method,omega,num_steps} in the config.
ROLL=( --config.model.sit_gt_on_state=perturb
       --config.model.sit_gt_on_target=fm_velocity
       --config.model.sit_gt_on_lambda=0.0
       --config.model.sit_gt_on_rollout_dt=0.0625
       --config.model.sit_gt_on_rollout_solver=heun
       --config.model.sit_gt_on_rollout_omega=1.5 )

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

fid_of () {  # fid_of <tag> -- step-1000 FID from the newest workdir for that tag
  local w
  w=$(ls -dt files/logs/finetuning/caltech_plain_SiT_GTon_wsl_"$1"_* 2>/dev/null | head -1)
  [ -n "$w" ] && [ -f "$w/eval_metrics.csv" ] || return 1
  awk -F, 'NR>1 && $8 != "" {v=$8} END{if (v=="") exit 1; print v}' "$w/eval_metrics.csv"
}

# --- gate ------------------------------------------------------------------
echo "=== smoke_roll starting $(date) ==="
USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=False \
  bash scripts/run_caltech_plain_sit_gton_wsl.sh smoke_roll \
  --config.training.log_per_step=1 \
  --config.training.max_train_steps=2 \
  "${ROLL[@]}" --config.model.sit_gt_on_rollout_k=1 > ~/smoke_roll.log 2>&1
SMOKE_RC=$?
echo "=== smoke_roll exited $SMOKE_RC at $(date) ==="
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "=== INFERENCE-ROLLOUT PATH FAILS TO RUN -- chain stops, nothing launched ==="
  exit 1
fi
grep -o "steps_per_second=[0-9.]*" ~/smoke_roll.log | tail -2
grep -o "gt_on_drift_rms=[0-9.e+-]*" ~/smoke_roll.log | tail -2

# --- Phase 1 ---------------------------------------------------------------
run gtroll1     "${ROLL[@]}" --config.model.sit_gt_on_rollout_k=1   # EXP-052
run gtroll_ctl  --config.model.sit_gt_on_state=interp \
                --config.model.sit_gt_on_target=fm_velocity \
                --config.model.sit_gt_on_lambda=0.0                 # EXP-053

# --- Phase 2, gated on arm A not collapsing --------------------------------
A_FID=$(fid_of gtroll1) || A_FID=""
echo "=== gate: EXP-052 step-1000 FID = ${A_FID:-<missing>} (collapse if > $COLLAPSE_FID) ==="
if [ -z "$A_FID" ]; then
  echo "=== no FID for gtroll1 -- Phase 2 NOT launched, arm A must be read first ==="
  exit 1
fi
if awk -v a="$A_FID" -v c="$COLLAPSE_FID" 'BEGIN{exit !(a>c)}'; then
  echo "=== EXP-052 COLLAPSED ($A_FID) -- Phase 2 NOT launched, the rollout"
  echo "=== distance sweep would only measure the breakage more finely ==="
  exit 0
fi

run gtroll2  "${ROLL[@]}" --config.model.sit_gt_on_rollout_k=2      # EXP-054
run gtroll4  "${ROLL[@]}" --config.model.sit_gt_on_rollout_k=4      # EXP-055

echo "=== chain_gtroll done $(date) ==="
