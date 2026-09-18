#!/usr/bin/env bash
# GT-anchored rollout states, third chain: the two plan items that needed code.
#
#   EXP-075  gtcmix    Experiment C. Roll out K=4 and supervise ONE uniformly
#                      drawn state per example (index j in {1..4}), with the
#                      supervision time following the chosen state. The mixture
#                      contains the fixed-K arm, so it is comparable to EXP-055.
#   EXP-076  gtbnoise  Phase 3, noise-end band  t' in [0, 0.27],   K=1 rollout
#   EXP-077  gtbnoisec   ... and its paired control, rollout removed
#   EXP-078  gtbmid    Phase 3, middle band     t' in [0.27, 0.54], K=1 rollout
#   EXP-079  gtbmidc     ... and its paired control
#   EXP-080  gtbdata   Phase 3, data-end band   t' in [0.54, 0.8],  K=1 rollout
#   EXP-081  gtbdatac    ... and its paired control
#
# Each band is run WITH its control because a band changes the base difficulty on
# its own -- EXP-053 measured ~0.8 FID from truncation alone -- so a band arm's
# raw FID says nothing. The quantity Phase 3 is about is the within-band arm minus
# control difference, which is why the pairs are adjacent here: stopping the chain
# early leaves complete pairs rather than orphaned arms.
#
# Two smoke gates, because both features are new on GPU even though the CPU smoke
# test passes all 15 arms: one for the per-example index, one for the t' band.
#
# Kill THIS script's pid to stop the chain; never `screen -X quit`.
set -u
cd /home/ymohammadibahram/meanflow/imeanflow

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID (chain_gt2) $(date) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "=== pid $WAIT_PID gone $(date) ==="
fi
while pgrep -f "[m]ain[_]sit[.]py" >/dev/null 2>&1; do sleep 120; done
echo "=== gpu free $(date) ==="

git fetch -q origin gt-on-policy && git reset -q --hard origin/gt-on-policy
git log --oneline -1

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

smoke () {  # smoke <tag> <extra flags...> -- 2 steps, returns the trainer's rc
  local tag="$1"; shift
  echo "=== $tag starting $(date) ==="
  USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=False \
    bash scripts/run_caltech_plain_sit_gton_wsl.sh "$tag" \
    --config.training.log_per_step=1 \
    --config.training.max_train_steps=2 \
    "$@" > ~/"$tag".log 2>&1
  local rc=$?
  echo "=== $tag exited $rc at $(date) ==="
  grep -o "steps_per_second=[0-9.]*" ~/"$tag".log | tail -1
  grep -o "gt_on_drift_rms=[0-9.e+-]*" ~/"$tag".log | tail -1
  grep -o "t_prime_mean=[0-9.e+-]*" ~/"$tag".log | tail -1
  return $rc
}

# --- Experiment C, behind its own gate -------------------------------------
if ! smoke smoke_cmix "${ROLL[@]}" \
     --config.model.sit_gt_on_rollout_k=4 \
     --config.model.sit_gt_on_rollout_index_random=True; then
  echo "=== per-example rollout index FAILS ON GPU -- chain stops, nothing run ==="
  exit 1
fi
run gtcmix "${ROLL[@]}" \
    --config.model.sit_gt_on_rollout_k=4 \
    --config.model.sit_gt_on_rollout_index_random=True      # EXP-075

# --- Phase 3, behind its own gate ------------------------------------------
if ! smoke smoke_band "${ROLL[@]}" \
     --config.model.sit_gt_on_rollout_k=1 \
     --config.model.sit_gt_on_t_min=0.54 \
     --config.model.sit_gt_on_t_max=0.8; then
  echo "=== the t' band FAILS ON GPU -- Phase 3 not launched ==="
  echo "=== chain_gt3 done $(date) ==="
  exit 1
fi

band () {  # band <name> <t_min> <t_max> -- the arm and its paired control
  local n="$1" lo="$2" hi="$3"
  run "gtb$n"  "${ROLL[@]}" --config.model.sit_gt_on_rollout_k=1 \
      --config.model.sit_gt_on_t_min="$lo" --config.model.sit_gt_on_t_max="$hi"
  run "gtb${n}c" --config.model.sit_gt_on_state=interp \
      --config.model.sit_gt_on_target=fm_velocity \
      --config.model.sit_gt_on_lambda=0.0 \
      --config.model.sit_gt_on_t_min="$lo" --config.model.sit_gt_on_t_max="$hi"
}

band noise 0.0  0.27    # EXP-076 / EXP-077
band mid   0.27 0.54    # EXP-078 / EXP-079
band data  0.54 0.8     # EXP-080 / EXP-081

echo "=== chain_gt3 done $(date) ==="
