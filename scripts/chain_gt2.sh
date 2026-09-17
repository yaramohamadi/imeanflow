#!/usr/bin/env bash
# GT-anchored rollout states, second chain: the three arms that need no new code,
# in the order of what each would change about the conclusion.
#
# Phases 1 and 2 established: the state must carry x1 (that turns a 10x collapse
# into a 7-FID degradation), and past that the rollout distance is a one-way dial
# whose penalty grows as ~drift^2. What Phases 1-2 could NOT separate is WHY the
# residual penalty exists, and they never ran the construction with the plain-FM
# anchor on. These three arms attack exactly that.
#
#   EXP-072  gtchord1  Phase 4. Same states as EXP-052, bit for bit; the ONLY
#                      change is the target: the state-consistent chord
#                      (x1 - xt_hat)/(1 - t') instead of the plain FM x1 - eps.
#                      Separates "the target propagates drift" from "off-manifold
#                      states just compound error". Pairs with EXP-052 (35.304)
#                      and EXP-053 (28.246).
#   EXP-073  gtlam05   lambda=0.5: the plain FM anchor switched on for the first
#                      time on ANY rollout arm. Every arm on file is lambda=0,
#                      i.e. a full objective replacement.
#   EXP-074  gtroll8   Phase 5 proxy, K=8. Tests the extrapolation that drift
#                      near 0.66 reproduces EXP-045's collapse, at a third of the
#                      cost of the full 16-step trajectory.
#
# No result gate between the arms: all three are informative whichever way the
# first two land, and the GPU would otherwise sit idle. There IS a smoke gate in
# front of EXP-074, because K=8 unrolls twice the rollout graph any recorded arm
# has compiled and it is the only arm here whose cost is not already measured.
#
# Kill THIS script's pid to stop the chain; never `screen -X quit`, which takes
# the running trainer with it.
set -u
cd /home/ymohammadibahram/meanflow/imeanflow

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID $(date) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "=== pid $WAIT_PID gone $(date) ==="
fi
while pgrep -f "[m]ain[_]sit[.]py" >/dev/null 2>&1; do sleep 120; done
echo "=== gpu free $(date) ==="

git fetch -q origin gt-on-policy && git reset -q --hard origin/gt-on-policy
git log --oneline -1

# The rollout step IS the inference step: Heun, CFG 1.5, dt = 1/16, matching
# sampling.{method,omega,num_steps}. Identical to chain_gtroll's ROLL so the arms
# below are paired with EXP-052/053/054/055 rather than merely similar.
ROLL=( --config.model.sit_gt_on_state=perturb
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

# --- Phase 4: the target comparison at matched states ----------------------
run gtchord1 "${ROLL[@]}" \
    --config.model.sit_gt_on_target=data \
    --config.model.sit_gt_on_lambda=0.0 \
    --config.model.sit_gt_on_rollout_k=1                    # EXP-072

# --- the anchored version, never run at any K ------------------------------
run gtlam05  "${ROLL[@]}" \
    --config.model.sit_gt_on_target=fm_velocity \
    --config.model.sit_gt_on_lambda=0.5 \
    --config.model.sit_gt_on_rollout_k=1                    # EXP-073

# --- Phase 5 proxy, behind its own 2-step smoke gate -----------------------
echo "=== smoke_roll8 starting $(date) ==="
USE_WANDB=False RUN_FINAL_BEST_FID_EVAL=False \
  bash scripts/run_caltech_plain_sit_gton_wsl.sh smoke_roll8 \
  --config.training.log_per_step=1 \
  --config.training.max_train_steps=2 \
  "${ROLL[@]}" \
  --config.model.sit_gt_on_target=fm_velocity \
  --config.model.sit_gt_on_lambda=0.0 \
  --config.model.sit_gt_on_rollout_k=8 > ~/smoke_roll8.log 2>&1
SMOKE_RC=$?
echo "=== smoke_roll8 exited $SMOKE_RC at $(date) ==="
grep -o "steps_per_second=[0-9.]*" ~/smoke_roll8.log | tail -2
grep -o "gt_on_drift_rms=[0-9.e+-]*" ~/smoke_roll8.log | tail -2
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "=== K=8 DOES NOT RUN (compile or memory) -- EXP-074 not launched ==="
  echo "=== chain_gt2 done $(date) ==="
  exit 1
fi

run gtroll8  "${ROLL[@]}" \
    --config.model.sit_gt_on_target=fm_velocity \
    --config.model.sit_gt_on_lambda=0.0 \
    --config.model.sit_gt_on_rollout_k=8                    # EXP-074

echo "=== chain_gt2 done $(date) ==="
