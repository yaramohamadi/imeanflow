p = "scripts/gen_and_launch.sh"
s = open(p).read()

# 1) final sweep 1 2 250 -> 1 2  (4-step already covered by intermediate/best)
old_fe = "export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2 250' FINAL_EVAL_USE_WANDB=False"
new_fe = "export RUN_FINAL_BEST_FID_EVAL=True FINAL_EVAL_STEPS='1 2' FINAL_EVAL_USE_WANDB=False"
assert old_fe in s, "final eval line not found"
s = s.replace(old_fe, new_fe, 1)

# 2) force intermediate FID + best-selection to 4-step: set sampling.num_steps=4
#    (was 16 in config -> primary metric 16). Add the override to the run cmd.
old_run = "  --config.training.force_fid_per_step=2500 \\\n  --config.training.debug_log_during_train=False"
new_run = "  --config.training.force_fid_per_step=2500 \\\n  --config.sampling.num_steps=4 \\\n  --config.training.debug_log_during_train=False"
assert old_run in s, "run cmd tail not found"
s = s.replace(old_run, new_run, 1)

open(p, "w").write(s)
print("patched gen_and_launch: sampling.num_steps=4 (intermediate+best 4-step), FINAL_EVAL_STEPS='1 2'")
