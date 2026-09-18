#!/usr/bin/env python3
"""Idempotent patcher: add full-state resume support to CA-iMF (150k->300k).

Edits (in place, on dev5):
  1) train_caimf.py    -- restore FULL CAIMFTrainState from caimf.resume_from
  2) caltech_imf_caimf_posttrain_config.yml -- add `resume_from: ""` key
  3) scripts/run_imf_caimf.sh -- forward RESUME_FROM + MAX_POSTTRAIN_BATCHES env

Backs up each file to <file>.bak_resume before first edit. Safe to re-run.
"""
import os, sys, shutil

ADV = "/opt/dlami/nvme/meanflow/imeanflow_adversarial"
TRAIN = f"{ADV}/train_caimf.py"
CFG = f"{ADV}/configs/caltech_imf_caimf_posttrain_config.yml"
RUN = f"{ADV}/scripts/run_imf_caimf.sh"

def backup(p):
    b = p + ".bak_resume"
    if not os.path.exists(b):
        shutil.copy2(p, b)
        print(f"  backup -> {b}")

def patch(path, old, new, tag):
    with open(path) as f:
        s = f.read()
    if new in s:
        print(f"[SKIP] {tag}: already patched")
        return
    if old not in s:
        print(f"[FAIL] {tag}: anchor NOT found")
        sys.exit(3)
    if s.count(old) != 1:
        print(f"[FAIL] {tag}: anchor appears {s.count(old)}x (need exactly 1)")
        sys.exit(3)
    backup(path)
    with open(path, "w") as f:
        f.write(s.replace(old, new, 1))
    print(f"[OK]   {tag}")

# ---- 1) train_caimf.py: full-state resume block -------------------------
old1 = (
    "    state, gen_tx, dis_tx = _create_state(config, model, discriminator, rng)\n"
    "\n"
    "    distributed = jax.local_device_count() > 1\n"
)
new1 = (
    "    state, gen_tx, dis_tx = _create_state(config, model, discriminator, rng)\n"
    "\n"
    "    # --- Full-state resume (e.g. 150k -> 300k continuation) --------------\n"
    "    # load_from restores generator params ONLY (step reset to 0, fresh\n"
    "    # optimizer + freshly-initialized discriminator). caimf.resume_from\n"
    "    # instead restores the COMPLETE adversarial CAIMFTrainState -- generator\n"
    "    # and EMA params, both optimizer states, the discriminator params/opt,\n"
    "    # and the step/gen_step/dis_step counters -- so training continues\n"
    "    # exactly where it stopped, preserving the D/G equilibrium and the\n"
    "    # post-warmup D:G cadence. Point it at a workdir (latest checkpoint_* is\n"
    "    # chosen) or a specific checkpoint_* directory.\n"
    "    resume_from = str(ca_cfg.get(\"resume_from\", \"\") or \"\")\n"
    "    if resume_from:\n"
    "        resume_from = os.path.abspath(resume_from)\n"
    "        restored = checkpoints.restore_checkpoint(resume_from, state)\n"
    "        restored_step = int(np.asarray(restored.step).reshape(-1)[0])\n"
    "        if restored_step <= 0:\n"
    "            raise ValueError(\n"
    "                f\"caimf.resume_from={resume_from} restored step \"\n"
    "                f\"{restored_step}; expected a full-state checkpoint (step>0). \"\n"
    "                \"Point it at a periodic checkpoint_* dir, not best_fid/.\"\n"
    "            )\n"
    "        state = restored\n"
    "        log_for_0(\n"
    "            \"Resumed FULL CA-iMF state from %s at step %d \"\n"
    "            \"(continuing to max_posttrain_batches=%d).\",\n"
    "            resume_from,\n"
    "            restored_step,\n"
    "            int(ca_cfg.max_posttrain_batches),\n"
    "        )\n"
    "    # ---------------------------------------------------------------------\n"
    "\n"
    "    distributed = jax.local_device_count() > 1\n"
)
patch(TRAIN, old1, new1, "train_caimf.py resume block")

# ---- 2) config: add resume_from key -------------------------------------
old2 = "    max_posttrain_batches: 150000\n"
new2 = "    max_posttrain_batches: 150000\n    resume_from: \"\"  # full-state workdir/checkpoint_* to continue from (empty = fresh warm-start)\n"
patch(CFG, old2, new2, "config resume_from key")

# ---- 3) run_imf_caimf.sh: forward RESUME_FROM + MAX_POSTTRAIN_BATCHES ----
old3 = (
    "cd \"$REPO\"\n"
    "export CUDA_VISIBLE_DEVICES=\"$GPU_LIST\"\n"
)
new3 = (
    "cd \"$REPO\"\n"
    "\n"
    "# Optional full-state resume: continue a prior run to a larger step budget.\n"
    "# RESUME_FROM = a periodic checkpoint_* dir (or workdir) with FULL train\n"
    "# state; MAX_POSTTRAIN_BATCHES overrides the config step cap (e.g. 300000).\n"
    "EXTRA_CFG_ARGS=()\n"
    "if [[ -n \"${RESUME_FROM:-}\" ]]; then\n"
    "  [[ -d \"$RESUME_FROM\" ]] || { echo \"RESUME_FROM dir not found: $RESUME_FROM\" >&2; exit 3; }\n"
    "  EXTRA_CFG_ARGS+=(--config.caimf.resume_from=\"$RESUME_FROM\")\n"
    "  echo \"RESUME (full state) from: $RESUME_FROM\"\n"
    "fi\n"
    "if [[ -n \"${MAX_POSTTRAIN_BATCHES:-}\" ]]; then\n"
    "  EXTRA_CFG_ARGS+=(--config.caimf.max_posttrain_batches=\"$MAX_POSTTRAIN_BATCHES\")\n"
    "  echo \"max_posttrain_batches override: $MAX_POSTTRAIN_BATCHES\"\n"
    "fi\n"
    "export CUDA_VISIBLE_DEVICES=\"$GPU_LIST\"\n"
)
patch(RUN, old3, new3, "run_imf_caimf.sh env forwarding (block A)")

old4 = (
    "  --fd_dino.cache_ref=\"$REPO/files/fdd_stats/$FDD\" \\\n"
    "  --workdir=\"$WORKDIR\"\n"
)
# NOTE: fd_dino flag is actually --config.fd_dino.cache_ref; match real text below.
old4 = (
    "  --config.fd_dino.cache_ref=\"$REPO/files/fdd_stats/$FDD\" \\\n"
    "  --workdir=\"$WORKDIR\"\n"
)
new4 = (
    "  --config.fd_dino.cache_ref=\"$REPO/files/fdd_stats/$FDD\" \\\n"
    "  \"${EXTRA_CFG_ARGS[@]}\" \\\n"
    "  --workdir=\"$WORKDIR\"\n"
)
patch(RUN, old4, new4, "run_imf_caimf.sh env forwarding (block B)")

print("\nALL PATCHES APPLIED.")
