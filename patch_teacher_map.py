#!/usr/bin/env python3
"""FIX the EMA divergence: teacher_v_cond_fn's fallthrough default (the branch
that fires for the DMF single-head backbone, since _uses_imf_dit_backbone()
returns False when '_DMF' is in model_str) queried the net at RAW t and returned
RAW epsilon -- it skipped _map_target_model_time (flip 1-t + x999) on the input
AND _compute_target_wrapped_velocity (dit_native eps->velocity) on the output.
So the EMA-teacher v_c target was garbage -> runs diverged / went flat.

Fix: make that fallthrough mirror _predict_target_velocity (boundary velocity,
r=t) but via self.net.apply({"params": teacher_param_tree}, ...) instead of the
bound online module. This matches _run_target_backbone's last branch
(self.net(x, t, r, y)) exactly, with the correct time-mapping and output-wrapping.
Idempotent."""
import io, sys, shutil, time

PATH = "/opt/dlami/nvme/meanflow/imeanflow/imf.py"

OLD = (
    "        del omega\n"
    "        return self.net.apply(\n"
    "            {\"params\": teacher_param_tree},\n"
    "            x,\n"
    "            t_batch,\n"
    "            t_batch,\n"
    "            y,\n"
    "        )\n"
)
NEW = (
    "        del omega\n"
    "        # Objective-EMA teacher for the DMF single-head backbone. The online\n"
    "        # conditioned/boundary velocity for this config is produced by\n"
    "        # _predict_target_velocity (u_fn fallthrough, r=t): it maps the model\n"
    "        # time via _map_target_model_time (flip 1-t + x999) and wraps the raw\n"
    "        # epsilon output via _compute_target_wrapped_velocity (dit_native).\n"
    "        # Mirror that here with the EMA params so v_c is in the SAME space as\n"
    "        # the online target (was: raw t + raw epsilon -> garbage v_c).\n"
    "        context = self._prepare_target_prediction_context(x, t, t)\n"
    "        raw_output = self.net.apply(\n"
    "            {\"params\": teacher_param_tree},\n"
    "            context[\"model_x\"],\n"
    "            context[\"model_t\"].reshape(bz).astype(self.dtype),\n"
    "            context[\"model_r\"].reshape(bz).astype(self.dtype),\n"
    "            y,\n"
    "        )\n"
    "        t_scalar = self._batch_scalar(t, bz, dtype=self.dtype)\n"
    "        return self._compute_target_wrapped_velocity(\n"
    "            raw_output,\n"
    "            x.astype(self.dtype),\n"
    "            t_scalar,\n"
    "            context=context,\n"
    "        )\n"
)

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "Mirror that here with the EMA params" in src:
    print("ALREADY PATCHED -- no change")
    sys.exit(0)

if src.count(OLD) != 1:
    print("ERROR: expected exactly 1 anchor match, found %d" % src.count(OLD))
    sys.exit(1)

bak = PATH + ".bak_teachermap_" + time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(PATH, bak)
with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src.replace(OLD, NEW, 1))
print("PATCHED ok; backup at", bak)
