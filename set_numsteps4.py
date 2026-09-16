#!/usr/bin/env python3
"""Set best_fid-selection NFE to 4 (was 16) in the EMA-vc config, so
training-time validation / best_fid checkpoint selection uses 4-step FID
(matches the online DiT baseline's 4-step headline). metric_num_steps is
already [4]. Idempotent, anchored to the sampling: block."""
import io, sys, shutil, time
C = "/opt/dlami/nvme/meanflow/imeanflow/configs/caltech_dit_dmf_ddpmv_ema_config.yml"
with io.open(C, "r", encoding="utf-8") as f:
    lines = f.readlines()
# find "sampling:" then its "num_steps:" child
out, in_samp, done = [], False, False
for ln in lines:
    s = ln.rstrip("\n")
    if s.strip() == "sampling:":
        in_samp = True
        out.append(ln); continue
    if in_samp and not done and s.strip().startswith("num_steps:"):
        indent = ln[:len(ln)-len(ln.lstrip())]
        out.append(f"{indent}num_steps: 4\n")
        done = True
        continue
    # leave sampling block on a dedented top-level key
    if in_samp and s and not s.startswith((" ", "\t")):
        in_samp = False
    out.append(ln)
if not done:
    print("ERROR: sampling.num_steps not found"); sys.exit(1)
new = "".join(out)
with io.open(C, "r", encoding="utf-8") as f:
    if f.read() == new:
        print("ALREADY 4 -- no change"); sys.exit(0)
shutil.copy2(C, C + ".bak_ns4_" + time.strftime("%Y%m%d_%H%M%S"))
with io.open(C, "w", encoding="utf-8") as f:
    f.write(new)
print("PATCHED sampling.num_steps -> 4")
