#!/usr/bin/env python3
"""Env/config-gated global-norm gradient clipping for the iMF/DMF optimizer.

Patches utils/trainstate_util.py so the bare optax.adamw becomes
optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw) ONLY when
config.training.grad_clip_norm > 0. When the key is absent or 0.0 the built
optimizer is byte-identical to before -> every existing run (SiT/JiT/online/emavc)
is unaffected. clip_by_global_norm runs BEFORE adamw so it caps the raw
(accumulated) gradient global-norm, which is exactly the ddpm-v jvp blow-up.
Idempotent: bails if already patched.
"""
import re, shutil, sys, time

F = "utils/trainstate_util.py"
src = open(F).read()

if "grad_clip_norm" in src:
    print("[patch_gradclip] already patched; no-op")
    sys.exit(0)

old = (
    "    tx = optax.adamw(\n"
    "        learning_rate=lr_fn,\n"
    "        weight_decay=0,\n"
    "        b2=config.training.adam_b2,\n"
    "    )\n"
)
if old not in src:
    print("[patch_gradclip] FATAL: optimizer block not found verbatim", file=sys.stderr)
    sys.exit(2)

new = (
    "    _grad_clip_norm = float(config.training.get(\"grad_clip_norm\", 0.0))\n"
    "    _adamw = optax.adamw(\n"
    "        learning_rate=lr_fn,\n"
    "        weight_decay=0,\n"
    "        b2=config.training.adam_b2,\n"
    "    )\n"
    "    if _grad_clip_norm > 0.0:\n"
    "        tx = optax.chain(optax.clip_by_global_norm(_grad_clip_norm), _adamw)\n"
    "    else:\n"
    "        tx = _adamw\n"
)

bak = f"{F}.bak_gradclip_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(F, bak)
open(F, "w").write(src.replace(old, new, 1))
print(f"[patch_gradclip] patched {F} (backup {bak})")
