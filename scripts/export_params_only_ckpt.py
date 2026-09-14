"""Re-save a training checkpoint keeping only `params`.

The Caltech plain-SiT checkpoint is 9.4 GB because it also carries `ema_params`
(which are the *pre-training* initialisation for this run, since `use_ema:
false`) and the Adam moments. Post-training only needs `params`, so exporting
just that subtree cuts the bytes that have to cross the network by ~3.5x.

Usage:
    JAX_PLATFORMS=cpu python scripts/export_params_only_ckpt.py SRC_CKPT DST_DIR
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from flax.training import checkpoints


def tree_bytes(tree):
    return sum(int(np.asarray(x).nbytes) for x in jax.tree_util.tree_leaves(tree))


def fingerprint(params):
    """A few leaf statistics, so the transferred copy can be verified."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    out = []
    for path, value in flat[:3]:
        key = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        arr = np.asarray(value, dtype=np.float64)
        out.append((key, arr.shape, float(arr.mean()), float(arr.std())))
    for path, value in flat:
        key = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        if "y_embedder" in key and "embedding" in key:
            arr = np.asarray(value, dtype=np.float64)
            out.append((key, arr.shape, float(arr.mean()), float(arr.std())))
    return out


def main():
    # orbax/tensorstore rejects relative checkpoint paths.
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    restored = checkpoints.restore_checkpoint(src, target=None)
    print("top-level keys:", list(restored.keys()), flush=True)
    for key, value in restored.items():
        leaves = jax.tree_util.tree_leaves(value)
        print(
            f"  {key}: {tree_bytes(value) / 1e9:.3f} GB over {len(leaves)} leaves",
            flush=True,
        )

    params = restored.get("params")
    if params is None:
        raise SystemExit(f"no 'params' subtree in {src}")

    leaves = jax.tree_util.tree_leaves(params)
    print(f"params: {tree_bytes(params) / 1e9:.3f} GB, {len(leaves)} leaves")
    print("dtypes:", sorted({str(np.asarray(x).dtype) for x in leaves}))
    for key, shape, mean, std in fingerprint(params):
        print(f"  FINGERPRINT {key} shape={shape} mean={mean:.6e} std={std:.6e}")

    os.makedirs(dst, exist_ok=True)
    checkpoints.save_checkpoint(dst, {"params": params}, step=step, keep=1, overwrite=True)
    print("saved to", dst, flush=True)


if __name__ == "__main__":
    main()
