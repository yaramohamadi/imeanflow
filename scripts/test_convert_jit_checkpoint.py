"""Smoke-test PyTorch JiT checkpoint conversion into the Flax JiT tree.

This script intentionally does not initialize the full JiT-H Flax model or save
the converted tree by default. It validates the expensive/key-sensitive part:
loading the real .pth file, selecting the checkpoint state, mapping keys, and
transposing tensor layouts into the Flax parameter structure.
"""

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ckpt_util import (
    _convert_torch_jit_state_dict_to_flax,
    _load_torch_checkpoint_state_dict,
)


def _walk_leaves(tree, prefix=()):
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _walk_leaves(value, prefix + (key,))
    else:
        yield "/".join(prefix), tree


def _format_gib(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="files/weights/JiT-H-16-256.pth",
        help="Path to a JiT PyTorch checkpoint.",
    )
    parser.add_argument(
        "--prefer-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer model_ema2/model_ema1 when present.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of converted leaves to print.",
    )
    parser.add_argument(
        "--check-target-shapes",
        action="store_true",
        help="Also initialize the requested Flax JiT model and verify converted leaf shapes.",
    )
    parser.add_argument(
        "--model-str",
        default="flaxJiT_H_16",
        help="Flax JiT model factory to use for --check-target-shapes.",
    )
    args = parser.parse_args()

    source = _load_torch_checkpoint_state_dict(args.checkpoint, prefer_ema=args.prefer_ema)
    converted = _convert_torch_jit_state_dict_to_flax(source)
    leaves = list(_walk_leaves(converted))

    total_bytes = sum(np.asarray(value).nbytes for _, value in leaves)
    print(f"checkpoint: {args.checkpoint}")
    print(f"source tensors: {len(source)}")
    print(f"converted leaves: {len(leaves)}")
    print(f"converted tensor bytes: {_format_gib(total_bytes)}")
    print("sample converted leaves:")
    for path, value in leaves[: args.show]:
        arr = np.asarray(value)
        print(f"  {path}: shape={arr.shape} dtype={arr.dtype}")

    required_paths = {
        "net/x_embedder/proj1/kernel",
        "net/x_embedder/proj2/kernel",
        "net/blocks_0/attn/qkv/_flax_linear/kernel",
        "net/blocks_0/mlp/w12/_flax_linear/kernel",
        "net/final_layer/linear/_flax_linear/kernel",
    }
    converted_paths = {path for path, _ in leaves}
    missing = sorted(required_paths - converted_paths)
    if missing:
        raise SystemExit(f"missing required converted paths: {missing}")

    print("required JiT paths: OK")

    if args.check_target_shapes:
        import jax
        import jax.numpy as jnp
        from flax import serialization

        from plain_jit import PlainJiT

        model = PlainJiT(
            model_str=args.model_str,
            input_size=256,
            in_channels=3,
            num_classes=1000,
            eval=True,
        )
        x = jnp.ones((1, 256, 256, 3), dtype=jnp.float32)
        t = jnp.ones((1,), dtype=jnp.float32)
        y = jnp.ones((1,), dtype=jnp.int32)
        target = serialization.to_state_dict(
            model.init(jax.random.PRNGKey(0), x, t, y)["params"]
        )
        target_shapes = {
            path: value.shape for path, value in _walk_leaves(target)
        }
        mismatches = []
        for path, value in leaves:
            target_shape = target_shapes.get(path)
            if target_shape != value.shape:
                mismatches.append((path, value.shape, target_shape))
        if mismatches:
            print("shape mismatches:")
            for path, converted_shape, target_shape in mismatches[:20]:
                print(f"  {path}: converted={converted_shape} target={target_shape}")
            raise SystemExit(f"{len(mismatches)} converted shapes did not match target")
        print("target shape check: OK")


if __name__ == "__main__":
    main()
