#!/usr/bin/env python3
"""Sample an untouched ImageNet pMF-H/16 checkpoint into a preview grid."""

import argparse
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from ml_collections import ConfigDict
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plain_pmf import pixelMeanFlow, generate
from utils.sample_util import get_sampling_param_dtype


def _make_config(args):
    config = ConfigDict()
    config.dataset = ConfigDict()
    config.dataset.image_size = 256
    config.dataset.image_channels = 3
    config.dataset.num_classes = 1000

    config.model = ConfigDict()
    config.model.model_str = "pmfDiT_H_16"
    config.model.num_classes = 1000
    config.model.cfg_beta = 1.0
    config.model.cfg_max = 7.0
    config.model.P_mean = 0.0
    config.model.P_std = 0.8
    config.model.noise_scale = 2.0
    config.model.lpips = False
    config.model.convnext = False
    config.model.tr_uniform = True

    config.sampling = ConfigDict()
    config.sampling.num_steps = args.num_steps
    config.sampling.omega = args.cfg_scale
    config.sampling.t_min = args.t_min
    config.sampling.t_max = args.t_max
    config.sampling.half_precision = args.half_precision
    config.sampling.half_precision_dtype = args.half_precision_dtype

    config.training = ConfigDict()
    config.training.half_precision_dtype = args.half_precision_dtype
    return config


def _select_params(restored, ema):
    if isinstance(restored, dict):
        ema_params = restored.get("ema_params")
        if isinstance(ema_params, dict):
            for key in (str(int(ema)), str(ema), ema, float(ema), int(ema)):
                if key in ema_params:
                    print(f"using ema_params[{key!r}]")
                    return ema_params[key]
            print(f"ema_params keys: {list(ema_params.keys())}")
            first_key = next(iter(ema_params))
            print(f"using ema_params[{first_key!r}]")
            return ema_params[first_key]
        if "params" in restored:
            print("using params")
            return restored["params"]
    raise ValueError("Could not find params or ema_params in restored checkpoint.")


def _init_restore_target(model):
    image = jnp.ones((1, 256, 256, 3), dtype=jnp.float32)
    time = jnp.ones((1,), dtype=jnp.float32)
    label = jnp.ones((1,), dtype=jnp.int32)
    print("initializing pMF-H/16 target tree for Orbax restore...")
    variables = jax.jit(model.init)({"params": jax.random.PRNGKey(0)}, image, time, label)
    params = variables["params"]
    target = {
        "step": jnp.array(0, dtype=jnp.int32),
        "params": params,
        "ema_params": {
            "500": params,
            "1000": params,
            "2000": params,
        },
    }
    print("target tree initialized")
    return target


def _cast_params(params, dtype):
    if dtype is None:
        return params
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        params,
    )


def _save_grid(images, output_path):
    images = np.asarray(images)
    num_images, height, width, channels = images.shape
    grid_cols = int(math.ceil(math.sqrt(num_images)))
    grid_rows = int(math.ceil(num_images / grid_cols))
    grid = np.zeros((grid_rows * height, grid_cols * width, channels), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    Image.fromarray(grid).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", default="files/weights/pMF-H-16-full")
    parser.add_argument("--output-dir", default="files/logs/pmf_imagenet_h16_sample")
    parser.add_argument("--ema", type=float, default=1000)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--t-min", type=float, default=0.2)
    parser.add_argument("--t-max", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--half-precision", action="store_true", default=True)
    parser.add_argument("--half-precision-dtype", default="float16")
    args = parser.parse_args()

    if not os.path.isdir(args.load_from):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.load_from}. "
            "Unzip files/weights/pMF-H-16-full.zip first."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _make_config(args)
    model = pixelMeanFlow(**config.model.to_dict(), eval=True)
    print(f"restoring checkpoint: {args.load_from}")
    target = _init_restore_target(model)
    restored = checkpoints.restore_checkpoint(args.load_from, target=target)
    params = _select_params(restored, args.ema)
    params = _cast_params(params, get_sampling_param_dtype(config))

    sample_fn = jax.jit(
        lambda p, r, idx: generate(
            {"params": p},
            model,
            r,
            idx.shape[0],
            config,
            int(config.sampling.num_steps),
            float(config.sampling.omega),
            float(config.sampling.t_min),
            float(config.sampling.t_max),
            sample_idx=idx,
        )
    )

    key = jax.random.PRNGKey(args.seed)
    samples = []
    labels = []
    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        sample_idx = jnp.arange(start, end, dtype=jnp.int32)
        rng = jax.random.fold_in(key, start)
        batch = sample_fn(params, rng, sample_idx)
        batch = jnp.clip(batch, -1.0, 1.0)
        batch = 127.5 * batch + 128.0
        batch = jnp.clip(batch, 0, 255).astype(jnp.uint8)
        samples.append(np.asarray(jax.device_get(batch)))
        labels.extend([int(x) % 1000 for x in np.asarray(sample_idx)])
        print(f"sampled {end}/{args.num_samples}")

    samples = np.concatenate(samples, axis=0)[: args.num_samples]
    output_stem = f"imagenet_pmf_h16_{args.num_steps}step_ema{int(args.ema)}"
    output_path = output_dir / f"{output_stem}_grid.png"
    labels_path = output_dir / f"{output_stem}_labels.txt"
    _save_grid(samples, output_path)
    labels_path.write_text("\n".join(str(x) for x in labels) + "\n")
    print(f"saved grid: {output_path}")
    print(f"saved labels: {labels_path}")


if __name__ == "__main__":
    main()
