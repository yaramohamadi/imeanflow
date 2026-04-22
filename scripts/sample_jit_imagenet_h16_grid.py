#!/usr/bin/env python3
"""Sample an untouched ImageNet-class JiT-H checkpoint into a preview grid."""

import argparse
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.load_config import get_config
from train_jit import _build_plain_jit, _restore_eval_state, get_sampling_param_dtype


def _make_config(args):
    config = get_config("plain_jit_finetune")
    config.unlock()
    config.eval_only = True
    config.load_from = args.load_from
    config.partial_load = True

    config.dataset.name = "imgnet_pixels"
    config.dataset.root = ""
    config.dataset.image_size = 256
    config.dataset.image_channels = 3
    config.dataset.num_classes = 1000
    config.dataset.num_classes_from_data = False

    config.model.model_str = "flaxJiT_H_16"
    config.model.num_classes = 1000
    config.model.target_use_null_class = True

    config.sampling.num_classes = 1000
    config.sampling.num_steps = args.num_steps
    config.sampling.method = args.method
    config.sampling.omega = args.cfg_scale
    config.sampling.cfg_scale = args.cfg_scale
    config.sampling.t_min = args.t_min
    config.sampling.t_max = args.t_max
    config.sampling.half_precision = args.half_precision
    config.sampling.half_precision_dtype = args.half_precision_dtype
    return config


def _sample_batch(model, params, labels, rng, config):
    sample_dtype = model.dtype
    batch_size = labels.shape[0]
    sample_shape = (
        batch_size,
        int(config.dataset.image_size),
        int(config.dataset.image_size),
        int(config.dataset.image_channels),
    )
    z = jax.random.normal(rng, sample_shape, dtype=sample_dtype)
    t_steps = jnp.linspace(
        0.0,
        1.0,
        int(config.sampling.num_steps) + 1,
        dtype=sample_dtype,
    )
    variable = {"params": params}
    omega = jnp.asarray(config.sampling.omega, dtype=sample_dtype)
    t_min = jnp.asarray(config.sampling.t_min, dtype=sample_dtype)
    t_max = jnp.asarray(config.sampling.t_max, dtype=sample_dtype)

    def body_fn(i, current):
        t = jnp.full((batch_size,), t_steps[i], dtype=sample_dtype)
        t_next = jnp.full((batch_size,), t_steps[i + 1], dtype=sample_dtype)
        if config.sampling.method == "heun":
            next_sample = model.apply(
                variable,
                current,
                t,
                t_next,
                labels,
                omega,
                t_min,
                t_max,
                method=model.heun_step,
            )
        else:
            next_sample = model.apply(
                variable,
                current,
                t,
                t_next,
                labels,
                omega,
                t_min,
                t_max,
                method=model.euler_step,
            )
        return next_sample.astype(current.dtype)

    return jax.lax.fori_loop(0, int(config.sampling.num_steps), body_fn, z)


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
    parser.add_argument(
        "--load-from",
        default="/home/ens/AT74470/imeanflow/files/weights/JiT-H-16-256.pth",
    )
    parser.add_argument("--output-dir", default="files/logs/jit_imagenet_h16_sample")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--cfg-scale", type=float, default=2.2)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--half-precision", action="store_true", default=True)
    parser.add_argument("--half-precision-dtype", default="float16")
    args = parser.parse_args()

    if not os.path.exists(args.load_from):
        raise FileNotFoundError(args.load_from)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _make_config(args)
    model = _build_plain_jit(config, eval_mode=True)
    state = _restore_eval_state(config, model, use_ema=False)
    params = state.params
    param_dtype = get_sampling_param_dtype(config)
    if param_dtype is not None:
        params = jax.tree_util.tree_map(
            lambda x: x.astype(param_dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            params,
        )

    sample_fn = jax.jit(lambda p, y, r: _sample_batch(model, p, y, r, config))
    key = jax.random.PRNGKey(args.seed)
    labels_key, sample_key = jax.random.split(key)
    labels = jax.random.randint(
        labels_key,
        (args.num_samples,),
        minval=0,
        maxval=1000,
        dtype=jnp.int32,
    )

    samples = []
    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        batch_labels = labels[start:end]
        rng = jax.random.fold_in(sample_key, start)
        batch = sample_fn(params, batch_labels, rng)
        batch = jnp.clip(batch, -1.0, 1.0)
        batch = 127.5 * batch + 128.0
        batch = jnp.clip(batch, 0, 255).astype(jnp.uint8)
        samples.append(np.asarray(jax.device_get(batch)))
        print(f"sampled {end}/{args.num_samples}")

    samples = np.concatenate(samples, axis=0)[: args.num_samples]
    output_stem = f"imagenet_jit_h16_{int(config.sampling.num_steps)}step"
    output_path = output_dir / f"{output_stem}_grid.png"
    labels_path = output_dir / f"{output_stem}_labels.txt"
    _save_grid(samples, output_path)
    labels_path.write_text("\n".join(str(int(x)) for x in np.asarray(labels)) + "\n")
    print(f"saved grid: {output_path}")
    print(f"saved labels: {labels_path}")


if __name__ == "__main__":
    main()
