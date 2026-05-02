#!/usr/bin/env python3
"""Sample a plain-SiT checkpoint into an image grid and decode the seed noises."""

import argparse
import math
import os
import sys
import time
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from PIL import Image

import train_sit
from configs.load_config import get_config
from utils.logging_util import log_for_0, supress_checkpt_info
from utils.preview_util import make_stacked_grid_panel, make_uint8_image_grid
from utils.sample_util import maybe_cast_state_for_sampling
from utils.sit_sample_util import sample_step_with_initial_noise
from utils.vae_util import LatentManager


def _find_single_checkpoint(parent):
    checkpoints = [
        os.path.join(parent, name)
        for name in os.listdir(parent)
        if name.startswith("checkpoint_")
        and os.path.isdir(os.path.join(parent, name))
    ]
    checkpoints.sort()
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint_* directory under {parent}, "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def resolve_checkpoint(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if path.endswith((".pt", ".pth", ".pth.tar")):
            return path
        raise ValueError(
            "Expected a checkpoint directory, run directory, or raw torch checkpoint "
            f"(.pt/.pth/.pth.tar), got file: {path}"
        )
    if os.path.basename(path).startswith("checkpoint_"):
        return path

    if not os.path.isdir(path):
        raise ValueError(f"Checkpoint/run path does not exist: {path}")

    if os.path.isdir(os.path.join(path, "best_fid")):
        return _find_single_checkpoint(os.path.join(path, "best_fid"))
    if os.path.isdir(os.path.join(path, "latest_eval")):
        return _find_single_checkpoint(os.path.join(path, "latest_eval"))

    return _find_single_checkpoint(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a saved plain-SiT image grid and decoded starting noises."
    )
    parser.add_argument("checkpoint_or_run_dir")
    parser.add_argument("--config-mode", default="plain_sit_finetune")
    parser.add_argument("--workdir", default="files/debug/plain_sit_grid_and_noise")
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument(
        "--half-precision-dtype",
        default="float16",
        choices=("bfloat16", "bf16", "float16", "fp16"),
    )
    parser.add_argument("--samples-output", default="samples_250_grid.png")
    parser.add_argument("--noise-output", default="decoded_seed_noise_grid.png")
    parser.add_argument("--panel-output", default="samples_and_noise_panel.png")
    parser.add_argument("--latents-output", default="sample_and_noise_latents.npz")
    return parser.parse_args()


def _to_uint8_images(decoded_bchw):
    images = decoded_bchw.transpose(0, 2, 3, 1)
    images = 127.5 * images + 128.0
    return np.asarray(jnp.clip(images, 0, 255).astype(jnp.uint8))


def _save_grid(images_uint8, output_path):
    grid_size = int(math.isqrt(len(images_uint8)))
    if grid_size * grid_size != len(images_uint8):
        raise ValueError(
            f"--num-images must be a perfect square for grid saving, got {len(images_uint8)}."
        )
    grid = make_uint8_image_grid(images_uint8, grid_size)
    Image.fromarray(grid).save(output_path)


def main():
    supress_checkpt_info()
    args = parse_args()

    if args.device_batch_size <= 0:
        raise ValueError("--device-batch-size must be positive.")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    grid_size = int(math.isqrt(args.num_images))
    if grid_size * grid_size != args.num_images:
        raise ValueError("--num-images must be a perfect square, e.g. 16, 25, 36, 64.")

    checkpoint_dir = resolve_checkpoint(args.checkpoint_or_run_dir)
    config = get_config(args.config_mode)
    config.eval_only = True
    config.partial_load = False
    config.load_from = checkpoint_dir
    config.logging.use_wandb = False
    config.sampling.num_steps = args.num_steps
    config.fid.sample_device_batch_size = args.device_batch_size
    if args.half_precision:
        config.training.half_precision = True
        config.training.half_precision_dtype = args.half_precision_dtype
    if args.omega is not None:
        config.sampling.omega = args.omega
    if args.t_min is not None:
        config.sampling.t_min = args.t_min
    if args.t_max is not None:
        config.sampling.t_max = args.t_max

    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = train_sit.infer_num_classes_from_latents(
            config.dataset.root
        )
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes

    image_size = config.dataset.image_size
    use_ema = config.training.get("use_ema", True)
    model = train_sit._build_plain_sit(config, eval_mode=True)

    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("Checkpoint: %s", checkpoint_dir)
    log_for_0("num_images: %d", args.num_images)
    log_for_0("device_batch_size: %d", args.device_batch_size)
    log_for_0("sampling.num_steps: %d", args.num_steps)
    log_for_0("sampling.method: %s", config.sampling.get("method", "euler"))
    log_for_0("sampling.omega: %.4f", float(config.sampling.omega))

    restore_start = time.time()
    state = train_sit._restore_eval_state(config, model, image_size, use_ema)
    state = jax_utils.replicate(state)
    state = maybe_cast_state_for_sampling(state, config)
    log_for_0("Restore/replicate/cast time: %.2fs", time.time() - restore_start)

    latent_manager = LatentManager(
        config.dataset.vae,
        args.device_batch_size,
        image_size,
    )
    p_sample_step = jax.pmap(
        partial(
            sample_step_with_initial_noise,
            model=model,
            rng_init=jax.random.PRNGKey(args.seed),
            config=config,
            device_batch_size=args.device_batch_size,
            num_steps=args.num_steps,
        ),
        axis_name="batch",
    )

    local_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
        jax.local_device_count(),
        dtype=jnp.int32,
    )
    kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.omega),
            "t_min": float(config.sampling.t_min),
            "t_max": float(config.sampling.t_max),
        }
    )

    target_count = args.num_images
    batch_size_global = args.device_batch_size * jax.device_count()
    num_batches = int(np.ceil(target_count / batch_size_global))
    initial_latents_all = []
    final_latents_all = []

    log_for_0("Note: the first sample may be significantly slower due to compilation.")
    for batch_idx in range(num_batches):
        sample_idx = local_sample_idx + batch_idx * jax.device_count()
        log_for_0("Sampling batch %d / %d...", batch_idx + 1, num_batches)
        batch_start = time.time()

        params = state.ema_params if use_ema else state.params
        variable = {"params": params}
        initial_latents, final_latents = p_sample_step(
            variable,
            sample_idx=sample_idx,
            **kwargs,
        )
        initial_latents.block_until_ready()
        final_latents.block_until_ready()
        log_for_0("Batch %d sampling time: %.2fs", batch_idx + 1, time.time() - batch_start)

        initial_latents = np.asarray(jax.device_get(initial_latents)).reshape(
            -1, *initial_latents.shape[2:]
        )
        final_latents = np.asarray(jax.device_get(final_latents)).reshape(
            -1, *final_latents.shape[2:]
        )
        initial_latents_all.append(initial_latents)
        final_latents_all.append(final_latents)

    initial_latents_bchw = np.concatenate(initial_latents_all, axis=0)[:target_count]
    final_latents_bchw = np.concatenate(final_latents_all, axis=0)[:target_count]

    log_for_0("Decoding initial noise latents...")
    initial_images_uint8 = _to_uint8_images(latent_manager.decode(initial_latents_bchw))
    log_for_0("Decoding final sampled latents...")
    final_images_uint8 = _to_uint8_images(latent_manager.decode(final_latents_bchw))

    if jax.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)

        samples_path = os.path.join(args.workdir, args.samples_output)
        noise_path = os.path.join(args.workdir, args.noise_output)
        panel_path = os.path.join(args.workdir, args.panel_output)
        latents_path = os.path.join(args.workdir, args.latents_output)

        _save_grid(final_images_uint8, samples_path)
        _save_grid(initial_images_uint8, noise_path)

        panel = make_stacked_grid_panel(
            {
                "final samples": final_images_uint8,
                "decoded seed noise": initial_images_uint8,
            },
            grid_size,
        )
        Image.fromarray(panel).save(panel_path)

        np.savez(
            latents_path,
            initial_noise_latents_bchw=initial_latents_bchw,
            final_sample_latents_bchw=final_latents_bchw,
            num_steps=np.int32(args.num_steps),
            seed=np.int32(args.seed),
            omega=np.float32(config.sampling.omega),
            t_min=np.float32(config.sampling.t_min),
            t_max=np.float32(config.sampling.t_max),
        )

        log_for_0("Saved sample grid: %s", samples_path)
        log_for_0("Saved decoded noise grid: %s", noise_path)
        log_for_0("Saved comparison panel: %s", panel_path)
        log_for_0("Saved raw latents: %s", latents_path)


if __name__ == "__main__":
    main()
