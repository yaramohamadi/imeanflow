#!/usr/bin/env python3
"""Run plain-SiT sampling batches from a checkpoint without FID/DINO metrics."""

import argparse
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
from utils.preview_util import make_uint8_image_grid
from utils.sample_util import maybe_cast_state_for_sampling, run_p_sample_step
from utils.sit_sample_util import sample_step
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
        description="Run plain-SiT sampling batches from a checkpoint."
    )
    parser.add_argument("checkpoint_or_run_dir")
    parser.add_argument("--config-mode", default="caltech_plain_sit_finetune")
    parser.add_argument("--workdir", default="files/debug/plain_sit_250_once")
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--preview-samples", type=int, default=64)
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
    parser.add_argument("--output", default="samples_250_once.png")
    return parser.parse_args()


def main():
    supress_checkpt_info()
    args = parse_args()

    if args.device_batch_size <= 0:
        raise ValueError("--device-batch-size must be positive.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.preview_samples < 0:
        raise ValueError("--preview-samples must be non-negative.")

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
    log_for_0("device_batch_size: %d", args.device_batch_size)
    global_batch_size = args.device_batch_size * jax.device_count()
    num_batches = int(np.ceil(args.num_samples / global_batch_size))
    total_generated = num_batches * global_batch_size
    log_for_0("global_batch_size: %d", global_batch_size)
    log_for_0("target samples: %d", args.num_samples)
    log_for_0("batches: %d", num_batches)
    log_for_0("padded generated samples: %d", total_generated)
    log_for_0("sampling.num_steps: %d", args.num_steps)
    log_for_0("sampling.method: %s", config.sampling.get("method", "euler"))
    log_for_0("sampling.omega: %.4f", float(config.sampling.omega))
    log_for_0("training.half_precision: %s", config.training.half_precision)
    log_for_0(
        "training.half_precision_dtype: %s",
        config.training.get("half_precision_dtype", "float16"),
    )

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
            sample_step,
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

    preview_chunks = []
    kept_preview = 0
    total_device_seconds = 0.0
    total_transfer_seconds = 0.0
    total_wall_start = time.time()

    log_for_0("Note: the first sample may be significantly slower due to compilation.")
    for batch_idx in range(num_batches):
        sample_idx = local_sample_idx + batch_idx * jax.device_count()
        should_log = (
            batch_idx == 0
            or batch_idx + 1 == num_batches
            or batch_idx % args.log_every == 0
        )
        if should_log:
            log_for_0("Sampling step %d / %d...", batch_idx, num_batches)

        sample_start = time.time()
        samples = run_p_sample_step(
            p_sample_step,
            state,
            sample_idx=sample_idx,
            latent_manager=latent_manager,
            ema=use_ema,
            **kwargs,
        )
        samples.block_until_ready()
        device_seconds = time.time() - sample_start
        total_device_seconds += device_seconds

        transfer_start = time.time()
        samples = np.asarray(jax.device_get(samples))
        transfer_seconds = time.time() - transfer_start
        total_transfer_seconds += transfer_seconds

        samples_done = min((batch_idx + 1) * global_batch_size, args.num_samples)
        if should_log:
            log_for_0(
                "Sampling batch %d / %d timing: device %.2fs, device_get %.2fs, samples %d / %d",
                batch_idx,
                num_batches,
                device_seconds,
                transfer_seconds,
                samples_done,
                args.num_samples,
            )

        if kept_preview < args.preview_samples:
            take = min(args.preview_samples - kept_preview, len(samples))
            preview_chunks.append(samples[:take])
            kept_preview += take

    total_wall_seconds = time.time() - total_wall_start

    if jax.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)
        if preview_chunks:
            preview_samples = np.concatenate(preview_chunks, axis=0)
            grid_size = int(np.floor(np.sqrt(len(preview_samples))))
        else:
            preview_samples = None
            grid_size = 0
        if grid_size > 0 and preview_samples is not None:
            grid = make_uint8_image_grid(
                preview_samples[: grid_size * grid_size],
                grid_size,
            )
            output_path = os.path.join(args.workdir, args.output)
            Image.fromarray(grid).save(output_path)
            log_for_0("Saved sample grid: %s", output_path)
        log_for_0(
            "Sampling run done: target_samples=%d, generated_samples=%d, batches=%d, device_total %.2fs, device_get_total %.2fs, wall_total %.2fs, samples/sec %.4f",
            args.num_samples,
            total_generated,
            num_batches,
            total_device_seconds,
            total_transfer_seconds,
            total_wall_seconds,
            args.num_samples / total_wall_seconds,
        )


if __name__ == "__main__":
    main()
