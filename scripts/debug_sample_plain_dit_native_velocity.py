#!/usr/bin/env python3
"""Sample a plain-DiT checkpoint with p_sample, native, or transport velocity."""

import argparse
import math
import os
import sys
import time
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
REPO_FILES_ROOT = os.path.join(REPO_ROOT, "files")
LOCAL_DATASETS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "datasets"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from PIL import Image

import train_dit
from configs.load_config import get_config
from utils.dit_sample_util import sample_step
from utils.logging_util import log_for_0, supress_checkpt_info
from utils.sample_util import (
    get_sample_devices,
    get_sample_local_device_count,
    maybe_cast_state_for_sampling,
)
from utils.vae_util import DiTLatentManager


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
        description=(
            "Generate a plain-DiT sample grid with p_sample, native DDPM velocity, "
            "or linear transport velocity."
        )
    )
    parser.add_argument("checkpoint_or_run_dir", nargs="?", default=None)
    parser.add_argument("--config-mode", default="plain_dit_finetune")
    parser.add_argument("--workdir", default="files/debug/plain_dit_native_velocity")
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument(
        "--method",
        default="native_velocity",
        choices=("native_velocity", "transport_velocity", "p_sample"),
    )
    parser.add_argument(
        "--native-velocity-cfg-space",
        default="epsilon",
        choices=("epsilon", "velocity"),
    )
    parser.add_argument(
        "--native-velocity-derivative-mode",
        default="finite_difference",
        choices=("finite_difference", "analytic"),
    )
    parser.add_argument(
        "--native-velocity-sigma-clamp",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--transport-velocity-cfg-space",
        default="velocity",
        choices=("epsilon", "velocity"),
    )
    parser.add_argument(
        "--transport-velocity-time-map",
        default="noise_ratio",
        choices=("noise_ratio", "flipped_linear", "linear"),
        help="How linear transport time is mapped to the source DiT DDPM timestep.",
    )
    parser.add_argument(
        "--transport-velocity-eps",
        type=float,
        default=1e-3,
        help="Linear transport start time used to avoid the singular t=0 endpoint.",
    )
    parser.set_defaults(transport_velocity_scale_input=True)
    parser.add_argument(
        "--transport-velocity-scale-input",
        dest="transport_velocity_scale_input",
        action="store_true",
        help="Scale linear-transport x_t to the matched DDPM/VP input norm.",
    )
    parser.add_argument(
        "--no-transport-velocity-scale-input",
        dest="transport_velocity_scale_input",
        action="store_false",
        help="Call DiT directly on the linear-transport x_t without VP norm scaling.",
    )
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--label-space",
        default="auto",
        choices=("auto", "dataset", "imagenet1000"),
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--sample-first-device-only", action="store_true")
    parser.add_argument("--sample-num-local-devices", type=int, default=None)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument(
        "--half-precision-dtype",
        default="float16",
        choices=("bfloat16", "bf16", "float16", "fp16"),
    )
    parser.add_argument("--grid-output", default="samples_grid.png")
    parser.add_argument("--latents-output", default="sample_latents.npz")
    return parser.parse_args()


def _maybe_remap_local_path(path):
    if not path:
        return path

    normalized = os.path.abspath(os.path.expanduser(str(path)))
    scratch_prefixes = {
        "/scratch/ymbahram/datasets": LOCAL_DATASETS_ROOT,
        "/scratch/ymbahram/weights": os.path.join(REPO_FILES_ROOT, "weights"),
        "/scratch/ymbahram/fid_stats": os.path.join(REPO_FILES_ROOT, "fid_stats"),
        "/scratch/ymbahram/fdd_stats": os.path.join(REPO_FILES_ROOT, "fdd_stats"),
    }
    for old_prefix, new_prefix in scratch_prefixes.items():
        if normalized.startswith(old_prefix):
            suffix = os.path.relpath(normalized, old_prefix)
            candidate = os.path.abspath(os.path.join(new_prefix, suffix))
            return candidate
    return normalized


def _infer_label_space(label_space_arg, checkpoint_path):
    if label_space_arg != "auto":
        return label_space_arg
    if checkpoint_path.endswith((".pt", ".pth", ".pth.tar")):
        return "imagenet1000"
    return "dataset"


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
    height, width, channels = images_uint8.shape[1:]
    grid = np.zeros(
        (grid_size * height, grid_size * width, channels),
        dtype=np.uint8,
    )
    for idx, image in enumerate(images_uint8):
        row = idx // grid_size
        col = idx % grid_size
        grid[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = image
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

    config = get_config(args.config_mode)
    config.eval_only = True
    config.partial_load = False
    config.logging.use_wandb = False
    config.sampling.num_steps = args.num_steps
    config.sampling.method = args.method
    config.sampling.native_velocity_cfg_space = args.native_velocity_cfg_space
    config.sampling.native_velocity_derivative_mode = (
        args.native_velocity_derivative_mode
    )
    config.sampling.native_velocity_sigma_clamp = args.native_velocity_sigma_clamp
    config.sampling.transport_velocity_cfg_space = args.transport_velocity_cfg_space
    config.sampling.transport_velocity_time_map = args.transport_velocity_time_map
    config.sampling.transport_velocity_eps = args.transport_velocity_eps
    config.sampling.transport_velocity_scale_input = (
        args.transport_velocity_scale_input
    )
    config.fid.sample_device_batch_size = args.device_batch_size
    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root
    if args.sample_first_device_only:
        config.fid.sample_first_device_only = True
    if args.sample_num_local_devices is not None:
        config.fid.sample_num_local_devices = int(args.sample_num_local_devices)
    config.dataset.root = _maybe_remap_local_path(config.dataset.root)
    default_checkpoint = _maybe_remap_local_path(config.load_from)
    checkpoint_dir = resolve_checkpoint(
        args.checkpoint_or_run_dir if args.checkpoint_or_run_dir is not None else default_checkpoint
    )
    config.load_from = checkpoint_dir
    if "cache_ref" in config.fid:
        config.fid.cache_ref = _maybe_remap_local_path(config.fid.cache_ref)
    if "cache_ref" in config.fd_dino:
        config.fd_dino.cache_ref = _maybe_remap_local_path(config.fd_dino.cache_ref)
    if args.half_precision:
        config.sampling.half_precision = True
        config.sampling.half_precision_dtype = args.half_precision_dtype
    if args.omega is not None:
        config.sampling.omega = args.omega
        config.sampling.cfg_scale = args.omega

    label_space = _infer_label_space(args.label_space, checkpoint_dir)
    if label_space == "imagenet1000":
        config.dataset.num_classes_from_data = False
        config.dataset.num_classes = 1000
        config.model.num_classes = 1000
        config.sampling.num_classes = 1000
    elif config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = train_dit.infer_num_classes_from_latents(
            config.dataset.root
        )
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes

    image_size = config.dataset.image_size
    use_ema = config.training.get("use_ema", False)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    model = train_dit._build_plain_dit(config, eval_mode=True)

    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("sampling local device count: %d", sample_local_device_count)
    log_for_0("Checkpoint: %s", checkpoint_dir)
    log_for_0("label_space: %s", label_space)
    log_for_0("model.num_classes: %d", int(config.model.num_classes))
    log_for_0("dataset.root: %s", config.dataset.root)
    log_for_0("num_images: %d", args.num_images)
    log_for_0("device_batch_size: %d", args.device_batch_size)
    log_for_0("sampling.method: %s", args.method)
    log_for_0(
        "sampling.native_velocity_cfg_space: %s",
        config.sampling.native_velocity_cfg_space,
    )
    log_for_0(
        "sampling.native_velocity_derivative_mode: %s",
        config.sampling.native_velocity_derivative_mode,
    )
    log_for_0(
        "sampling.native_velocity_sigma_clamp: %.6g",
        float(config.sampling.native_velocity_sigma_clamp),
    )
    log_for_0(
        "sampling.transport_velocity_cfg_space: %s",
        config.sampling.transport_velocity_cfg_space,
    )
    log_for_0(
        "sampling.transport_velocity_time_map: %s",
        config.sampling.transport_velocity_time_map,
    )
    log_for_0(
        "sampling.transport_velocity_eps: %.6g",
        float(config.sampling.transport_velocity_eps),
    )
    log_for_0(
        "sampling.transport_velocity_scale_input: %s",
        bool(config.sampling.transport_velocity_scale_input),
    )
    log_for_0("sampling.num_steps: %d", args.num_steps)
    log_for_0("sampling.omega: %.4f", float(config.sampling.omega))

    restore_start = time.time()
    state = train_dit._restore_eval_state(config, model, image_size, use_ema)
    state = jax_utils.replicate(state, devices=sample_devices)
    state = maybe_cast_state_for_sampling(state, config)
    log_for_0("Restore/replicate/cast time: %.2fs", time.time() - restore_start)

    latent_manager = DiTLatentManager(
        config.dataset.vae,
        args.device_batch_size,
        image_size,
        decode_num_local_devices=sample_local_device_count,
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
        devices=sample_devices,
    )

    local_sample_idx = jax.process_index() * sample_local_device_count + jnp.arange(
        sample_local_device_count,
        dtype=jnp.int32,
    )
    kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.omega),
            "t_min": float(config.sampling.t_min),
            "t_max": float(config.sampling.t_max),
        },
        devices=sample_devices,
    )

    target_count = args.num_images
    batch_size_global = args.device_batch_size * sample_local_device_count * jax.process_count()
    num_batches = int(np.ceil(target_count / batch_size_global))
    final_latents_all = []

    log_for_0("Note: the first sample may be significantly slower due to compilation.")
    for batch_idx in range(num_batches):
        sample_idx = (
            local_sample_idx
            + batch_idx * sample_local_device_count * jax.process_count()
        )
        log_for_0("Sampling batch %d / %d...", batch_idx + 1, num_batches)
        batch_start = time.time()

        params = state.ema_params if use_ema else state.params
        variable = {"params": params}
        final_latents = p_sample_step(
            variable,
            sample_idx=sample_idx,
            **kwargs,
        )
        final_latents.block_until_ready()
        log_for_0("Batch %d sampling time: %.2fs", batch_idx + 1, time.time() - batch_start)

        final_latents = np.asarray(jax.device_get(final_latents)).reshape(
            -1, *final_latents.shape[2:]
        )
        final_latents_all.append(final_latents)

    final_latents_bchw = np.concatenate(final_latents_all, axis=0)[:target_count]

    log_for_0("Decoding sampled latents...")
    final_images_uint8 = _to_uint8_images(latent_manager.decode(final_latents_bchw))

    if jax.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)
        grid_path = os.path.join(args.workdir, args.grid_output)
        latents_path = os.path.join(args.workdir, args.latents_output)
        _save_grid(final_images_uint8, grid_path)
        np.savez(
            latents_path,
            latents_bchw=final_latents_bchw,
            method=np.asarray(args.method),
            num_steps=np.asarray(args.num_steps, dtype=np.int32),
            omega=np.asarray(float(config.sampling.omega), dtype=np.float32),
            native_velocity_cfg_space=np.asarray(
                str(config.sampling.native_velocity_cfg_space)
            ),
            native_velocity_derivative_mode=np.asarray(
                str(config.sampling.native_velocity_derivative_mode)
            ),
            native_velocity_sigma_clamp=np.asarray(
                float(config.sampling.native_velocity_sigma_clamp),
                dtype=np.float32,
            ),
            transport_velocity_cfg_space=np.asarray(
                str(config.sampling.transport_velocity_cfg_space)
            ),
            transport_velocity_time_map=np.asarray(
                str(config.sampling.transport_velocity_time_map)
            ),
            transport_velocity_eps=np.asarray(
                float(config.sampling.transport_velocity_eps),
                dtype=np.float32,
            ),
            transport_velocity_scale_input=np.asarray(
                bool(config.sampling.transport_velocity_scale_input)
            ),
        )
        log_for_0("Saved grid to %s", grid_path)
        log_for_0("Saved latents to %s", latents_path)


if __name__ == "__main__":
    main()
