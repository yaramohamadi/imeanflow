import argparse
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from flax import serialization
import torch
from diffusers.models import AutoencoderKL

from configs.load_config import get_config
from imf import generate, iMeanFlow
from train import infer_num_classes_from_latents
from utils.ckpt_util import load_checkpoint_params, restore_eval_checkpoint
from utils.vae_util import LatentManager


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Debug-sample the converted Caltech SiT/iMF/DMF path without FID or DINO."
    )
    parser.add_argument(
        "--config-mode",
        type=str,
        default="caltech_sit_dmf_finetune",
        help="Config mode loaded through configs/load_config.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory or raw .pt checkpoint to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where preview grids and metadata will be written.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[250, 100, 50, 25, 10, 4, 1],
        help="Sampling step counts to preview.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of images to generate per preview grid.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used for latent initialization.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Optional override for CFG scale.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Optional override for CFG interval minimum.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional override for CFG interval maximum.",
    )
    parser.add_argument(
        "--vae-type",
        type=str,
        default=None,
        help="Optional override for config.dataset.vae.",
    )
    parser.add_argument(
        "--sampling-family",
        type=str,
        default="converted_flow_map",
        choices=[
            "converted_flow_map",
            "sit_boundary_velocity",
            "meanflow_average_velocity",
        ],
        help="Use the converted flow-map sampler, a SiT-style boundary-velocity sampler, or an interval-conditioned MeanFlow average-velocity sampler.",
    )
    parser.add_argument(
        "--sit-sampling-method",
        type=str,
        default="euler",
        choices=["euler", "heun"],
        help="Fixed-step SiT-style ODE method for boundary-velocity previews.",
    )
    parser.add_argument(
        "--sit-flip-time",
        action="store_true",
        help="Map SiT transport time tau to model time t via t = 1 - tau.",
    )
    parser.add_argument(
        "--decode-style",
        type=str,
        default="auto",
        choices=["auto", "latent_manager", "sit_official"],
        help="Decode sampled latents with repo latent-manager stats or official SiT VAE scaling.",
    )
    parser.add_argument(
        "--label-space",
        type=str,
        default="config",
        choices=["config", "imagenet1000"],
        help="Use the config label space or force the original ImageNet-1000 label space.",
    )
    parser.add_argument(
        "--meanflow-reverse-time",
        action="store_true",
        help="Use the repo-native reverse-time grid (1 -> 0) for the average-velocity sampler instead of the official forward-time grid (0 -> 1).",
    )
    return parser.parse_args()


def _make_grid(images_uint8):
    num_images, height, width, channels = images_uint8.shape
    grid_cols = int(math.ceil(num_images ** 0.5))
    grid_rows = int(math.ceil(num_images / grid_cols))
    grid = np.zeros((grid_rows * height, grid_cols * width, channels), dtype=np.uint8)
    for idx, image in enumerate(images_uint8):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = image
    return grid


def _save_grid(images_uint8, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_make_grid(images_uint8)).save(output_path)


def _decode_latents(latents_bhwc, latent_manager):
    num_images = latents_bhwc.shape[0]
    decode_total = latent_manager.batch_size * jax.local_device_count()
    if num_images < decode_total:
        pad_shape = (decode_total - num_images,) + latents_bhwc.shape[1:]
        latents_bhwc = jnp.concatenate(
            [latents_bhwc, jnp.zeros(pad_shape, dtype=latents_bhwc.dtype)],
            axis=0,
        )
    latents_bchw = latents_bhwc.transpose(0, 3, 1, 2)
    images = latent_manager.decode(latents_bchw)[:num_images]
    images = images.transpose(0, 2, 3, 1)
    images = 127.5 * images + 128.0
    return np.asarray(jnp.clip(images, 0, 255).astype(jnp.uint8))


def _decode_latents_sit_official(latents_bhwc, vae, device):
    latents_bchw = np.asarray(latents_bhwc.transpose(0, 3, 1, 2))
    latents = torch.from_numpy(latents_bchw).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        samples = vae.decode(latents / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
    return samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def _merge_shape_compatible_params(target_params, source_tree):
    target_state = serialization.to_state_dict(target_params)
    source_state = serialization.to_state_dict(source_tree)

    def merge_state(target_subtree, source_subtree):
        if isinstance(target_subtree, dict):
            merged = {}
            source_subtree = source_subtree if isinstance(source_subtree, dict) else {}
            for key, target_value in target_subtree.items():
                merged[key] = merge_state(target_value, source_subtree.get(key))
            return merged
        if (
            source_subtree is not None
            and hasattr(target_subtree, "shape")
            and hasattr(source_subtree, "shape")
            and target_subtree.shape == source_subtree.shape
        ):
            return source_subtree
        return target_subtree

    merged_state = merge_state(target_state, source_state)
    return serialization.from_state_dict(target_params, merged_state)


def _restore_preview_params(model, config, checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith((".pt", ".pth", ".pth.tar")):
        x = jnp.ones(
            (1, config.dataset.image_size, config.dataset.image_size, config.dataset.image_channels),
            dtype=jnp.float32,
        )
        t = jnp.ones((1,), dtype=jnp.int32)
        y = jnp.ones((1,), dtype=jnp.int32)
        variables = model.init({"params": jax.random.PRNGKey(0)}, x, t, y)
        initial_params = variables["params"]
        target_state = serialization.to_state_dict(initial_params)
        source_tree = load_checkpoint_params(
            checkpoint_path,
            prefer_ema=False,
            target_state=target_state,
        )
        params = _merge_shape_compatible_params(initial_params, source_tree)
        return {"params": params}, 0

    state = restore_eval_checkpoint(checkpoint_path, use_ema=config.training.get("use_ema", False))
    return {"params": state.params}, int(jnp.asarray(state.step))


def _make_preview_labels(num_samples, num_classes):
    return jnp.arange(num_samples, dtype=jnp.int32) % num_classes


def _guided_boundary_velocity(model, variable, x, t_model, labels, omega):
    batch_size = x.shape[0]
    omega_arr = jnp.full((batch_size,), omega, dtype=jnp.float32)
    v_c, v_u = model.apply(
        variable,
        x,
        t_model,
        omega_arr,
        labels,
        method=model.v_fn,
    )
    guided_first_three = v_u[..., :3] + omega * (v_c[..., :3] - v_u[..., :3])
    return jnp.concatenate([guided_first_three, v_c[..., 3:]], axis=-1)


def _guided_average_velocity(model, variable, x, t_model, r_model, labels, omega, t_min, t_max):
    batch_size = x.shape[0]
    y_null = jnp.full((batch_size,), model.num_classes, dtype=jnp.int32)
    x_cat = jnp.concatenate([x, x], axis=0)
    y_cat = jnp.concatenate([labels, y_null], axis=0)
    t_cat = jnp.concatenate([t_model, t_model], axis=0)
    h = t_model - r_model
    h_cat = jnp.concatenate([h, h], axis=0)
    omega_cat = jnp.full((2 * batch_size,), omega, dtype=jnp.float32)
    t_min_cat = jnp.full((2 * batch_size,), t_min, dtype=jnp.float32)
    t_max_cat = jnp.full((2 * batch_size,), t_max, dtype=jnp.float32)
    u_cat, _ = model.apply(
        variable,
        x_cat,
        t_cat,
        h_cat,
        omega_cat,
        t_min_cat,
        t_max_cat,
        y_cat,
        method=model.u_fn,
    )
    u_c, u_u = jnp.split(u_cat, 2, axis=0)
    guided_first_three = u_u[..., :3] + omega * (u_c[..., :3] - u_u[..., :3])
    return jnp.concatenate([guided_first_three, u_c[..., 3:]], axis=-1)


def _sample_converted_boundary_velocity(
    *,
    variable,
    model,
    config,
    num_steps,
    omega,
    sample_seed,
    num_samples,
    flip_time,
    method,
):
    if num_steps < 2:
        raise ValueError(
            "SiT-style boundary velocity sampling requires num_steps >= 2. "
            f"Got {num_steps}."
        )
    rng = jax.random.PRNGKey(sample_seed)
    z = jax.random.normal(
        rng,
        (
            num_samples,
            config.dataset.image_size,
            config.dataset.image_size,
            config.dataset.image_channels,
        ),
        dtype=jnp.float32,
    )
    labels = _make_preview_labels(num_samples, config.dataset.num_classes)
    tau_steps = jnp.linspace(0.0, 1.0, num_steps, dtype=jnp.float32)

    def boundary_velocity(x, tau):
        t_scalar = 1.0 - tau if flip_time else tau
        t_batch = jnp.full((num_samples,), t_scalar, dtype=jnp.float32)
        return _guided_boundary_velocity(model, variable, x, t_batch, labels, omega)

    if method == "euler":
        x = z
        for idx in range(num_steps - 1):
            tau_cur = tau_steps[idx]
            tau_next = tau_steps[idx + 1]
            dt = tau_next - tau_cur
            x = x + dt * boundary_velocity(x, tau_cur)
        return x

    if method == "heun":
        x = z
        for idx in range(num_steps - 1):
            tau_cur = tau_steps[idx]
            tau_next = tau_steps[idx + 1]
            dt = tau_next - tau_cur
            k1 = boundary_velocity(x, tau_cur)
            pred = x + dt * k1
            k2 = boundary_velocity(pred, tau_next)
            x = x + 0.5 * dt * (k1 + k2)
        return x

    raise NotImplementedError(f"Unsupported sit boundary sampling method: {method}")


def _sample_meanflow_average_velocity(
    *,
    variable,
    model,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    sample_seed,
    num_samples,
    reverse_time,
):
    if num_steps < 1:
        raise ValueError(
            "MeanFlow average-velocity sampling requires num_steps >= 1. "
            f"Got {num_steps}."
        )
    rng = jax.random.PRNGKey(sample_seed)
    x = jax.random.normal(
        rng,
        (
            num_samples,
            config.dataset.image_size,
            config.dataset.image_size,
            config.dataset.image_channels,
        ),
        dtype=jnp.float32,
    )
    labels = _make_preview_labels(num_samples, config.dataset.num_classes)
    if reverse_time:
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1, dtype=jnp.float32)
    else:
        t_steps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)

    for idx in range(num_steps):
        t_scalar = t_steps[idx]
        r_scalar = t_steps[idx + 1]
        t_batch = jnp.full((num_samples,), t_scalar, dtype=jnp.float32)
        r_batch = jnp.full((num_samples,), r_scalar, dtype=jnp.float32)
        u = _guided_average_velocity(
            model,
            variable,
            x,
            t_batch,
            r_batch,
            labels,
            omega,
            t_min,
            t_max,
        )
        x = x + (r_scalar - t_scalar) * u
    return x


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config_mode)
    if args.label_space == "imagenet1000":
        config.dataset.num_classes_from_data = False
        config.dataset.num_classes = 1000
        config.model.num_classes = 1000
        config.sampling.num_classes = 1000
    elif config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
    if args.vae_type is not None:
        config.dataset.vae = args.vae_type

    omega = config.sampling.omega if args.omega is None else args.omega
    t_min = config.sampling.t_min if args.t_min is None else args.t_min
    t_max = config.sampling.t_max if args.t_max is None else args.t_max

    model = iMeanFlow(**config.model.to_dict(), eval=True)
    variable, restored_step = _restore_preview_params(model, config, args.checkpoint)
    checkpoint_is_raw_pt = os.path.isfile(args.checkpoint) and args.checkpoint.endswith(
        (".pt", ".pth", ".pth.tar")
    )
    decode_style = args.decode_style
    if decode_style == "auto":
        if args.sampling_family == "sit_boundary_velocity" and checkpoint_is_raw_pt:
            decode_style = "sit_official"
        else:
            decode_style = "latent_manager"

    latent_manager = None
    sit_vae = None
    sit_vae_device = None
    if decode_style == "latent_manager":
        latent_manager = LatentManager(
            config.dataset.vae,
            decode_batch_size=args.num_samples,
            input_size=config.dataset.image_size,
        )
    else:
        sit_vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sit_vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{config.dataset.vae}"
        ).to(sit_vae_device)

    metadata_lines = [
        f"config_mode={args.config_mode}",
        f"checkpoint={os.path.abspath(args.checkpoint)}",
        f"restored_step={restored_step}",
        f"num_samples={args.num_samples}",
        f"omega={omega}",
        f"t_min={t_min}",
        f"t_max={t_max}",
        f"sample_seed={args.sample_seed}",
        f"sampling_family={args.sampling_family}",
        f"decode_style={decode_style}",
        f"label_space={args.label_space}",
        f"effective_num_classes={config.dataset.num_classes}",
        f"meanflow_reverse_time={args.meanflow_reverse_time}",
    ]

    rng = jax.random.PRNGKey(args.sample_seed)
    for num_steps in args.num_steps:
        if args.sampling_family == "converted_flow_map":
            latents_bhwc = generate(
                variable=variable,
                model=model,
                rng=rng,
                n_sample=args.num_samples,
                config=config,
                num_steps=num_steps,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
                sample_idx=0,
            )
        elif args.sampling_family == "sit_boundary_velocity":
            latents_bhwc = _sample_converted_boundary_velocity(
                variable=variable,
                model=model,
                config=config,
                num_steps=num_steps,
                omega=omega,
                sample_seed=args.sample_seed,
                num_samples=args.num_samples,
                flip_time=args.sit_flip_time,
                method=args.sit_sampling_method,
            )
        else:
            latents_bhwc = _sample_meanflow_average_velocity(
                variable=variable,
                model=model,
                config=config,
                num_steps=num_steps,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
                sample_seed=args.sample_seed,
                num_samples=args.num_samples,
                reverse_time=args.meanflow_reverse_time,
            )

        if decode_style == "latent_manager":
            images = _decode_latents(latents_bhwc, latent_manager)
        else:
            images = _decode_latents_sit_official(latents_bhwc, sit_vae, sit_vae_device)
        output_path = output_dir / f"converted_path_steps_{num_steps:03d}.png"
        _save_grid(images, output_path)
        metadata_lines.append(str(output_path))

    (output_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
