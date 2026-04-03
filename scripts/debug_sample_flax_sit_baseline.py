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

from models.imfDiT import flaxSiT_XL_2
from utils.ckpt_util import load_checkpoint_params
from utils.sit_official_transport import create_transport


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Debug-sample an exact Flax SiT baseline loaded from the raw SiT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ens/AT74470/imeanflow/files/weights/SiT-XL-2-256.pt",
        help="Path to the raw SiT PyTorch checkpoint.",
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
        default=[250, 100, 50, 25, 10, 4, 2],
        help="Sampling step counts to preview.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of images to generate per preview grid.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale applied to the first three channels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for latent initialization.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit ImageNet labels to use. If omitted, sequential labels are used.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="euler",
        choices=["euler", "heun"],
        help="Fixed-step ODE sampler.",
    )
    parser.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="Official SiT path type used to derive the integration interval.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="Prediction type used to derive the integration interval.",
    )
    parser.add_argument(
        "--loss-weight",
        type=str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="Official SiT transport loss weighting.",
    )
    parser.add_argument("--train-eps", type=float, default=None)
    parser.add_argument("--sample-eps", type=float, default=None)
    parser.add_argument(
        "--flip-time",
        action="store_true",
        help="Map transport time tau to model time t with t = 1 - tau.",
    )
    parser.add_argument(
        "--vae-type",
        type=str,
        default="mse",
        choices=["mse", "ema"],
        help="Official SiT VAE decode variant.",
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


def _decode_latents(latents_bhwc, vae, device):
    latents_bchw = np.asarray(latents_bhwc.transpose(0, 3, 1, 2))
    latents = torch.from_numpy(latents_bchw).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        samples = vae.decode(latents / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
    return samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def _restore_flax_sit_params(model, checkpoint_path):
    x = jnp.ones((1, 32, 32, 4), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    y = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init({"params": jax.random.PRNGKey(0)}, x, t, y)
    initial_params = variables["params"]
    source_tree = load_checkpoint_params(
        checkpoint_path,
        prefer_ema=False,
        target_state={"net": serialization.to_state_dict(initial_params)},
    )
    params = serialization.from_state_dict(initial_params, source_tree["net"])
    return params


def _make_preview_labels(num_samples, labels):
    if labels is None:
        return jnp.arange(num_samples, dtype=jnp.int32) % 1000
    label_array = jnp.asarray(labels, dtype=jnp.int32)
    if label_array.shape[0] != num_samples:
        raise ValueError(
            f"--labels provided {label_array.shape[0]} labels, expected {num_samples}"
        )
    return label_array


def _guided_velocity(model, variables, x, t_model, labels, cfg_scale):
    null_labels = jnp.full((x.shape[0],), 1000, dtype=jnp.int32)
    cond = model.apply(variables, x, t_model, labels)
    uncond = model.apply(variables, x, t_model, null_labels)
    guided_first_three = uncond[..., :3] + cfg_scale * (cond[..., :3] - uncond[..., :3])
    return jnp.concatenate([guided_first_three, cond[..., 3:]], axis=-1)


def _sample_flax_sit(
    *,
    model,
    variables,
    labels,
    num_steps,
    cfg_scale,
    seed,
    sampling_method,
    transport,
    flip_time,
):
    if num_steps < 2:
        raise ValueError(
            "Exact Flax SiT preview requires num_steps >= 2. "
            f"Got {num_steps}."
        )

    rng = jax.random.PRNGKey(seed)
    x = jax.random.normal(rng, (labels.shape[0], 32, 32, 4), dtype=jnp.float32)
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    times = jnp.linspace(float(t0), float(t1), num_steps, dtype=jnp.float32)

    for idx in range(num_steps - 1):
        tau_cur = times[idx]
        tau_next = times[idx + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full((labels.shape[0],), 1.0 - tau_cur if flip_time else tau_cur, dtype=jnp.float32)
        velocity = _guided_velocity(model, variables, x, t_cur, labels, cfg_scale)
        if sampling_method == "euler":
            x = x + dt * velocity
        else:
            x_pred = x + dt * velocity
            t_next_model = jnp.full(
                (labels.shape[0],),
                1.0 - tau_next if flip_time else tau_next,
                dtype=jnp.float32,
            )
            velocity_next = _guided_velocity(
                model,
                variables,
                x_pred,
                t_next_model,
                labels,
                cfg_scale,
            )
            x = x + 0.5 * dt * (velocity + velocity_next)
    return x


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = os.path.abspath(args.checkpoint)
    model = flaxSiT_XL_2(
        input_size=32,
        in_channels=4,
        num_classes=1000,
        use_null_class=True,
        eval=True,
    )
    params = _restore_flax_sit_params(model, checkpoint_path)
    variables = {"params": params}
    labels = _make_preview_labels(args.num_samples, args.labels)

    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(torch_device)

    metadata_lines = [
        f"checkpoint={checkpoint_path}",
        f"num_samples={args.num_samples}",
        f"cfg_scale={args.cfg_scale}",
        f"seed={args.seed}",
        f"labels={np.asarray(labels).tolist()}",
        f"sampling_method={args.sampling_method}",
        f"path_type={args.path_type}",
        f"prediction={args.prediction}",
        f"flip_time={args.flip_time}",
        f"torch_decode_device={torch_device}",
    ]

    for num_steps in args.num_steps:
        latents = _sample_flax_sit(
            model=model,
            variables=variables,
            labels=labels,
            num_steps=num_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            sampling_method=args.sampling_method,
            transport=transport,
            flip_time=args.flip_time,
        )
        images = _decode_latents(latents, vae, torch_device)
        output_path = output_dir / f"flax_sit_steps_{num_steps:03d}.png"
        _save_grid(images, output_path)
        metadata_lines.append(str(output_path))

    (output_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n")


if __name__ == "__main__":
    main()
