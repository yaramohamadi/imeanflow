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
import torch
from diffusers.models import AutoencoderKL

from models.torch_SiT import SiT_XL_2
from utils.sit_official_transport import Sampler, create_transport


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Debug-sample the raw original SiT checkpoint using ported official SiT sampling."
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
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the latent initialization.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit ImageNet labels to use. If omitted, sequential labels are used.",
    )
    parser.add_argument("--mode", type=str, default="ODE", choices=["ODE", "SDE"])
    parser.add_argument("--vae-type", type=str, default="mse", choices=["mse", "ema"])
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
    )
    parser.add_argument(
        "--loss-weight",
        type=str,
        default=None,
        choices=[None, "velocity", "likelihood"],
    )
    parser.add_argument("--train-eps", type=float, default=None)
    parser.add_argument("--sample-eps", type=float, default=None)
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="dopri5",
        help="Official SiT sampler name. Use euler/heun if torchdiffeq is unavailable.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--diffusion-form",
        type=str,
        default="sigma",
        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
    )
    parser.add_argument("--diffusion-norm", type=float, default=1.0)
    parser.add_argument(
        "--last-step",
        type=str,
        default="Mean",
        choices=["Mean", "Tweedie", "Euler", "None"],
    )
    parser.add_argument("--last-step-size", type=float, default=0.04)
    return parser.parse_args()


def _load_state_dict(checkpoint_path):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return raw


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


def _decode_latents(latents_bchw, vae):
    with torch.no_grad():
        samples = vae.decode(latents_bchw / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
    samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return samples


def _sample_native_sit(model, labels, num_steps, cfg_scale, seed, device, args):
    if num_steps < 2:
        raise ValueError(
            "Official SiT transport sampling requires num_steps >= 2. "
            f"Got {num_steps}."
        )
    batch_size = labels.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    latents = torch.randn((batch_size, 4, 32, 32), generator=generator, device=device)
    null_labels = torch.full((batch_size,), 1000, device=device, dtype=torch.long)
    model_input = torch.cat([latents, latents], dim=0)
    y_input = torch.cat([labels, null_labels], dim=0)
    model_kwargs = dict(y=y_input, cfg_scale=cfg_scale)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)
    if args.mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=num_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )
    else:
        last_step = None if args.last_step == "None" else args.last_step
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=last_step,
            last_step_size=args.last_step_size,
            num_steps=num_steps,
        )

    samples = sample_fn(model_input, model.forward_with_cfg, **model_kwargs)
    samples = samples[-1]
    samples, _ = samples.chunk(2, dim=0)
    return samples


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.abspath(args.checkpoint)

    model = SiT_XL_2(
        input_size=32,
        in_channels=4,
        num_classes=1000,
        class_dropout_prob=0.1,
        learn_sigma=True,
    )
    state_dict = _load_state_dict(checkpoint_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    if args.labels:
        labels = torch.tensor(args.labels, dtype=torch.long, device=device)
        if labels.numel() != args.num_samples:
            raise ValueError(
                f"--labels provided {labels.numel()} labels, expected {args.num_samples}"
            )
    else:
        labels = torch.arange(args.num_samples, device=device, dtype=torch.long) % 1000

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)

    metadata_lines = [
        f"checkpoint={checkpoint_path}",
        f"device={device}",
        f"num_samples={args.num_samples}",
        f"cfg_scale={args.cfg_scale}",
        f"seed={args.seed}",
        f"labels={labels.detach().cpu().tolist()}",
        f"mode={args.mode}",
        f"path_type={args.path_type}",
        f"prediction={args.prediction}",
        f"sampling_method={args.sampling_method}",
        f"missing_keys={missing_keys}",
        f"unexpected_keys={unexpected_keys}",
    ]

    for num_steps in args.num_steps:
        latents = _sample_native_sit(
            model=model,
            labels=labels,
            num_steps=num_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=device,
            args=args,
        )
        images = _decode_latents(latents, vae)
        output_path = output_dir / f"native_sit_steps_{num_steps:03d}.png"
        _save_grid(images, output_path)
        metadata_lines.append(str(output_path))

    (output_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
