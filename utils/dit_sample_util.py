"""Sampling utilities for the JAX DiT diffusion path."""

import jax
import jax.numpy as jnp
from jax import random

from utils.dit_diffusion import create_diffusion


def _get_sampling_dtype(config):
    sampling = config.get("sampling", {})
    if not sampling.get("half_precision", False):
        return jnp.float32
    dtype_name = str(
        sampling.get(
            "half_precision_dtype",
            config.training.get("half_precision_dtype", "float16"),
        )
    ).lower()
    if dtype_name in ("fp16", "float16"):
        return jnp.float16
    if dtype_name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    raise ValueError(f"Unsupported half precision dtype: {dtype_name}")


def _make_sample_labels(num_samples, num_classes, sample_idx=None):
    if sample_idx is not None:
        labels = jnp.arange(num_samples, dtype=jnp.int32)
        labels = labels + jnp.asarray(sample_idx, dtype=jnp.int32) * num_samples
        return labels % num_classes
    return random.randint(
        random.PRNGKey(0), (num_samples,), 0, num_classes, dtype=jnp.int32
    )


def _guided_model_output(model, variable, x, t, labels, cfg_scale):
    num_samples = x.shape[0]

    def conditional(_):
        return model.apply(variable, x, t.astype(jnp.float32), labels)

    def guided(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)
        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t, t], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)
        out = model.apply(variable, x_cat, t_cat.astype(jnp.float32), y_cat)
        cond, uncond = jnp.split(out, 2, axis=0)
        guided_eps = uncond[..., :3] + cfg_scale * (cond[..., :3] - uncond[..., :3])
        return jnp.concatenate([guided_eps, cond[..., 3:]], axis=-1)

    return jax.lax.cond(jnp.equal(cfg_scale, 1.0), conditional, guided, operand=None)


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min=None,
    t_max=None,
    sample_idx=None,
):
    del t_min, t_max
    if num_steps < 1:
        raise ValueError(f"DiT sampling requires num_steps >= 1, got {num_steps}.")

    img_size = int(config.dataset.image_size)
    img_channels = int(config.dataset.image_channels)
    sample_dtype = _get_sampling_dtype(config)
    noise = random.normal(
        rng,
        (n_sample, img_size, img_size, img_channels),
        dtype=sample_dtype,
    )
    labels = _make_sample_labels(
        n_sample,
        int(config.dataset.num_classes),
        sample_idx=sample_idx,
    )
    cfg_scale = jnp.asarray(
        config.sampling.get("cfg_scale", omega),
        dtype=sample_dtype,
    )

    diffusion = create_diffusion(
        str(num_steps),
        noise_schedule=config.diffusion.noise_schedule,
        learn_sigma=config.diffusion.learn_sigma,
        predict_xstart=config.diffusion.predict_xstart,
        rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        diffusion_steps=config.diffusion.diffusion_steps,
    )

    def model_fn(x, t):
        return _guided_model_output(model, variable, x, t, labels, cfg_scale)

    return diffusion.p_sample_loop(
        model_fn,
        noise,
        rng=random.fold_in(rng, 1),
        clip_denoised=False,
        dtype=sample_dtype,
    )


def sample_step(
    variable,
    sample_idx,
    model,
    rng_init,
    device_batch_size,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
):
    rng_sample = random.fold_in(rng_init, sample_idx)
    images = generate(
        variable,
        model,
        rng_sample,
        device_batch_size,
        config,
        num_steps,
        omega,
        t_min,
        t_max,
        sample_idx=sample_idx,
    )
    return images.transpose(0, 3, 1, 2)
