"""Sampling utilities for the dedicated plain SiT path."""

import jax
from jax import random
import jax.numpy as jnp

from utils.sit_transport_jax import create_transport


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
    raise ValueError(
        "training.half_precision_dtype must be one of: bfloat16, bf16, float16, fp16. "
        f"Got {dtype_name!r}."
    )


def _make_sample_labels(num_samples, num_classes, sample_idx=None):
    if sample_idx is not None:
        labels = jnp.arange(num_samples, dtype=jnp.int32)
        labels = labels + jnp.asarray(sample_idx, dtype=jnp.int32) * num_samples
        return labels % num_classes

    return random.randint(
        random.PRNGKey(0), (num_samples,), 0, num_classes, dtype=jnp.int32
    )


def _guided_velocity(model, variable, x, t_model, labels, cfg_scale):
    num_samples = x.shape[0]

    def conditional_velocity(_):
        return model.apply(variable, x, t_model, labels)

    def classifier_free_guided_velocity(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)

        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t_model, t_model], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)

        out = model.apply(variable, x_cat, t_cat, y_cat)
        cond, uncond = jnp.split(out, 2, axis=0)

        guided_first_three = uncond[..., :3] + cfg_scale * (
            cond[..., :3] - uncond[..., :3]
        )
        return jnp.concatenate([guided_first_three, cond[..., 3:]], axis=-1)

    return jax.lax.cond(
        jnp.equal(cfg_scale, 1.0),
        conditional_velocity,
        classifier_free_guided_velocity,
        operand=None,
    )


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    sample_idx=None,
):
    """Generate latent samples with plain SiT and fixed-step ODE integration."""
    del t_min, t_max
    if num_steps < 1:
        raise ValueError(
            f"Plain SiT sampling requires num_steps >= 1, got {num_steps}."
        )

    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels
    sample_dtype = _get_sampling_dtype(config)
    x = jax.random.normal(
        rng,
        (n_sample, img_size, img_size, img_channels),
        dtype=sample_dtype,
    )

    labels = _make_sample_labels(
        n_sample,
        config.dataset.num_classes,
        sample_idx=sample_idx,
    )

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
    )
    tau0, tau1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    times = jnp.linspace(float(tau0), float(tau1), num_steps + 1, dtype=sample_dtype)

    method = str(config.sampling.get("method", "euler")).lower()
    flip_time = bool(config.sampling.get("flip_time", False))
    cfg_scale = jnp.asarray(omega, dtype=sample_dtype)

    def euler_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        velocity = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        return (x_cur + dt * velocity).astype(sample_dtype)

    def heun_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        velocity = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        x_pred = x_cur + dt * velocity
        t_next = jnp.full(
            (n_sample,),
            1.0 - tau_next if flip_time else tau_next,
            dtype=sample_dtype,
        )
        velocity_next = _guided_velocity(model, variable, x_pred, t_next, labels, cfg_scale)
        return (x_cur + 0.5 * dt * (velocity + velocity_next)).astype(sample_dtype)

    if method == "euler":
        return jax.lax.fori_loop(0, num_steps, euler_step, x)
    if method == "heun":
        return jax.lax.fori_loop(0, num_steps, heun_step, x)
    raise ValueError(f"Unsupported plain SiT sampling method: {method}")


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
    """PMapped sampling step returning latents in BCHW format."""
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
