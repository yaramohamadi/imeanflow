"""Fixed-NFE AFM endpoint utilities for a plain (single-time) SiT generator.

Official recipe: ByteDance-Seed/Adversarial-Flow-Models, configs/train/afm/train_afm_multi_nfe.yaml
(use_t_src=True, use_t_tgt=False, pred=v, steps=K). Their path has noise at t=1 and the step goes
t -> t - 1/K. Our SiT path has noise at s=0 and data at s=1 (afm_sit_meft.sit_linear_path), so the
same step is s -> s_next = s + 1/K on the sampler's own Euler grid, and the v-prediction endpoint
x_t + (r - t) v becomes x_s + (s_next - s) v. The grid is the one the plain-SiT Euler sampler uses
(utils.sit_sample_util.generate), so every trained jump is exactly a sampling jump.
"""

import jax
import jax.numpy as jnp

from afm_sit_meft import sit_linear_path


def fixed_grid_times(tau0, tau1, num_steps, dtype=jnp.float32):
    """The K+1 Euler times of the plain-SiT sampler (same jnp.linspace call)."""
    if int(num_steps) < 1:
        raise ValueError(f"AFM needs num_steps >= 1, got {num_steps}.")
    return jnp.linspace(float(tau0), float(tau1), int(num_steps) + 1, dtype=dtype)


def sample_step_indices(rng, batch_size, num_steps):
    """Uniform jump index k in {0, ..., K-1} (official: randint(0, steps))."""
    return jax.random.randint(rng, (batch_size,), 0, int(num_steps), dtype=jnp.int32)


def drop_labels(rng, labels, num_classes, drop_prob):
    """Class dropout to the null class (index num_classes) so CFG stays usable at sampling."""
    labels = jnp.asarray(labels, jnp.int32)
    if float(drop_prob) <= 0.0:
        return labels
    drop = jax.random.uniform(rng, labels.shape) < float(drop_prob)
    return jnp.where(drop, jnp.full_like(labels, int(num_classes)), labels)


def fixed_grid_endpoint_terms(velocity_fn, images, labels, noise, step_index, times):
    """Real and generated endpoints of one fixed-grid jump s_k -> s_{k+1}.

    velocity_fn(x, s, y) returns the SiT transport velocity (data - noise direction).
    Keys follow train_afm: "t" = source time, "r" = target time (D is scored at "r").
    """
    num_steps = times.shape[0] - 1
    s = times[step_index].astype(images.dtype)
    s_next = times[step_index + 1].astype(images.dtype)
    x_s = sit_linear_path(images, noise, s)
    x_next = sit_linear_path(images, noise, s_next)
    v = velocity_fn(x_s, s, labels)
    interval = s_next - s
    x_next_fake = x_s + interval.reshape((-1,) + (1,) * (images.ndim - 1)) * v
    if x_next_fake.shape != x_s.shape:
        raise ValueError(f"Endpoint shape {x_next_fake.shape} != input {x_s.shape}.")
    return {
        "x_t": x_s,
        "x_r": x_next,
        "x_r_fake": x_next_fake,
        "u": v,
        "t": s,
        "r": s_next,
        "interval": interval,
        "zero_mask": step_index == num_steps - 1,   # last jump (ends on data)
        "labels": labels,
    }
