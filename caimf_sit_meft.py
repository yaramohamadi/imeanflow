"""Finite-interval CA-iMF logits for the forward-time SiT-MeFT convention."""

import jax.numpy as jnp


def _check_full_score(score_mode):
    # The score ablations (caimf.finite_interval_logits score_mode) are iMF-only; the forward-time
    # SiT/JiT convention supports the full CAMF score only.
    if score_mode != "full":
        raise NotImplementedError("score_mode=%r is not implemented for SiT/JiT-MeFT CAMF." % score_mode)


def finite_interval_logits(discriminator, discriminator_params, samples, score_mode="full"):
    """Compute real/fake potential quotients over 0 <= t < r <= 1."""
    _check_full_score(score_mode)
    t, r = samples["t"], samples["r"]
    labels = samples["labels"]
    interval = r - t
    d_t = discriminator.apply(
        {"params": discriminator_params},
        samples["x_t"],
        t,
        t,
        r,
        labels,
    )
    d_r = discriminator.apply(
        {"params": discriminator_params},
        samples["x_r"],
        r,
        t,
        r,
        labels,
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params},
        samples["xhat_r"],
        r,
        t,
        r,
        labels,
    )
    real_logit = (d_r - d_t) / interval
    fake_logit = (d_fake - d_t) / interval
    return real_logit, fake_logit, (d_t, d_r, d_fake)


def finite_fake_logit(discriminator, discriminator_params, samples, score_mode="full"):
    """Compute only the generator-facing fake quotient."""
    _check_full_score(score_mode)
    t, r = samples["t"], samples["r"]
    labels = samples["labels"]
    d_t = discriminator.apply(
        {"params": discriminator_params},
        samples["x_t"],
        t,
        t,
        r,
        labels,
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params},
        samples["xhat_r"],
        r,
        t,
        r,
        labels,
    )
    return (d_fake - d_t) / jnp.maximum(r - t, 1e-6)
