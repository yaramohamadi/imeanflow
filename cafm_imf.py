"""Original continuous adversarial flow matching losses for an iMF velocity head.

This module implements the infinitesimal/JVP objective from Continuous
Adversarial Flow Models.  It is intentionally separate from ``caimf.py``,
which implements this repository's finite-interval CA-iMF objective.
"""

import jax
import jax.numpy as jnp


def discriminator_jvp_logits(discriminator, discriminator_params, samples):
    """Evaluate D's directional derivative for real and generated velocities."""
    x_t = samples["x_t"]
    t = samples["t"]
    labels = samples["labels"]

    def potential(x_value, t_value):
        return discriminator.apply(
            {"params": discriminator_params}, x_value, t_value, labels
        )

    tangent_t = jnp.ones_like(t)
    potential_value, real_logit = jax.jvp(
        potential,
        (x_t, t),
        (samples["velocity_real"], tangent_t),
    )
    _, fake_logit = jax.jvp(
        potential,
        (x_t, t),
        (samples["velocity_fake"], tangent_t),
    )
    return real_logit, fake_logit, potential_value


def generator_jvp_logit(discriminator, discriminator_params, samples):
    """Evaluate the generated-velocity JVP while retaining gradients to G."""
    x_t = samples["x_t"]
    t = samples["t"]
    labels = samples["labels"]

    def potential(x_value, t_value):
        return discriminator.apply(
            {"params": discriminator_params}, x_value, t_value, labels
        )

    _, fake_logit = jax.jvp(
        potential,
        (x_t, t),
        (samples["velocity_fake"], jnp.ones_like(t)),
    )
    return fake_logit


def discriminator_loss(real_logit, fake_logit, potential, lambda_cp=1e-3):
    """Official CAFM least-squares discriminator objective plus centering."""
    classification = jnp.mean(
        (real_logit - 1.0) ** 2 + (fake_logit + 1.0) ** 2
    )
    centering = jnp.mean(potential**2)
    total = classification + lambda_cp * centering
    return total, {
        "loss": total,
        "loss_d": total,
        "loss_d_classification": classification,
        "loss_d_centering": centering,
        "real_logit": jnp.mean(real_logit),
        "fake_logit": jnp.mean(fake_logit),
        "potential": jnp.mean(potential),
    }


def generator_loss(
    terms,
    fake_logit,
    lambda_imf=0.0,
    lambda_adv=1.0,
    lambda_ot=0.0,
):
    """Official CAFM generator objective.

    ``lambda_imf`` is accepted only for compatibility with the shared training
    harness.  Original CAFM does not contain an iMF/FM regression term.
    """
    if lambda_imf != 0.0:
        raise ValueError(
            "Original CAFM requires lambda_imf=0. Use CA-iMF for a mixed "
            "iMF + adversarial objective."
        )
    loss_adv = jnp.mean((fake_logit - 1.0) ** 2)
    loss_ot = jnp.mean(terms["velocity_fake"] ** 2)
    total = lambda_adv * loss_adv + lambda_ot * loss_ot
    return total, {
        "loss": total,
        "loss_g": total,
        "loss_imf": jnp.asarray(0.0, dtype=total.dtype),
        "loss_u": jnp.asarray(0.0, dtype=total.dtype),
        "loss_v": jnp.asarray(0.0, dtype=total.dtype),
        "loss_adv": loss_adv,
        "loss_ot": loss_ot,
        "fake_logit": jnp.mean(fake_logit),
    }
