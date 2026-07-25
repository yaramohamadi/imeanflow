"""Focused tests for original CAFM-on-iMF JVP losses."""

import jax
import jax.numpy as jnp

from cafm_imf import (
    discriminator_jvp_logits,
    discriminator_loss,
    generator_loss,
)


class ToyPotential:
    @staticmethod
    def apply(variables, x, t, labels):
        del labels
        scale = variables["params"]["scale"]
        return scale * jnp.sum(x, axis=(1, 2, 3)) + t


def test_jvp_uses_velocity_and_unit_time_tangent():
    samples = {
        "x_t": jnp.ones((2, 1, 1, 1)),
        "t": jnp.asarray([0.2, 0.7]),
        "labels": jnp.asarray([0, 1]),
        "velocity_real": jnp.full((2, 1, 1, 1), 2.0),
        "velocity_fake": jnp.full((2, 1, 1, 1), 3.0),
    }
    real, fake, potential = discriminator_jvp_logits(
        ToyPotential(), {"scale": jnp.asarray(2.0)}, samples
    )
    assert jnp.allclose(real, 5.0)
    assert jnp.allclose(fake, 7.0)
    assert jnp.allclose(potential, jnp.asarray([2.2, 2.7]))


def test_generator_gradient_flows_through_discriminator_jvp():
    def loss(velocity):
        samples = {
            "x_t": jnp.ones((1, 1, 1, 1)),
            "t": jnp.asarray([0.5]),
            "labels": jnp.asarray([0]),
            "velocity_real": jnp.ones((1, 1, 1, 1)),
            "velocity_fake": velocity,
        }
        _, fake, _ = discriminator_jvp_logits(
            ToyPotential(), {"scale": jnp.asarray(2.0)}, samples
        )
        return jnp.mean((fake - 1.0) ** 2)

    grad = jax.grad(loss)(jnp.full((1, 1, 1, 1), 3.0))
    assert jnp.any(grad != 0)


def test_official_losses_are_finite():
    d_loss, _ = discriminator_loss(
        jnp.asarray([0.5]), jnp.asarray([-0.5]), jnp.asarray([0.1])
    )
    g_loss, metrics = generator_loss(
        {"velocity_fake": jnp.ones((1, 2, 2, 1))},
        jnp.asarray([0.25]),
        lambda_imf=0.0,
        lambda_adv=1.0,
        lambda_ot=0.0,
    )
    assert jnp.isfinite(d_loss)
    assert jnp.isfinite(g_loss)
    assert metrics["loss_imf"] == 0
