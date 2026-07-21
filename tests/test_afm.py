import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from afm import (
    discriminator_adversarial_loss,
    generated_lower_endpoint,
    generator_adversarial_loss,
    linear_path,
    sample_time_pairs,
)
from models.afm_discriminator import AFMDiscriminator
from train_afm import AFMTrainState, _optional_imf_loss, validate_target_labels


def _toy_discriminator(params, x, time, labels):
    del labels
    return params * jnp.mean(x, axis=tuple(range(1, x.ndim))) + time


def test_time_sampling_respects_strict_interval_and_r_zero_mixture():
    r, t, interval, zero_mask = sample_time_pairs(
        jax.random.key(1), 4096, min_interval=0.07, p_r_zero=0.3
    )
    assert np.all(np.asarray(r) >= 0.0)
    assert np.all(np.asarray(r) < np.asarray(t))
    assert np.all(np.asarray(t) <= 1.0)
    assert np.min(np.asarray(interval)) >= 0.07 - 1e-6
    assert 0.25 < float(jnp.mean(zero_mask)) < 0.35


def test_generated_endpoint_shape_and_one_step_formula():
    x1 = jnp.ones((3, 4, 4, 2))
    u = 0.25 * jnp.ones_like(x1)
    r = jnp.zeros((3,))
    t = jnp.ones((3,))
    endpoint = generated_lower_endpoint(x1, u, r, t)
    assert endpoint.shape == x1.shape
    np.testing.assert_allclose(endpoint, x1 - u)


def test_discriminator_interface_is_endpoint_time_label_only():
    parameters = list(inspect.signature(AFMDiscriminator.__call__).parameters)
    assert parameters == ["self", "endpoint", "endpoint_time", "target_label"]


def test_d_update_detaches_generator_and_has_nonzero_d_gradient():
    x0 = jnp.zeros((2, 2, 2, 1))
    x1 = jnp.ones_like(x0)
    r = jnp.zeros((2,))
    t = jnp.ones((2,))
    labels = jnp.zeros((2,), jnp.int32)
    x_t = linear_path(x0, x1, t)
    x_r = linear_path(x0, x1, r)

    def loss(generator_parameter, discriminator_parameter):
        u = generator_parameter * jnp.ones_like(x_t)
        fake = jax.lax.stop_gradient(
            generated_lower_endpoint(x_t, u, r, t)
        )
        d_real = _toy_discriminator(discriminator_parameter, x_r, r, labels)
        d_fake = _toy_discriminator(discriminator_parameter, fake, r, labels)
        return discriminator_adversarial_loss(d_real, d_fake)

    grad_g, grad_d = jax.grad(loss, argnums=(0, 1))(jnp.asarray(0.2), jnp.asarray(1.0))
    assert float(grad_g) == 0.0
    assert abs(float(grad_d)) > 0.0


def test_g_update_has_nonzero_generator_gradient():
    x0 = jnp.zeros((2, 2, 2, 1))
    x1 = jnp.ones_like(x0)
    r = jnp.zeros((2,))
    t = jnp.ones((2,))
    labels = jnp.zeros((2,), jnp.int32)
    x_t = linear_path(x0, x1, t)
    x_r = linear_path(x0, x1, r)
    discriminator_parameter = jnp.asarray(1.0)

    def loss(generator_parameter):
        fake = generated_lower_endpoint(
            x_t, generator_parameter * jnp.ones_like(x_t), r, t
        )
        d_real = jax.lax.stop_gradient(
            _toy_discriminator(discriminator_parameter, x_r, r, labels)
        )
        d_fake = _toy_discriminator(discriminator_parameter, fake, r, labels)
        return generator_adversarial_loss(d_real, d_fake)

    assert abs(float(jax.grad(loss)(jnp.asarray(0.2)))) > 0.0


def test_zero_imf_weight_does_not_call_jvp_branch():
    def forbidden():
        raise AssertionError("JVP branch must not be called")

    assert float(_optional_imf_loss(0.0, forbidden)) == 0.0
    assert float(_optional_imf_loss(1.0, lambda: jnp.asarray(3.0))) == 3.0


def test_label_range_validation():
    validate_target_labels(np.asarray([0, 100]), 101)
    try:
        validate_target_labels(np.asarray([0, 101]), 101)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range target label was not rejected.")


def test_state_serialization_preserves_counters_rng_and_optimizer_state():
    params = {"w": jnp.asarray([1.0, 2.0])}
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)
    state = AFMTrainState(
        step=jnp.asarray(12),
        epoch=jnp.asarray(2),
        gen_step=jnp.asarray(3),
        dis_step=jnp.asarray(9),
        real_images_seen=jnp.asarray(48),
        params=params,
        ema_params=params,
        gen_opt_state=opt_state,
        dis_params=params,
        dis_opt_state=opt_state,
        source_params=None,
        rng=jax.random.PRNGKey(7),
    )
    restored = serialization.from_bytes(state, serialization.to_bytes(state))
    assert int(restored.step) == 12
    assert int(restored.gen_step) == 3
    assert int(restored.dis_step) == 9
    assert int(restored.real_images_seen) == 48
    np.testing.assert_array_equal(restored.rng, state.rng)
    assert jax.tree_util.tree_structure(restored.gen_opt_state) == jax.tree_util.tree_structure(opt_state)


if __name__ == "__main__":
    checks = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
