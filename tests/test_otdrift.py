import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from otdrift import (
    barycentric_targets,
    energy_distance,
    entropic_ot_cost,
    frozen_plan_loss,
    interval_levels,
    relative_epsilon,
    sinkhorn_divergence,
    squared_cost_matrix,
    target_interpolant,
    transport_plan,
)


def test_cost_matrix_is_dimension_normalised():
    x = jnp.zeros((3, 2, 2, 4))
    y = jnp.ones((5, 2, 2, 4))
    cost = squared_cost_matrix(x, y)
    assert cost.shape == (3, 5)
    # every coordinate differs by 1, and the cost is divided by D
    np.testing.assert_allclose(np.asarray(cost), 1.0, rtol=1e-6)


def test_plan_marginals_are_uniform():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (16, 6))
    y = jax.random.normal(jax.random.PRNGKey(1), (16, 6)) + 0.7
    cost = squared_cost_matrix(x, y)
    plan = transport_plan(cost, relative_epsilon(cost, 0.05), 400)
    np.testing.assert_allclose(np.asarray(jnp.sum(plan, axis=1)), 1.0 / 16, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.sum(plan, axis=0)), 1.0 / 16, atol=1e-5)
    np.testing.assert_allclose(float(jnp.sum(plan)), 1.0, atol=1e-5)


def test_divergence_vanishes_on_identical_sets():
    x = jax.random.normal(jax.random.PRNGKey(2), (24, 8))
    value = sinkhorn_divergence(x, x, relative_epsilon(squared_cost_matrix(x, x), 0.05), 300)
    assert abs(float(value)) < 1e-5


def test_divergence_is_positive_and_grows_with_the_gap():
    rng = jax.random.PRNGKey(3)
    x = jax.random.normal(rng, (32, 8))
    values = []
    for shift in (0.25, 0.5, 1.0, 2.0):
        y = jax.random.normal(jax.random.PRNGKey(4), (32, 8)) + shift
        cost = squared_cost_matrix(x, y)
        values.append(float(sinkhorn_divergence(x, y, relative_epsilon(cost, 0.05), 300)))
    assert all(value > 0.0 for value in values), values
    assert values == sorted(values), values


def test_barycentric_identity_matches_the_double_sum():
    """sum_ij P_ij |x_i - y_j|^2 == sum_i w_i |x_i - ybar_i|^2 + const(x)."""
    rng = jax.random.PRNGKey(5)
    x = jax.random.normal(rng, (12, 5))
    y = jax.random.normal(jax.random.PRNGKey(6), (12, 5)) + 0.4
    cost = squared_cost_matrix(x, y)
    plan = transport_plan(cost, relative_epsilon(cost, 0.05), 300)
    weights, targets = barycentric_targets(plan, y)

    def double_sum(z):
        return jnp.sum(plan * squared_cost_matrix(z, y))

    def barycentric(z):
        return jnp.sum(weights * jnp.sum((z - targets) ** 2, axis=1) / z.shape[1])

    # the two differ by a constant, so their gradients must agree exactly
    np.testing.assert_allclose(
        np.asarray(jax.grad(double_sum)(x)),
        np.asarray(jax.grad(barycentric)(x)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_frozen_plan_gradient_matches_autodiff_through_sinkhorn():
    """The envelope-theorem gradient is the point of the whole construction."""
    rng = jax.random.PRNGKey(7)
    x = jax.random.normal(rng, (24, 6))
    y = jax.random.normal(jax.random.PRNGKey(8), (24, 6)) + 0.6
    iters = 600
    eps_rel = 0.1

    def through(z):
        cost = squared_cost_matrix(z, y)
        return sinkhorn_divergence(z, y, relative_epsilon(cost, eps_rel), iters)

    def frozen(z):
        loss, _ = frozen_plan_loss(
            z, y, epsilon_relative=eps_rel, num_iters=iters, repulsion_clip=1e9
        )
        return loss

    grad_through = np.asarray(jax.grad(through)(x))
    grad_frozen = np.asarray(jax.grad(frozen)(x))
    cosine = float(
        np.sum(grad_through * grad_frozen)
        / (np.linalg.norm(grad_through) * np.linalg.norm(grad_frozen))
    )
    assert cosine > 0.99, cosine
    scale = np.linalg.norm(grad_frozen) / np.linalg.norm(grad_through)
    assert 0.5 < scale < 2.0, scale


def test_frozen_plan_loss_is_microbatch_decomposable():
    """Accumulating over microbatches must give the same gradient as one pass."""
    rng = jax.random.PRNGKey(9)
    x = jax.random.normal(rng, (16, 5))
    y = jax.random.normal(jax.random.PRNGKey(10), (16, 5)) + 0.5

    cost = squared_cost_matrix(x, y)
    epsilon = relative_epsilon(cost, 0.05)
    plan = transport_plan(cost, epsilon, 400)
    weights, targets = barycentric_targets(plan, y)

    def full(z):
        return jnp.sum(weights * jnp.sum((z - targets) ** 2, axis=1) / z.shape[1])

    grad_full = np.asarray(jax.grad(full)(x))

    grad_accumulated = np.zeros_like(grad_full)
    for start in range(0, 16, 4):
        stop = start + 4
        slice_weights = weights[start:stop]
        slice_targets = targets[start:stop]

        def piece(z_slice):
            return jnp.sum(
                slice_weights * jnp.sum((z_slice - slice_targets) ** 2, axis=1) / z_slice.shape[1]
            )

        grad_accumulated[start:stop] = np.asarray(jax.grad(piece)(x[start:stop]))

    np.testing.assert_allclose(grad_accumulated, grad_full, rtol=1e-6, atol=1e-7)


def test_debiasing_keeps_the_particle_set_from_collapsing():
    """Without the self term the loss drags every particle toward the barycentre."""
    rng = jax.random.PRNGKey(11)
    y = jax.random.normal(jax.random.PRNGKey(12), (48, 4)) * 1.0
    x = jax.random.normal(rng, (48, 4)) * 1.0 + 1.5

    def run(debias):
        z = x
        for _ in range(60):
            grad = jax.grad(
                lambda w: frozen_plan_loss(
                    w, y, epsilon_relative=0.05, num_iters=200, debias=debias
                )[0]
            )(z)
            z = z - 2.0 * grad
        return float(jnp.mean(jnp.std(z, axis=0)))

    target_spread = float(jnp.mean(jnp.std(y, axis=0)))
    spread_debiased = run(True)
    spread_biased = run(False)
    assert spread_debiased > spread_biased, (spread_debiased, spread_biased)
    assert spread_debiased > 0.5 * target_spread, (spread_debiased, target_spread)


def test_frozen_plan_aux_reports_usable_diagnostics():
    x = jax.random.normal(jax.random.PRNGKey(13), (16, 4, 4, 2))
    y = jax.random.normal(jax.random.PRNGKey(14), (16, 4, 4, 2)) + 0.3
    _, aux = frozen_plan_loss(x, y, epsilon_relative=0.05, num_iters=200)
    for key in (
        "ot_attraction",
        "ot_repulsion",
        "ot_epsilon",
        "ot_matched_cost",
        "ot_plan_entropy_ratio",
        "ot_fake_spread",
        "ot_real_spread",
    ):
        assert key in aux, key
        assert np.isfinite(float(aux[key])), key
    assert 0.0 < float(aux["ot_plan_entropy_ratio"]) <= 1.0


def test_energy_distance_vanishes_and_is_positive():
    x = jax.random.normal(jax.random.PRNGKey(15), (32, 6))
    assert abs(float(energy_distance(x, x))) < 1e-4
    y = jax.random.normal(jax.random.PRNGKey(16), (32, 6)) + 1.0
    assert float(energy_distance(x, y)) > 0.0


def test_interval_levels_are_interior_and_ordered():
    levels = np.asarray(interval_levels(3))
    assert levels.shape == (3,)
    assert (levels > 0.0).all() and (levels < 1.0).all()
    assert (np.diff(levels) > 0).all()


def test_target_interpolant_hits_both_endpoints():
    y = jax.random.normal(jax.random.PRNGKey(17), (4, 3))
    noise = jax.random.normal(jax.random.PRNGKey(18), (4, 3))
    at_zero = target_interpolant(y, noise, jnp.zeros((4,)))
    at_one = target_interpolant(y, noise, jnp.ones((4,)))
    np.testing.assert_allclose(np.asarray(at_zero), np.asarray(noise), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(at_one), np.asarray(y), rtol=1e-6)


def test_entropic_cost_decreases_as_the_blur_shrinks():
    x = jax.random.normal(jax.random.PRNGKey(19), (24, 5))
    y = jax.random.normal(jax.random.PRNGKey(20), (24, 5)) + 0.5
    cost = squared_cost_matrix(x, y)
    loose = float(entropic_ot_cost(x, y, relative_epsilon(cost, 0.5), 600))
    tight = float(entropic_ot_cost(x, y, relative_epsilon(cost, 0.02), 600))
    assert tight < loose, (tight, loose)


if __name__ == "__main__":
    checks = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
