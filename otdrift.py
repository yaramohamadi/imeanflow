"""Pure losses and utilities for OT-drift adaptation of a pretrained transport.

Distribution-level (entropic OT) supervision, as the critic-free counterpart to the
adversarial post-training in `caimf.py` / `afm.py`. Interval sampling and the generated
lower endpoint are imported from `afm.py` rather than re-derived, so the OT arms sit on
exactly the same trajectory construction as the adversarial arms.

Two loss entry points, and they are not interchangeable:

* `sinkhorn_divergence` differentiates through the Sinkhorn iterations. Correct, but it
  needs every particle in one graph, so it is for the toy and for tests.
* `frozen_plan_loss` solves the OT problem under `stop_gradient` and returns a
  per-particle quadratic target. By the envelope theorem this carries the same gradient,
  and because it is decomposable over generated particles it can be accumulated in
  microbatches -- which is what makes a 256-particle divergence affordable for a model
  that trains at batch 2.
"""

import jax
import jax.numpy as jnp
import numpy as np

from afm import generated_lower_endpoint, linear_path, sample_time_pairs  # noqa: F401


def flatten_particles(x):
    """(B, ...) -> (B, D)."""
    return x.reshape(x.shape[0], -1)


def squared_cost_matrix(x, y):
    """Pairwise squared euclidean cost between flattened particle sets, per dimension.

    Normalising by D keeps `sinkhorn_eps` comparable across latent shapes and across
    latent- versus feature-space runs.
    """
    x_flat = flatten_particles(x)
    y_flat = flatten_particles(y)
    if x_flat.shape[1] != y_flat.shape[1]:
        raise ValueError(
            f"particle dimension mismatch: {x_flat.shape[1]} vs {y_flat.shape[1]}."
        )
    dimension = x_flat.shape[1]
    squared = (
        jnp.sum(x_flat**2, axis=1)[:, None]
        + jnp.sum(y_flat**2, axis=1)[None, :]
        - 2.0 * (x_flat @ y_flat.T)
    )
    return jnp.maximum(squared, 0.0) / dimension


def _uniform_log_weights(size, dtype):
    return jnp.full((size,), -jnp.log(size), dtype=dtype)


def sinkhorn_potentials(cost, epsilon, num_iters, log_a=None, log_b=None):
    """Log-domain Sinkhorn (soft-min form). Returns the dual potentials (f, g).

    Each update is a soft-min over the *opposite* marginal:

        f_i <- -eps * logsumexp_j[ log b_j + (g_j - C_ij) / eps ]

    The weight inside the reduction has to be the marginal being summed over, which is
    what makes exp(f_i / eps) the exact reciprocal of the row scaling and hence makes the
    plan below satisfy sum_j P_ij = a_i. Putting `log a` there instead still converges to
    a fixed point, but one whose plan is off by a factor of n*m -- silently, since the
    rows stay uniform among themselves.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1.")
    try:
        positive = bool(epsilon > 0.0)
    except jax.errors.TracerBoolConversionError:
        positive = True  # traced under jit, so the value is not inspectable here
    if not positive:
        raise ValueError("epsilon must be positive.")
    n, m = cost.shape
    if log_a is None:
        log_a = _uniform_log_weights(n, cost.dtype)
    if log_b is None:
        log_b = _uniform_log_weights(m, cost.dtype)

    def body(carry, _):
        f, g = carry
        f = -epsilon * jax.nn.logsumexp(
            log_b[None, :] + (g[None, :] - cost) / epsilon, axis=1
        )
        g = -epsilon * jax.nn.logsumexp(
            log_a[:, None] + (f[:, None] - cost) / epsilon, axis=0
        )
        return (f, g), None

    init = (jnp.zeros((n,), cost.dtype), jnp.zeros((m,), cost.dtype))
    (f, g), _ = jax.lax.scan(body, init, None, length=num_iters)
    return f, g


def transport_plan(cost, epsilon, num_iters, log_a=None, log_b=None):
    """Entropic transport plan; rows sum to 1/n."""
    n, m = cost.shape
    if log_a is None:
        log_a = _uniform_log_weights(n, cost.dtype)
    if log_b is None:
        log_b = _uniform_log_weights(m, cost.dtype)
    f, g = sinkhorn_potentials(cost, epsilon, num_iters, log_a, log_b)
    log_plan = (
        log_a[:, None] + log_b[None, :] + (f[:, None] + g[None, :] - cost) / epsilon
    )
    return jnp.exp(log_plan)


def matched_cost(x, y, epsilon, num_iters):
    """<P*, C>: the mass-weighted transported cost. A diagnostic, not the objective."""
    cost = squared_cost_matrix(x, y)
    plan = transport_plan(cost, epsilon, num_iters)
    return jnp.sum(plan * cost)


def entropic_ot_cost(x, y, epsilon, num_iters):
    """Regularised entropic OT value <P*, C> + eps * KL(P* | a (x) b), read off the duals.

    The duals and not the raw transported cost, because it is *this* value whose
    derivative with respect to the cost matrix is exactly P*. That is the whole basis of
    `frozen_plan_loss`: with the regularised value the frozen plan carries the true
    gradient, whereas differentiating <P*, C> leaves a <dP/dx, C> term behind and the two
    disagree (~0.96 cosine on a 24-particle toy, which is close enough to look right and
    wrong enough to matter).
    """
    cost = squared_cost_matrix(x, y)
    f, g = sinkhorn_potentials(cost, epsilon, num_iters)
    # uniform marginals, so <f, a> + <g, b> is just the two means
    return jnp.mean(f) + jnp.mean(g)


def sinkhorn_divergence(x, y, epsilon, num_iters, debias=True):
    """Debiased Sinkhorn divergence OT_eps(x, y) - 0.5 OT_eps(x, x) - 0.5 OT_eps(y, y).

    Differentiable through the iterations. The self terms are what stop the loss from
    collapsing the particle set onto the target barycentre; `debias=False` exists only to
    measure how bad that is.
    """
    value = entropic_ot_cost(x, y, epsilon, num_iters)
    if not debias:
        return value
    value = value - 0.5 * entropic_ot_cost(x, x, epsilon, num_iters)
    value = value - 0.5 * entropic_ot_cost(y, y, epsilon, num_iters)
    return value


def relative_epsilon(cost, epsilon_relative):
    """Blur as a fraction of the mean cost, so one setting transfers across spaces."""
    return epsilon_relative * jnp.maximum(jnp.mean(cost), 1e-12)


def barycentric_targets(plan, y):
    """Per-row weight and barycentric projection of `y` under `plan`.

    sum_j P_ij |x_i - y_j|^2 = w_i |x_i - ybar_i|^2 + (terms with no x dependence),
    so the attraction term is exactly a weighted MSE against `ybar` once `plan` is
    detached. This identity is what makes the loss microbatchable.
    """
    weights = jnp.sum(plan, axis=1)
    y_flat = flatten_particles(y)
    projected = (plan @ y_flat) / jnp.maximum(weights, 1e-12)[:, None]
    return weights, projected.reshape((plan.shape[0],) + y.shape[1:])


def frozen_plan_loss(
    x_hat,
    y,
    *,
    epsilon_relative=0.05,
    epsilon=None,
    num_iters=100,
    debias=True,
    repulsion_clip=10.0,
):
    """Envelope-theorem Sinkhorn loss: solve OT without gradient, regress with gradient.

    `x_hat` carries the gradient; the plans, the weights and the barycentric targets are
    all `stop_gradient`. Returns (loss, aux) with the diagnostics that tell us whether
    the objective is doing anything: plan entropy (is the blur so large that the plan is
    uniform and the gradient is just "move to the mean"?) and particle spread (is the
    debiasing term actually holding diversity up?).

    `epsilon` pins an absolute blur; otherwise it is `epsilon_relative` times the mean
    cross cost, which keeps one setting usable across latent and feature spaces. Either
    way the *same* epsilon is used for the cross and self terms -- the debiasing only
    cancels under a common blur, and giving the self term its own relative epsilon quietly
    breaks the divergence.
    """
    cost_xy = squared_cost_matrix(jax.lax.stop_gradient(x_hat), y)
    if epsilon is None:
        epsilon = relative_epsilon(cost_xy, epsilon_relative)
    plan_xy = jax.lax.stop_gradient(transport_plan(cost_xy, epsilon, num_iters))
    weights_xy, targets_xy = barycentric_targets(plan_xy, jax.lax.stop_gradient(y))
    weights_xy = jax.lax.stop_gradient(weights_xy)
    targets_xy = jax.lax.stop_gradient(targets_xy)

    reduce_axes = tuple(range(1, x_hat.ndim))
    dimension = 1
    for size in x_hat.shape[1:]:
        dimension *= size
    attraction = jnp.sum(
        weights_xy * jnp.sum((x_hat - targets_xy) ** 2, axis=reduce_axes) / dimension
    )

    loss = attraction
    repulsion = jnp.asarray(0.0, jnp.float32)
    if debias:
        x_detached = jax.lax.stop_gradient(x_hat)
        cost_xx = squared_cost_matrix(x_detached, x_detached)
        plan_xx = jax.lax.stop_gradient(transport_plan(cost_xx, epsilon, num_iters))
        weights_xx, targets_xx = barycentric_targets(plan_xx, x_detached)
        weights_xx = jax.lax.stop_gradient(weights_xx)
        targets_xx = jax.lax.stop_gradient(targets_xx)
        # One factor of 2 because both indices of the self term depend on x_hat; the
        # other copy is detached so the term stays decomposable over particles.
        repulsion = jnp.sum(
            weights_xx
            * jnp.sum((x_hat - targets_xx) ** 2, axis=reduce_axes)
            / dimension
        )
        # The repulsion is a negative quadratic and is unbounded below on its own. The
        # plan is re-solved every step so it does not run away in practice, but clip the
        # ratio anyway: without this a single step with a near-degenerate plan can
        # dominate the update.
        bound = repulsion_clip * jax.lax.stop_gradient(attraction)
        repulsion = jnp.minimum(repulsion, bound)
        loss = loss - repulsion

    plan_entropy = -jnp.sum(plan_xy * jnp.log(jnp.maximum(plan_xy, 1e-30)))
    uniform_entropy = jnp.log(plan_xy.shape[0] * plan_xy.shape[1])
    x_flat = flatten_particles(jax.lax.stop_gradient(x_hat))
    y_flat = flatten_particles(y)
    aux = {
        "loss_ot": loss,
        "ot_attraction": attraction,
        "ot_repulsion": repulsion,
        "ot_epsilon": epsilon,
        "ot_matched_cost": jnp.sum(plan_xy * cost_xy),
        "ot_plan_entropy_ratio": plan_entropy / uniform_entropy,
        "ot_fake_spread": jnp.mean(jnp.std(x_flat, axis=0)),
        "ot_real_spread": jnp.mean(jnp.std(y_flat, axis=0)),
    }
    return loss, aux


def energy_distance(x, y):
    """Debiased energy distance -- the divergence-family control for Stage 2a."""
    cost_xy = squared_cost_matrix(x, y)
    cost_xx = squared_cost_matrix(x, x)
    cost_yy = squared_cost_matrix(y, y)
    return (
        2.0 * jnp.mean(jnp.sqrt(cost_xy + 1e-12))
        - jnp.mean(jnp.sqrt(cost_xx + 1e-12))
        - jnp.mean(jnp.sqrt(cost_yy + 1e-12))
    )


def interval_levels(num_levels, minimum=0.0, maximum=1.0):
    """Fixed grid of trajectory levels at which the distributions are matched.

    Deliberately numpy and not jnp: these levels are static configuration, and under
    omnistaging a `jnp.linspace` here becomes a tracer inside any jitted step, so a caller
    that wants to loop over the levels (to key a per-level metric, or to unroll K Sinkhorn
    solves) cannot read them. Static things stay static.
    """
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1.")
    return np.linspace(minimum, maximum, num_levels + 2)[1:-1].astype(np.float32)


def target_interpolant(y, noise, time):
    """p_t^T for a linear-path MeanFlow: (1 - t) * noise + t * y.

    This is the concrete answer to the proposal's "what is the target path for a distilled
    model" question -- for a linear-path transport the target intermediate marginal is
    available in closed form and needs no diffusion noising process.
    """
    return linear_path(noise, y, time)
