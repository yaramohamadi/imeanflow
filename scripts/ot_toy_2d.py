"""Stage 0a: does Sinkhorn adaptation of a pretrained MeanFlow transport work at all?

A 2-D sanity toy, in a setting where the right answer is computable. A MeanFlow average-
velocity net is pretrained on a source GMM, then adapted to a shifted target GMM by six
arms that form a clean 2x2 of (initialisation) x (objective), plus the two trajectory
variants:

  pretrained        no adaptation                                      reference
  regress_scratch   random init  + regression MeanFlow loss on target  proposal row 1
  regress_ft        source init  + regression MeanFlow loss on target  proposal row 2
  ot_final          source init  + Sinkhorn on the 1-step samples      proposal row 3
  ot_traj           source init  + Sinkhorn along the trajectory       proposal row 4 <- the method
  ot_traj_scratch   random init  + Sinkhorn along the trajectory       isolates the prior

What it is for, in order of importance:

1. End-to-end exercise of `frozen_plan_loss` as an actual training signal. The unit tests
   show its gradient is right; they do not show that a generator trained on it moves.
2. Cheap intuition for lambda_traj, epsilon and the particle count before spending H100
   hours.
3. A checkable answer. W2 is measured exactly by `linear_sum_assignment`, not estimated,
   and the learned correction is compared against the exact OT displacement from the
   pretrained model's own output set to the target set -- which is the concrete version of
   the proposal's claim that the target correction is a *small* redirection of the source
   transport.

Conventions follow the **plain-imfDiT adversarial path**, which is what Stage 1 trains and
what `afm.py` is written against: `z_t = (1-t) y + t e`, so **t=0 is data and t=1 is noise**,
`v = e - y`, and `u(x, r, t)` is the average velocity with `x_r = x_t - (t-r) u` (that is
`afm.generated_lower_endpoint`, unchanged). Sampling therefore **descends** t from 1 to 0,
and the lower endpoint r < t is the *cleaner* one -- which is the whole point: at r=0 the
comparator is pure target data, so the trajectory terms are supervised against something
informative. An earlier version of this file used the SiT/DMF convention (t=1 = data) and so
matched against increasingly pure Gaussian noise as r fell; that run was discarded [EXP-085].
The repo contains both conventions -- see `otdrift.target_interpolant`.

A 2-D toy cannot tell us whether Sinkhorn survives 4096 dimensions; that is Stage 0b's job
(`ot_signal_probe.py`). If this toy passes and the probe fails, the probe wins.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from otdrift import (
    frozen_plan_loss,
    generated_lower_endpoint,
    interval_levels,
    linear_path,
    sample_time_pairs,
    target_interpolant,
)

MIN_INTERVAL = 0.05


# --------------------------------------------------------------------------- data


def ring_mixture(rng, count, num_modes, radius, spread, rotation=0.0, offset=(0.0, 0.0)):
    """GMM with modes on a ring. Rotation and offset give a tunable source-target gap."""
    rng_mode, rng_noise = jax.random.split(rng)
    which = jax.random.randint(rng_mode, (count,), 0, num_modes)
    angles = 2.0 * jnp.pi * which / num_modes + rotation
    centres = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)
    centres = centres + jnp.asarray(offset, jnp.float32)[None, :]
    return centres + spread * jax.random.normal(rng_noise, (count, 2), jnp.float32)


def make_samplers(args):
    def source(rng, count):
        return ring_mixture(rng, count, args.num_modes, args.radius, args.spread)

    def target(rng, count):
        return ring_mixture(
            rng, count, args.num_modes, args.radius * args.target_radius_scale,
            args.spread, rotation=args.target_rotation,
            offset=(args.target_offset, 0.0),
        )

    return source, target


# --------------------------------------------------------------------------- model


def init_params(rng, width, depth):
    """Plain MLP on [x (2), t, r]. No flax: the toy should have no moving parts."""
    sizes = [4] + [width] * depth + [2]
    params = []
    for index in range(len(sizes) - 1):
        rng, key = jax.random.split(rng)
        scale = jnp.sqrt(2.0 / sizes[index])
        params.append((
            scale * jax.random.normal(key, (sizes[index], sizes[index + 1]), jnp.float32),
            jnp.zeros((sizes[index + 1],), jnp.float32),
        ))
    return params


def u_fn(params, x, r, t):
    """Average velocity u(x, r, t) -> R^2."""
    hidden = jnp.concatenate([x, t[:, None], r[:, None]], axis=1)
    for weight, bias in params[:-1]:
        hidden = jax.nn.silu(hidden @ weight + bias)
    weight, bias = params[-1]
    return hidden @ weight + bias


def generate(params, z, num_steps):
    """Descend t from 1 (noise) to 0 (data) in `num_steps` equal steps.

    Descending, because t=1 is noise in this convention. Each step is exactly
    `generated_lower_endpoint`: x_r = x_t - (t - r) u. num_steps=1 is the NFE-1 sample,
    x_0 = e - u(e, 0, 1).
    """
    grid = jnp.linspace(1.0, 0.0, num_steps + 1)
    x = z
    for index in range(num_steps):
        t = jnp.full((x.shape[0],), grid[index])
        r = jnp.full((x.shape[0],), grid[index + 1])
        x = generated_lower_endpoint(x, u_fn(params, x, r, t), r, t)
    return x


# --------------------------------------------------------------------------- losses


def meanflow_loss(params, rng, sampler, batch_size):
    """The regression objective: u = v - (t - r) du/dt, with du/dt by JVP along the path."""
    rng_data, rng_noise, rng_time = jax.random.split(rng, 3)
    y = sampler(rng_data, batch_size)
    e = jax.random.normal(rng_noise, (batch_size, 2), jnp.float32)
    r, t, interval, _ = sample_time_pairs(rng_time, batch_size, MIN_INTERVAL)

    # production convention: z_t = (1-t) y + t e, so d z_t / dt = e - y
    x_t = linear_path(y, e, t)
    velocity = e - y

    def along_path(x_arg, t_arg):
        return u_fn(params, x_arg, r, t_arg)

    u, du_dt = jax.jvp(along_path, (x_t, t), (velocity, jnp.ones_like(t)))
    target = jax.lax.stop_gradient(velocity - interval[:, None] * du_dt)
    return jnp.mean((u - target) ** 2)


def ot_loss(params, rng, sampler, batch_size, args, use_trajectory):
    """L_final + lambda_traj * sum_k lambda_k S(x_hat_{r_k}, p_{r_k}^T).

    The final term is the real NFE-1 sample against target data. The trajectory terms use a
    FIXED grid of levels rather than freshly sampled r per particle: with per-particle r the
    transport plan is free to match a generated particle at r=0.1 against a real one at
    r=0.9, which matches a mixture-over-r to a mixture-over-r and is a strictly weaker
    condition than matching each level.

    `interval_levels` is interior, so r=0 never appears here; the r=0 endpoint is exactly the
    final term, with one difference worth keeping in mind -- the final term starts from pure
    noise and runs the model's own one-step map, while a trajectory term starts from an x_t
    built out of *real* target data (AFM's construction). The trajectory terms therefore test
    the interval map on states the model does not itself have to reach.
    """
    rng_z, rng_y, rng_rest = jax.random.split(rng, 3)
    z = jax.random.normal(rng_z, (batch_size, 2), jnp.float32)
    x_hat = generate(params, z, 1)
    y = sampler(rng_y, batch_size)
    loss, aux = frozen_plan_loss(
        x_hat, y, epsilon_relative=args.eps_rel, num_iters=args.sinkhorn_iters
    )
    metrics = {"loss_final": loss, "plan_entropy_ratio": aux["ot_plan_entropy_ratio"],
               "fake_spread": aux["ot_fake_spread"], "real_spread": aux["ot_real_spread"]}

    if not use_trajectory:
        return loss, metrics

    # python floats, not traced array entries: the level indexes a metrics key below
    levels = [float(v) for v in interval_levels(args.num_levels)]
    traj_total = 0.0
    for level in levels:
        rng_rest, key_t, key_y, key_y2, key_z, key_z2 = jax.random.split(rng_rest, 6)
        r = jnp.full((batch_size,), level)
        # t above the level, so u is trained across the interval simplex rather than only
        # at t=1
        span = jnp.maximum(1.0 - level - MIN_INTERVAL, 0.0)
        t = level + MIN_INTERVAL + span * jax.random.uniform(key_t, (batch_size,))

        # x_t = (1-t) y + t e, the production convention: t is the *noisier* end, so the
        # step down to r moves toward data and the comparator at r is informative
        y_upper = sampler(key_y, batch_size)
        z_upper = jax.random.normal(key_z, (batch_size, 2), jnp.float32)
        x_t = linear_path(y_upper, z_upper, t)
        u = u_fn(params, x_t, r, t)
        x_r_hat = generated_lower_endpoint(x_t, u, r, t)

        # independent draws for the comparator: matching against the same y would smuggle
        # in a pointwise correspondence the method is not allowed to have
        y_real = sampler(key_y2, batch_size)
        z_real = jax.random.normal(key_z2, (batch_size, 2), jnp.float32)
        x_r_real = target_interpolant(y_real, z_real, r)  # default = production convention

        level_loss, _ = frozen_plan_loss(
            x_r_hat, x_r_real, epsilon_relative=args.eps_rel,
            num_iters=args.sinkhorn_iters,
        )
        traj_total = traj_total + level_loss
        metrics[f"loss_level_{level:.2f}"] = level_loss

    traj_total = traj_total / len(levels)
    metrics["loss_traj"] = traj_total
    return loss + args.lambda_traj * traj_total, metrics


# --------------------------------------------------------------------------- metrics


def exact_w2(x, y):
    """Squared 2-Wasserstein between equal-size uniform sets, solved exactly."""
    from scipy.optimize import linear_sum_assignment

    x_np, y_np = np.asarray(x, np.float64), np.asarray(y, np.float64)
    cost = ((x_np[:, None, :] - y_np[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean()), cols


def correction_alignment(x_source, x_adapted, y):
    """How close is the learned correction to the exact OT displacement?

    The proposal's claim is that the target transport is the source transport plus a *small*
    correction. The exact OT displacement from the pretrained model's own output set to the
    target set is the smallest correction that could do the job, so cosine ~1 with a norm
    ratio ~1 means the adaptation found it; a much larger norm means it moved further than
    it had to.
    """
    _, assignment = exact_w2(x_source, y)
    ideal = np.asarray(y, np.float64)[assignment] - np.asarray(x_source, np.float64)
    actual = np.asarray(x_adapted, np.float64) - np.asarray(x_source, np.float64)
    ideal_norm = np.linalg.norm(ideal, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    live = (ideal_norm > 1e-8) & (actual_norm > 1e-8)
    if not live.any():
        return float("nan"), float("nan"), float(actual_norm.mean())
    cosine = float(np.mean(
        (ideal[live] * actual[live]).sum(1) / (ideal_norm[live] * actual_norm[live])
    ))
    return cosine, float(actual_norm[live].mean() / ideal_norm[live].mean()), \
        float(actual_norm.mean())


def evaluate(params, base_params, rng, target_sampler, count, nfe_list):
    rng_z, rng_y = jax.random.split(rng)
    z = jax.random.normal(rng_z, (count, 2), jnp.float32)
    y = target_sampler(rng_y, count)
    out = {}
    for nfe in nfe_list:
        value, _ = exact_w2(generate(params, z, nfe), y)
        out[f"w2_nfe{nfe}"] = value
    x_source = generate(base_params, z, 1)
    cosine, ratio, magnitude = correction_alignment(x_source, generate(params, z, 1), y)
    out["ot_alignment_cosine"] = cosine
    out["correction_over_ideal"] = ratio
    out["correction_norm"] = magnitude
    return out


# --------------------------------------------------------------------------- training


def train(name, params, base_params, sampler, target_sampler, steps, lr, args,
          loss_kind, rng, rows, eval_every, seed):
    optimizer = optax.adam(lr)
    state = optimizer.init(params)

    if loss_kind == "regress":
        def objective(p, key):
            return meanflow_loss(p, key, sampler, args.batch_size), {}
    else:
        def objective(p, key):
            return ot_loss(p, key, sampler, args.batch_size, args,
                           loss_kind == "ot_traj")

    @jax.jit
    def step(p, opt_state, key):
        (value, metrics), grads = jax.value_and_grad(objective, has_aux=True)(p, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(p, updates), opt_state, value, metrics

    started = time.time()
    for index in range(steps + 1):
        # `or index == steps`: without it an eval_every that does not divide `steps` means
        # the converged model is never measured, and the smoke run reported the source
        # transport at step 0 as if that were its quality
        if index % eval_every == 0 or index == steps:
            rng, key_eval = jax.random.split(rng)
            row = {"arm": name, "seed": seed, "step": index,
                   "seconds": time.time() - started}
            row.update(evaluate(params, base_params, key_eval, target_sampler,
                                args.eval_count, [1, 4]))
            rows.append(row)
            print(f"  s{seed} {name:16s} step {index:6d}  "
                  f"W2(NFE1)={row['w2_nfe1']:.4f}  W2(NFE4)={row['w2_nfe4']:.4f}  "
                  f"cos={row['ot_alignment_cosine']:+.3f}  "
                  f"|d|/ideal={row['correction_over_ideal']:.2f}", flush=True)
        if index == steps:
            break
        rng, key_step = jax.random.split(rng)
        params, state, _, _ = step(params, state, key_step)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--target-rotation", type=float, default=0.5236,
                        help="radians; default pi/6 puts target modes between source ones")
    parser.add_argument("--target-offset", type=float, default=1.0)
    parser.add_argument("--target-radius-scale", type=float, default=1.2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pretrain-steps", type=int, default=8000)
    parser.add_argument("--adapt-steps", type=int, default=4000)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-lr", type=float, default=3e-4)
    parser.add_argument("--eps-rel", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--lambda-traj", type=float, default=1.0)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seeds", default="0,1,2",
                        help="one seed is not a result; each seed repeats the whole "
                             "pretrain + all arms, so the prior differs per seed too")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="pretrained,regress_scratch,regress_ft,"
                                          "ot_final,ot_traj,ot_traj_scratch")
    args = parser.parse_args()

    source_sampler, target_sampler = make_samplers(args)
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(v) for v in args.seeds.split(",") if v]
    rows = []

    for seed in seeds:
        rng = jax.random.PRNGKey(seed)
        rng, key_init = jax.random.split(rng)
        source_params = init_params(key_init, args.width, args.depth)

        print(f"\n=== seed {seed}: pretraining the source transport "
              f"({args.pretrain_steps} steps) ===", flush=True)
        source_params = train(
            "source_pretrain", source_params, source_params, source_sampler,
            source_sampler, args.pretrain_steps, args.pretrain_lr, args, "regress",
            rng, rows, max(args.eval_every * 4, 1), seed,
        )

        rng, key_scratch = jax.random.split(rng)
        scratch_params = init_params(key_scratch, args.width, args.depth)

        plans = {
            "pretrained": (source_params, "regress", 0),
            "regress_scratch": (scratch_params, "regress", args.adapt_steps),
            "regress_ft": (source_params, "regress", args.adapt_steps),
            "ot_final": (source_params, "ot_final", args.adapt_steps),
            "ot_traj": (source_params, "ot_traj", args.adapt_steps),
            "ot_traj_scratch": (scratch_params, "ot_traj", args.adapt_steps),
        }
        for arm in arms:
            if arm not in plans:
                raise ValueError(f"unknown arm {arm}")
            init, kind, steps = plans[arm]
            rng, key_arm = jax.random.split(rng)
            train(arm, init, source_params, target_sampler, target_sampler, steps,
                  args.adapt_lr, args, kind, key_arm, rows, args.eval_every, seed)

    fields = sorted({key for row in rows for key in row})
    lead = ["arm", "seed", "step"]
    fields = lead + [f for f in fields if f not in lead]
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    print(f"\nfinal state per arm, mean +- sd over {len(seeds)} seeds "
          f"(W2 is exact, lower is better):")
    print(f"{'arm':18s}{'W2 NFE1':>18s}{'W2 NFE4':>18s}{'OT cos':>18s}{'|d|/ideal':>18s}")
    for arm in ["source_pretrain"] + arms:
        finals = []
        for seed in seeds:
            hits = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            if hits:
                finals.append(max(hits, key=lambda r: r["step"]))
        if not finals:
            continue

        def spread(key):
            values = np.array([f[key] for f in finals], np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return "        n/a"
            return f"{values.mean():>10.4f}+-{values.std(ddof=0):<6.3f}"

        print(f"{arm:18s}{spread('w2_nfe1')}{spread('w2_nfe4')}"
              f"{spread('ot_alignment_cosine')}{spread('correction_over_ideal')}")


if __name__ == "__main__":
    main()
