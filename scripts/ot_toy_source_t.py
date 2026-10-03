"""Stage 0c: the redirected method -- one OT step from a source state at a FIXED noise level.

This is not `ot_toy_2d.py`'s `ot_traj`. That arm supervised a MeanFlow model at several
interval levels and still *generated from noise* at test time. The object here is different:

    G_theta(x, t)  :  a source-domain state at noise level t  ->  a clean TARGET image, 1 step

with `t` an input, held FIXED per arm. Each arm answers "if the frozen source model hands us
a state at level t, how well can one OT step finish the job in the target domain?"

Train/test asymmetry is deliberate and is what the user specified:
  * TRAIN input  = forward-noised source-domain data, x_t = (1-t) y_S + t e.  No source model
    in the training loop at all.
  * TEST input   = the frozen source model's own partial generation, run from t=1 down to t.
The two agree only if the source model's partial path lands on the forward-noise marginal.
That gap is a cost of the design, so every arm is evaluated BOTH ways and the difference is
reported (`w2_forward` vs `w2_onpolicy_src{1,2,4}`).

Time convention is the production one, as in `ot_toy_2d.py`: z_t = (1-t) y + t e, so **t=1 is
noise and t=0 is clean data**. So t=1 means "source model did nothing, we start from noise"
and t=0 means "source model finished, we start from a clean source image".

## Why the head had to change, and what that costs

MeanFlow's one-step map is x_r = x_t - (t - r) u, so at r=0 it is

    G_theta(x, t) = x - t * u_theta(x, 0, t)        ("meanflow" head)

which is **identically the identity at t=0 for every theta**: the displacement is forced to
vanish exactly where this method needs it to equal the source->target domain gap. That is not
a tuning problem, it is the parameterisation being wrong for the job. So the arms use

    G_theta(x, t) = x - u_theta(x, 0, t)            ("unit" head)

which is well-posed at every t and coincides with the source model's own NFE-1 generation at
t=1 (where the two heads are the same function). The price is paid at the init: the source
weights' correct displacement at level t is t*u, so read as a unit displacement they overshoot
by 1/t. The init is therefore exact at t=1 and progressively mis-scaled below it. That is a
property of the arms, not a bug, and `mf_head_*` arms are included so the claim is measured
rather than asserted -- `mf_head_t1` should tie `src_t1` (same function), and `mf_head_t0`
should sit exactly at the no-op W2 (the predicted null that confirms the degeneracy).

## Metrics

  mmd_forward            multi-scale MMD^2 to target data, train-matched input. (EXP-126/127
                         logged exact sample W2^2 here; dropped, it is dominated by chance
                         per-mode counts at these n.)
  mmd_onpolicy_src{1,2,4} the same, with the input produced on-policy by the frozen source
                         model at 1/2/4 steps. At t=1 there is nothing for the source model
                         to do, so these equal mmd_forward by construction.
  mode_mi                mutual information (bits) between the input's source mode and the
                         output's nearest target mode, max log2(6)=2.585, chance 0. This is
                         the "does it ignore the input" test: a model that discards x and
                         just emits a target sample scores 0 while still scoring a good W2.
                         Undefined at t=1 (pure noise has no source mode).
  out_spread             per-coordinate std of the generated set, against `target_spread`.
                         Guards the classic minibatch-OT diversity collapse.

Total NFE at test time is (source steps) + 1, so the t<1 arms cost more than CAMF's 1. They
have to earn it; that is what the figure is for.
"""

import argparse
import csv
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import optax

import ot_toy_2d as toy
from otdrift import frozen_plan_loss, generated_lower_endpoint

SRC_NFE = (1, 2, 4)


# --------------------------------------------------------------------------- the map


def one_step(params, x, t_level, head):
    """G(x, t): one step from a level-t source state to a clean target image."""
    t = jnp.full((x.shape[0],), t_level, jnp.float32)
    u = toy.u_fn(params, x, jnp.zeros_like(t), t)
    if head == "meanflow":
        return generated_lower_endpoint(x, u, jnp.zeros_like(t), t)
    return x - u


def forward_noised(rng, source_sampler, count, t_level):
    """The TRAINING input: x_t = (1-t) y_S + t e. Returns the state and its mode labels."""
    key_y, key_e = jax.random.split(rng)
    y, labels = source_sampler(key_y, count)
    e = jax.random.normal(key_e, (count, 2), jnp.float32)
    return (1.0 - t_level) * y + t_level * e, labels


def source_state(base_params, z, t_level, num_steps):
    """The TEST input: the frozen source model run from t=1 down to `t_level`."""
    if t_level >= 1.0:
        return z
    grid = jnp.linspace(1.0, t_level, num_steps + 1)
    x = z
    for index in range(num_steps):
        t = jnp.full((x.shape[0],), grid[index])
        r = jnp.full((x.shape[0],), grid[index + 1])
        x = generated_lower_endpoint(x, toy.u_fn(base_params, x, r, t), r, t)
    return x


# --------------------------------------------------------------------------- data


def labelled_samplers(args):
    """Like `toy.make_samplers`, but the source sampler also returns the mode index.

    `mode_mi` needs to know which source mode each training input came from, and recovering
    it after the fact by nearest-centre would launder a guess into the metric.
    """
    def ring(rng, count, radius, rotation, offset):
        key_mode, key_noise = jax.random.split(rng)
        which = jax.random.randint(key_mode, (count,), 0, args.num_modes)
        angles = 2.0 * jnp.pi * which / args.num_modes + rotation
        centres = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)
        centres = centres + jnp.asarray([offset, 0.0], jnp.float32)[None, :]
        return centres + args.spread * jax.random.normal(key_noise, (count, 2),
                                                         jnp.float32), which

    def source(rng, count):
        return ring(rng, count, args.radius, 0.0, 0.0)

    def target(rng, count):
        return ring(rng, count, args.radius * args.target_radius_scale,
                    args.target_rotation, args.target_offset)

    return source, target


def target_centres(args):
    index = np.arange(args.num_modes, dtype=np.float64)
    angles = 2.0 * np.pi * index / args.num_modes + args.target_rotation
    radius = args.radius * args.target_radius_scale
    return np.stack([radius * np.cos(angles) + args.target_offset,
                     radius * np.sin(angles)], axis=1)


# --------------------------------------------------------------------------- loss


def ot_step_loss(params, rng, source_sampler, target_sampler, t_level, head, args):
    """Debiased Sinkhorn between the one-step outputs and an independent clean target batch.

    The comparator is clean target data, never an interpolant: this map is supposed to land on
    the target distribution in one step, so matching it against anything noisier would score a
    weaker condition than the one we claim.
    """
    key_in, key_y = jax.random.split(rng)
    x_t, _ = forward_noised(key_in, source_sampler, args.batch_size, t_level)
    x_hat = one_step(params, x_t, t_level, head)
    y, _ = target_sampler(key_y, args.batch_size)
    loss, aux = frozen_plan_loss(x_hat, y, epsilon_relative=args.eps_rel,
                                 num_iters=args.sinkhorn_iters)
    return loss, {"plan_entropy_ratio": aux["ot_plan_entropy_ratio"],
                  "fake_spread": aux["ot_fake_spread"],
                  "real_spread": aux["ot_real_spread"]}


# --------------------------------------------------------------------------- metrics


def mode_mutual_info(source_labels, outputs, centres):
    """I(source mode ; nearest target mode) in bits. 0 = the output ignores the input."""
    out = np.asarray(outputs, np.float64)
    assigned = np.argmin(((out[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)
    labels = np.asarray(source_labels, np.int64)
    k = centres.shape[0]
    joint = np.zeros((k, k), np.float64)
    np.add.at(joint, (labels, assigned), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    live = joint > 0
    return float(np.sum(joint[live] * np.log2(joint[live] / (px @ py)[live])))


def frechet(x, y):
    """FID-style W2^2 between Gaussians fitted to the two sets. No point matching, so unlike
    `exact_w2` it is not dominated by chance per-mode counts at small n."""
    from scipy.linalg import sqrtm

    x_np, y_np = np.asarray(x, np.float64), np.asarray(y, np.float64)
    cov_x, cov_y = np.cov(x_np.T), np.cov(y_np.T)
    return float(((x_np.mean(0) - y_np.mean(0)) ** 2).sum()
                 + np.trace(cov_x + cov_y - 2.0 * np.real(sqrtm(cov_x @ cov_y))))


def grid_nfe(t_level, step):
    """Source steps from t=1 down to t at a fixed step size; 0 at t=1."""
    return int(round((1.0 - t_level) / step))


def evaluate(params, base_params, rng, source_sampler, target_sampler, t_level, head,
             centres, args):
    key_in, key_z, key_y = jax.random.split(rng, 3)
    y, _ = target_sampler(key_y, args.eval_count)
    y_np = np.asarray(y, np.float64)

    x_fwd, labels = forward_noised(key_in, source_sampler, args.eval_count, t_level)
    out_fwd = one_step(params, x_fwd, t_level, head)
    row = {"fd_forward": frechet(out_fwd, y),
           "mode_mi": mode_mutual_info(labels, out_fwd, centres),
           "out_spread": float(np.asarray(out_fwd, np.float64).std(0).mean()),
           "target_spread": float(y_np.std(0).mean())}

    z = jax.random.normal(key_z, (args.eval_count, 2), jnp.float32)
    for nfe in SRC_NFE:
        state = source_state(base_params, z, t_level, nfe)
        out_on = one_step(params, state, t_level, head)
        row[f"fd_onpolicy_src{nfe}"] = frechet(out_on, y)
        row[f"mmd_onpolicy_src{nfe}"] = toy.mmd_multi(out_on, y)[0]
    if args.src_step > 0:
        # fixed source step size: t=0.25 at step 0.25 is 3 source steps, then the OT step
        nfe = grid_nfe(t_level, args.src_step)
        out_on = one_step(params, source_state(base_params, z, t_level, nfe), t_level, head)
        row["src_grid_nfe"] = nfe
        row["fd_onpolicy_grid"] = frechet(out_on, y)
        row["mmd_onpolicy_grid"], per = toy.mmd_multi(out_on, y)
        for sigma, v in zip(toy.MMD_SIGMAS, per):
            row[f"mmd{sigma:g}_onpolicy_grid"] = v
    # how far the test-time input is from the one training showed the model
    row["input_shift_mmd"] = toy.mmd_multi(
        source_state(base_params, z, t_level, max(SRC_NFE)), x_fwd)[0]
    row["mmd_forward"], per = toy.mmd_multi(out_fwd, y)
    for sigma, v in zip(toy.MMD_SIGMAS, per):
        row[f"mmd{sigma:g}_forward"] = v
    return row


# --------------------------------------------------------------------------- training


def run_arm(name, params, base_params, source_sampler, target_sampler, t_level, head,
            centres, args, rng, rows, seed):
    optimizer = optax.adam(args.adapt_lr)
    state = optimizer.init(params)

    @jax.jit
    def step(p, opt_state, key):
        def objective(q):
            return ot_step_loss(q, key, source_sampler, target_sampler, t_level, head, args)
        (value, metrics), grads = jax.value_and_grad(objective, has_aux=True)(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(p, updates), opt_state, value, metrics

    started = time.time()
    for index in range(args.adapt_steps + 1):
        if index % args.eval_every == 0 or index == args.adapt_steps:
            rng, key_eval = jax.random.split(rng)
            row = {"arm": name, "seed": seed, "step": index, "t_level": t_level,
                   "head": head, "seconds": time.time() - started}
            row.update(evaluate(params, base_params, key_eval, source_sampler,
                                target_sampler, t_level, head, centres, args))
            rows.append(row)
            print(f"  s{seed} {name:14s} step {index:5d}  "
                  f"MMD(fwd)={row['mmd_forward']:.2e}  "
                  f"FD(fwd)={row['fd_forward']:.4f}  "
                  f"FD(src1)={row['fd_onpolicy_src1']:.4f}  "
                  f"MI={row['mode_mi']:.3f}b  "
                  f"spread={row['out_spread']:.2f}/{row['target_spread']:.2f}", flush=True)
        if index == args.adapt_steps:
            break
        rng, key_step = jax.random.split(rng)
        params, state, _, _ = step(params, state, key_step)
    return params


# the arms. (t level, head, init) -- `init` is "source" or "scratch".
ARMS = {
    "src_t1":      (1.0, "unit", "source"),
    "src_t0.5":    (0.5, "unit", "source"),
    "src_t0":      (0.0, "unit", "source"),
    "scratch_t1":  (1.0, "unit", "scratch"),
    "scratch_t0":  (0.0, "unit", "scratch"),
    "mf_head_t1":  (1.0, "meanflow", "source"),
    "mf_head_t0":  (0.0, "meanflow", "source"),
}


def summarise(rows, arms, seeds):
    keys = ["mmd_forward"] + [f"mmd_onpolicy_src{n}" for n in SRC_NFE] + ["fd_forward"] + \
           ["mode_mi", "out_spread", "input_shift_mmd"]
    print("\nfinal eval per arm, mean +- sd over %d seeds (multi-scale MMD^2 and FD, lower better):" %
          len(seeds))
    print(f"{'arm':12s}{'t':>5s}{'head':>10s}" + "".join(f"{k:>20s}" for k in keys))
    for arm in arms:
        chosen = []
        for seed in seeds:
            hits = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            if hits:
                chosen.append(max(hits, key=lambda r: r["step"]))
        if not chosen:
            continue
        cells = []
        for key in keys:
            values = np.array([row[key] for row in chosen], np.float64)
            values = values[np.isfinite(values)]
            cells.append("                 n/a" if values.size == 0 else
                         f"{values.mean():>12.4f}+-{values.std(ddof=0):<6.3f}")
        t_level, head, _ = ARMS[arm]
        print(f"{arm:12s}{t_level:>5.2f}{head:>10s}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("num-modes", 6), ("width", 128), ("depth", 3),
                          ("batch-size", 512), ("pretrain-steps", 8000),
                          ("adapt-steps", 4000), ("sinkhorn-iters", 100),
                          ("eval-count", 1024), ("eval-every", 1000)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in [("radius", 3.0), ("spread", 0.25), ("target-rotation", 0.5236),
                          ("target-offset", 1.0), ("target-radius-scale", 1.2),
                          ("pretrain-lr", 1e-3), ("adapt-lr", 3e-4), ("eps-rel", 0.05)]:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="each seed repeats the source pretrain too, so the prior "
                             "differs per seed; 3 seeds could not rank EXP-086's arms")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--t-levels", default="",
                        help="extra source-init unit-head arms src_t<t>, e.g. 0,0.25,0.5")
    parser.add_argument("--dump-src-nfe", type=int, default=max(SRC_NFE),
                        help="source-model steps for the dumped on-policy input")
    parser.add_argument("--src-step", type=float, default=0.0,
                        help="if > 0, also eval (and dump) the on-policy input made with a "
                             "fixed source step size, i.e. round((1-t)/step) steps")
    parser.add_argument("--dump-count", type=int, default=2048)
    parser.add_argument("--save-params", default="",
                        help="optional folder; each trained arm is pickled per seed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump", default="", help="optional .npz of seed-0 point clouds")
    args = parser.parse_args()

    source_sampler, target_sampler = labelled_samplers(args)
    plain_source, _ = toy.make_samplers(args)   # unlabelled, for the MeanFlow pretrain
    centres = target_centres(args)
    for level in [float(v) for v in args.t_levels.split(",") if v]:
        ARMS.setdefault(f"src_t{level:g}", (level, "unit", "source"))
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(v) for v in args.seeds.split(",") if v]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}; known: {sorted(ARMS)}")
    rows, bundle = [], {}

    for seed in seeds:
        rng = jax.random.PRNGKey(seed)
        rng, key_init = jax.random.split(rng)
        source_params = toy.init_params(key_init, args.width, args.depth)

        print(f"\n=== seed {seed}: pretraining the source transport "
              f"({args.pretrain_steps} steps) ===", flush=True)
        source_params = toy.train(
            "source_pretrain", source_params, source_params, plain_source, plain_source,
            args.pretrain_steps, args.pretrain_lr, args, "regress", rng, [],
            args.pretrain_steps, seed,
        )

        rng, key_scratch = jax.random.split(rng)
        scratch_params = toy.init_params(key_scratch, args.width, args.depth)

        if args.dump and seed == seeds[0]:
            rng, key_src, key_tgt = jax.random.split(rng, 3)
            bundle["real_source"] = np.asarray(source_sampler(key_src, args.dump_count)[0])
            bundle["real_target"] = np.asarray(target_sampler(key_tgt, args.dump_count)[0])

        for arm in arms:
            t_level, head, init = ARMS[arm]
            rng, key_arm = jax.random.split(rng)
            print(f"\n--- seed {seed} {arm}: t={t_level} head={head} init={init} ---",
                  flush=True)
            trained = run_arm(
                arm, source_params if init == "source" else scratch_params, source_params,
                source_sampler, target_sampler, t_level, head, centres, args, key_arm,
                rows, seed,
            )
            if args.save_params:
                os.makedirs(args.save_params, exist_ok=True)
                with open(os.path.join(args.save_params, f"{arm}_seed{seed}.pkl"), "wb") as f:
                    pickle.dump({"trained": jax.device_get(trained),
                                 "source": jax.device_get(source_params)}, f)
            if args.dump and seed == seeds[0]:
                rng, key_in, key_z = jax.random.split(rng, 3)
                x_fwd, _ = forward_noised(key_in, source_sampler, args.dump_count, t_level)
                z = jax.random.normal(key_z, (args.dump_count, 2), jnp.float32)
                nfe = (grid_nfe(t_level, args.src_step) if args.src_step > 0
                       else args.dump_src_nfe)
                state = source_state(source_params, z, t_level, nfe)
                bundle[f"{arm}_in_forward"] = np.asarray(x_fwd)
                bundle[f"{arm}_in_onpolicy"] = np.asarray(state)
                bundle[f"{arm}_out_forward"] = np.asarray(
                    one_step(trained, x_fwd, t_level, head))
                bundle[f"{arm}_out_onpolicy"] = np.asarray(
                    one_step(trained, state, t_level, head))

    fields = ["arm", "seed", "step", "t_level", "head"]
    fields += sorted({k for row in rows for k in row} - set(fields))
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    if args.dump:
        np.savez_compressed(args.dump, **bundle)
        print(f"wrote {args.dump} ({len(bundle)} arrays)")

    summarise(rows, arms, seeds)


if __name__ == "__main__":
    main()
