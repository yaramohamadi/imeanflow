"""Stage 0b: can a minibatch Sinkhorn divergence see a domain shift in 4096-D latents?

This is a kill-test for the OT-drift direction and it runs no training. The debiased
Sinkhorn divergence between finite particle sets is a biased, high-variance estimator, and
`imfDiT_XL_2` trains at batch_size 2. If the divergence cannot separate a shifted
distribution from the target's own sampling noise at the particle counts a trainer could
plausibly assemble, no amount of training fixes that and the objective is not usable.

What is measured, on cached SD-VAE latents (32x32x4 = 4096-D, no decode, no generator):

  floor         S(A, A')            two disjoint draws from the target. The noise floor.
  mix alpha     S(mix_alpha, A')    each particle drawn from another domain w.p. alpha.
  sigma s       S(A + s*noise, A')  a known-magnitude perturbation, for calibration.

and the decision quantity

  SNR(condition, N) = [mean S(condition) - mean S(floor)] / std over draws

The alpha axis exists because the honest comparator -- ImageNet-iMF *generated* samples
against the target set -- needs a GPU to sample and all eight here are full. Mixing two
real domains gives a graded, reproducible shift instead, so what this measures is the
*detectability threshold* rather than the specific source-target gap. That threshold is
the number the decision actually turns on: if a small alpha is invisible at N=256, then a
correction that has nearly arrived is invisible too, and the objective stops providing
gradient exactly when it matters.

Epsilon is fixed per N from the noise-floor pair rather than recomputed per condition, so
the values in one row are comparable to each other; a per-condition relative epsilon would
shrink the blur as the gap grows and confound the two effects. `--eps-mode relative`
reproduces what the trainer would do.
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

from otdrift import relative_epsilon, sinkhorn_divergence, squared_cost_matrix

LATENT_SCALE = 0.18215


def load_latent_pool(root, dataset, count, rng, split="train"):
    """Load `count` cached moment tensors and sample latents exactly as training does.

    The cache holds 8-channel (mean, std) moments saved as (C, H, W); the encoder splits on
    the last axis, so they are transposed back to (H, W, C) before splitting. Replicating
    `CachedDiTLatentEncoder.cached_encode` matters -- probing the raw moments instead of a
    sampled latent would measure a distribution the model never sees.
    """
    import torch

    directory = os.path.join(root, f"{dataset}_processed_latents", split)
    names = sorted(os.listdir(directory))
    if len(names) < count:
        raise ValueError(f"{directory} has {len(names)} samples, need {count}.")
    names = names[:count]

    moments = np.empty((count, 32, 32, 8), np.float32)
    for index, name in enumerate(names):
        payload = torch.load(os.path.join(directory, name), map_location="cpu")
        tensor = payload["image"].numpy()  # (C, H, W)
        moments[index] = np.transpose(tensor, (1, 2, 0))

    mean, std = np.split(moments, 2, axis=-1)
    noise = np.asarray(jax.random.normal(rng, mean.shape, jnp.float32))
    return jnp.asarray((mean + std * noise) * LATENT_SCALE)


def make_divergence(num_iters):
    @jax.jit
    def divergence(x, y, epsilon):
        return sinkhorn_divergence(x, y, epsilon, num_iters)

    return divergence


def build_conditions(alphas, sigmas):
    conditions = [("floor", 0.0)]
    conditions += [("mix", float(a)) for a in alphas]
    conditions += [("sigma", float(s)) for s in sigmas]
    return conditions


def draw_pair(rng, pool_target, pool_other, n, kind, level):
    """Return (x, y): y is always a clean target draw, x carries the condition."""
    rng, key_x, key_y = jax.random.split(rng, 3)
    size = pool_target.shape[0]
    if 2 * n > size:
        raise ValueError(f"pool of {size} cannot give two disjoint draws of {n}.")
    order = jax.random.permutation(key_x, size)
    index_x, index_y = order[:n], order[n : 2 * n]
    x = pool_target[index_x]
    y = pool_target[index_y]

    if kind == "floor":
        return x, y
    if kind == "mix":
        rng, key_pick, key_which = jax.random.split(rng, 3)
        other = pool_other[jax.random.choice(key_pick, pool_other.shape[0], (n,), False)]
        swap = jax.random.uniform(key_which, (n,)) < level
        x = jnp.where(swap[:, None, None, None], other, x)
        return x, y
    if kind == "sigma":
        rng, key_noise = jax.random.split(rng)
        return x + level * jax.random.normal(key_noise, x.shape, x.dtype), y
    raise ValueError(f"unknown condition {kind}.")


def repeats_for(n, base, budget):
    """Fewer draws at large N: a single N=2048 solve is ~O(100x) an N=256 one."""
    return max(4, min(base, max(1, budget // max(n, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", default="/opt/dlami/nvme/meanflow/datasets")
    parser.add_argument("--target", default="cub-200-2011")
    parser.add_argument("--other", default="food-101")
    parser.add_argument("--pool", type=int, default=5000, help="latents loaded per domain")
    parser.add_argument("--n-list", default="2,8,32,128,512,2048")
    parser.add_argument("--alphas", default="0.02,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--sigmas", default="0.1,0.25,0.5")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--repeat-budget", type=int, default=6144,
                        help="repeats at size N are capped at budget // N")
    parser.add_argument("--sinkhorn-iters", type=int, default=200)
    parser.add_argument("--eps-rel", type=float, default=0.05)
    parser.add_argument("--eps-mode", default="fixed_from_floor",
                        choices=("fixed_from_floor", "relative"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="per-draw CSV")
    parser.add_argument("--summary", required=True, help="per-(condition, N) SNR CSV")
    args = parser.parse_args()

    sizes = [int(v) for v in args.n_list.split(",") if v]
    alphas = [float(v) for v in args.alphas.split(",") if v]
    sigmas = [float(v) for v in args.sigmas.split(",") if v]
    conditions = build_conditions(alphas, sigmas)

    rng = jax.random.PRNGKey(args.seed)
    rng, key_target, key_other = jax.random.split(rng, 3)
    print(f"loading {args.pool} latents from {args.target} and {args.other} ...", flush=True)
    pool_target = load_latent_pool(args.latent_root, args.target, args.pool, key_target)
    pool_other = load_latent_pool(args.latent_root, args.other, args.pool, key_other)
    print(f"  target {pool_target.shape}  other {pool_other.shape}", flush=True)

    rows = []
    for n in sizes:
        if 2 * n > args.pool:
            print(f"N={n} skipped: pool {args.pool} too small for two disjoint draws")
            continue
        divergence = make_divergence(args.sinkhorn_iters)
        repeats = repeats_for(n, args.repeats, args.repeat_budget)

        # one blur per N, taken from the noise floor, so a row is internally comparable
        rng, key_eps = jax.random.split(rng)
        x_ref, y_ref = draw_pair(key_eps, pool_target, pool_other, n, "floor", 0.0)
        fixed_epsilon = relative_epsilon(squared_cost_matrix(x_ref, y_ref), args.eps_rel)

        for kind, level in conditions:
            started = time.time()
            for repeat in range(repeats):
                rng, key_draw = jax.random.split(rng)
                x, y = draw_pair(key_draw, pool_target, pool_other, n, kind, level)
                if args.eps_mode == "relative":
                    epsilon = relative_epsilon(squared_cost_matrix(x, y), args.eps_rel)
                else:
                    epsilon = fixed_epsilon
                clock = time.time()
                value = float(divergence(x, y, epsilon))
                rows.append({
                    "n": n, "condition": kind, "level": level, "repeat": repeat,
                    "divergence": value, "epsilon": float(epsilon),
                    "seconds": time.time() - clock,
                })
            print(f"  N={n:5d} {kind:5s} {level:<5g} x{repeats:<3d} "
                  f"mean={np.mean([r['divergence'] for r in rows if r['n'] == n and r['condition'] == kind and r['level'] == level]):+.6f} "
                  f"({time.time() - started:.1f}s)", flush=True)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} draws)")

    # SNR against the floor at the same N. The floor's own spread is the denominator: it is
    # the variability a trainer would see between consecutive batches with nothing changing.
    summary = []
    for n in sorted({r["n"] for r in rows}):
        floor = [r["divergence"] for r in rows if r["n"] == n and r["condition"] == "floor"]
        floor_mean, floor_std = float(np.mean(floor)), float(np.std(floor, ddof=1))
        for kind, level in conditions:
            values = [r["divergence"] for r in rows
                      if r["n"] == n and r["condition"] == kind and r["level"] == level]
            spread = max(floor_std, float(np.std(values, ddof=1)))
            summary.append({
                "n": n, "condition": kind, "level": level, "draws": len(values),
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "floor_mean": floor_mean, "floor_std": floor_std,
                "snr": (float(np.mean(values)) - floor_mean) / spread if spread > 0 else float("nan"),
                "median_seconds": float(np.median(
                    [r["seconds"] for r in rows
                     if r["n"] == n and r["condition"] == kind and r["level"] == level])),
            })

    with open(args.summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {args.summary}\n")

    print("SNR vs the same-N noise floor (blank = the floor row itself):")
    header = "condition      " + "".join(f"{n:>10d}" for n in sorted({r['n'] for r in rows}))
    print(header)
    for kind, level in conditions:
        if kind == "floor":
            continue
        cells = ""
        for n in sorted({r["n"] for r in rows}):
            hit = [s for s in summary if s["n"] == n and s["condition"] == kind
                   and s["level"] == level]
            cells += f"{hit[0]['snr']:>10.2f}" if hit else f"{'-':>10}"
        print(f"{kind}={level:<9g}{cells}")


if __name__ == "__main__":
    main()
