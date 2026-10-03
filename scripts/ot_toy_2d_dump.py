"""Dump the Stage-0a toy's particle sets so the 2-D distributions can be plotted.

`ot_toy_2d.py` records metrics only, which is right for the experiment record but leaves
nothing to look at. This re-runs one seed of the same pipeline, with the same defaults, and
saves the actual point clouds: the source and target GMMs, and every arm's generated set at
NFE 1 and NFE 4.

It imports `ot_toy_2d` rather than re-deriving anything, so the samples it writes are the
same construction the metrics came from. It does NOT write to the experiment record -- the
numbers of record are EXP-086's; this is a visualisation of the same process re-run, and the
W2 it prints will differ from EXP-086's by eval noise.
"""

import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np

import ot_toy_2d as toy


# The toy's argparse defaults, duplicated here as data. Keep in sync with
# `ot_toy_2d.main`; a mismatch means the figures describe a different experiment.
DEFAULTS = dict(
    num_modes=6, radius=3.0, spread=0.25, target_rotation=0.5236, target_offset=1.0,
    target_radius_scale=1.2, width=128, depth=3, batch_size=512, pretrain_steps=8000,
    adapt_steps=4000, pretrain_lr=1e-3, adapt_lr=3e-4, eps_rel=0.05, sinkhorn_iters=100,
    num_levels=3, lambda_traj=1.0, eval_count=1024, eval_every=500,
)

ARMS = ["pretrained", "regress_scratch", "regress_ft", "ot_final", "ot_traj",
        "ot_traj_scratch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--dump-count", type=int, default=2048,
                        help="particles per saved cloud; larger than eval_count is fine, "
                             "these are for looking at, not for the metric")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    parsed = parser.parse_args()
    args = types.SimpleNamespace(**{k: getattr(parsed, k) for k in DEFAULTS})

    source_sampler, target_sampler = toy.make_samplers(args)
    rng = jax.random.PRNGKey(parsed.seed)
    rng, key_init = jax.random.split(rng)
    source_params = toy.init_params(key_init, args.width, args.depth)

    rows = []
    print(f"pretraining the source transport ({args.pretrain_steps} steps)", flush=True)
    source_params = toy.train(
        "source_pretrain", source_params, source_params, source_sampler, source_sampler,
        args.pretrain_steps, args.pretrain_lr, args, "regress", rng, rows,
        max(args.eval_every * 4, 1), parsed.seed,
    )

    rng, key_scratch = jax.random.split(rng)
    scratch_params = toy.init_params(key_scratch, args.width, args.depth)
    plans = {
        "pretrained": (source_params, "regress", 0),
        "regress_scratch": (scratch_params, "regress", args.adapt_steps),
        "regress_ft": (source_params, "regress", args.adapt_steps),
        "ot_final": (source_params, "ot_final", args.adapt_steps),
        "ot_traj": (source_params, "ot_traj", args.adapt_steps),
        "ot_traj_scratch": (scratch_params, "ot_traj", args.adapt_steps),
    }

    # one fixed noise draw for every arm, so the clouds are comparable point-for-point
    rng, key_z, key_src, key_tgt = jax.random.split(rng, 4)
    z = jax.random.normal(key_z, (parsed.dump_count, 2), np.float32)
    bundle = {
        "real_source": np.asarray(source_sampler(key_src, parsed.dump_count)),
        "real_target": np.asarray(target_sampler(key_tgt, parsed.dump_count)),
        "noise": np.asarray(z),
    }

    for arm in [a for a in parsed.arms.split(",") if a]:
        init, kind, steps = plans[arm]
        rng, key_arm = jax.random.split(rng)
        print(f"\n=== {arm} ({kind}, {steps} steps) ===", flush=True)
        params = toy.train(arm, init, source_params, target_sampler, target_sampler, steps,
                           args.adapt_lr, args, kind, key_arm, rows, args.eval_every,
                           parsed.seed)
        for nfe in (1, 4):
            bundle[f"{arm}_nfe{nfe}"] = np.asarray(toy.generate(params, z, nfe))

    np.savez_compressed(parsed.out, **bundle)
    print(f"\nwrote {parsed.out}: " + ", ".join(
        f"{k}{v.shape}" for k, v in sorted(bundle.items())))


if __name__ == "__main__":
    main()
