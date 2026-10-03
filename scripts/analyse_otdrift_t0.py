"""Why does the t=0 arm look like arcs, and why does it lose on-policy?

Reads the seed-0 clouds EXP-126 dumped and decomposes the error instead of eyeballing it.
No training here, no jax: this is arithmetic on saved point sets.

Three questions, three sections:
  1. mass    -- is the map a bijection on modes, or do modes merge and leave targets empty?
  2. shape   -- is a blob translated (good) or stretched into an arc (what the figure shows)?
                reported as the std along each principal axis against the blob's own 0.25.
  3. blur    -- how fine a structure can the training loss actually see? The entropic
                regulariser is eps = 0.05 * mean cost, so its length scale is sqrt(eps).
                Anything smaller than that is free, and the optimiser will not fix it.
Plus the on-policy gain: how much output error one unit of input error buys.
"""

import numpy as np

NUM_MODES = 6
RADIUS, SPREAD = 3.0, 0.25
TGT_ROT, TGT_OFFSET, TGT_SCALE = 0.5236, 1.0, 1.2


def ring(radius, rotation, offset):
    k = np.arange(NUM_MODES)
    a = 2 * np.pi * k / NUM_MODES + rotation
    return np.stack([radius * np.cos(a) + offset, radius * np.sin(a)], 1)


SRC_C = ring(RADIUS, 0.0, 0.0)
TGT_C = ring(RADIUS * TGT_SCALE, TGT_ROT, TGT_OFFSET)


def assign(points, centres):
    return np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), 1)


def frame(centre):
    """unit radial and tangential vectors at a target centre (about the target ring's own hub)"""
    v = centre - np.array([TGT_OFFSET, 0.0])
    r = v / np.linalg.norm(v)
    return r, np.array([-r[1], r[0]])


def blur_length(fake, real):
    """sqrt(eps) with eps = eps_rel * mean squared cost -- the loss's resolution limit."""
    cost = ((fake[:, None, :] - real[None]) ** 2).sum(-1)
    return float(np.sqrt(0.05 * cost.mean()))


def section_mass(label, src_mode, out_mode):
    print(f"\n[{label}] mass flow, source mode -> target mode (rows sum to 1)")
    table = np.zeros((NUM_MODES, NUM_MODES))
    for s, t in zip(src_mode, out_mode):
        table[s, t] += 1
    table /= table.sum(1, keepdims=True)
    print("      " + "".join(f"  tgt{t}" for t in range(NUM_MODES)) + "   -> argmax")
    for s in range(NUM_MODES):
        print(f"  src{s} " + "".join(f"{v:6.2f}" for v in table[s]) + f"   {table[s].argmax()}")
    received = np.bincount(out_mode, minlength=NUM_MODES) / out_mode.size
    print("  mass received per target mode: " + " ".join(f"{v:.3f}" for v in received)
          + f"   (uniform = {1 / NUM_MODES:.3f}, max dev {np.abs(received - 1 / NUM_MODES).max():.3f})")
    return table


def section_shape(label, src_mode, points):
    print(f"\n[{label}] per-mode output shape. blob std should stay {SPREAD:.2f} in both axes.")
    print("  src  n    centre err   std along principal   std radial  std tangential  stretch")
    for s in range(NUM_MODES):
        pts = points[src_mode == s]
        mean = pts.mean(0)
        target = TGT_C[assign(mean[None], TGT_C)[0]]
        centred = pts - mean
        evals = np.linalg.eigvalsh(np.cov(centred.T))[::-1]
        rad, tan = frame(target)
        print(f"  {s}  {pts.shape[0]:4d}   {np.linalg.norm(mean - target):8.3f}      "
              f"{np.sqrt(evals[0]):.3f} / {np.sqrt(evals[1]):.3f}        "
              f"{(centred @ rad).std():.3f}        {(centred @ tan).std():.3f}       "
              f"{np.sqrt(evals[0] / max(evals[1], 1e-12)):5.1f}x")


def section_decompose(label, src_mode, points):
    """split the squared error into 'blob in the wrong place' vs 'blob the wrong shape'."""
    place, shape, n = 0.0, 0.0, points.shape[0]
    for s in range(NUM_MODES):
        pts = points[src_mode == s]
        mean = pts.mean(0)
        target = TGT_C[assign(mean[None], TGT_C)[0]]
        w = pts.shape[0] / n
        place += w * ((mean - target) ** 2).sum()
        shape += w * ((pts - mean) ** 2).sum(1).mean()
    ideal = 2 * SPREAD ** 2
    print(f"\n[{label}] error budget, per point, squared units")
    print(f"  wrong place  (mode centre vs its target centre) {place:8.4f}  -> rms {np.sqrt(place):.3f}")
    print(f"  wrong shape  (spread about the mode centre)     {shape:8.4f}  -> rms {np.sqrt(shape):.3f}")
    print(f"  of which unavoidable (the target blob itself)   {ideal:8.4f}  -> rms {np.sqrt(ideal):.3f}")
    print(f"  excess spread = the arcs                        {shape - ideal:8.4f}  "
          f"-> rms {np.sqrt(max(shape - ideal, 0)):.3f}")


def main():
    d = np.load("data/toy_source_t_clouds.npz")
    real_target = d["real_target"]

    print("=" * 88)
    print("scale reference")
    print(f"  blob std                      {SPREAD:.3f}")
    print(f"  nearest-neighbour mode gap    {np.linalg.norm(TGT_C[0] - TGT_C[1]):.3f}")
    print(f"  source -> target mode shift   "
          f"{np.linalg.norm(TGT_C[assign(SRC_C, TGT_C)] - SRC_C, axis=1).mean():.3f}")
    for arm in ["src_t1", "src_t0.5", "src_t0"]:
        blur = blur_length(d[f"{arm}_out_forward"], real_target)
        print(f"  loss blur sqrt(eps), {arm:9s} {blur:.3f}   "
              f"= {blur / SPREAD:.1f}x the blob")
    # eps is set as a FRACTION OF THE MEAN COST, and the mean cost is set by the ring size,
    # not by the blob size. So the blur tracks the global scale and ignores the structure we
    # are asking the loss to resolve. This is what eps_rel would have to be instead:
    mean_cost = blur_length(d["src_t0_out_forward"], real_target) ** 2 / 0.05
    print(f"  mean cost {mean_cost:.1f}; eps_rel for a blur of one blob std: "
          f"{SPREAD ** 2 / mean_cost:.4f}  (we used 0.05, i.e. "
          f"{0.05 / (SPREAD ** 2 / mean_cost):.0f}x too coarse)")

    for arm in ["src_t0", "scratch_t0"]:
        print("\n" + "=" * 88)
        print(f"{arm}: input is a clean source image, so every input has a true source mode")
        src_mode = assign(d[f"{arm}_in_forward"], SRC_C)
        out = d[f"{arm}_out_forward"]
        section_mass(arm, src_mode, assign(out, TGT_C))
        section_shape(arm, src_mode, out)
        section_decompose(arm, src_mode, out)

    print("\n" + "=" * 88)
    print("is the stretching specific to t=0?  per-TARGET-mode output spread, every arm.")
    print("(assigning outputs by nearest target centre needs no source label, so t=1 is")
    print(f" comparable here. the target's own blob is {SPREAD:.3f} in both axes.)")
    print("  arm         mean std radial   mean std tangential   mean stretch")
    for arm in ["src_t1", "src_t0.5", "src_t0", "scratch_t1", "scratch_t0"]:
        out = d[f"{arm}_out_forward"]
        mode = assign(out, TGT_C)
        rads, tans, stretch = [], [], []
        for t in range(NUM_MODES):
            pts = out[mode == t]
            if pts.shape[0] < 20:
                continue
            centred = pts - pts.mean(0)
            rad, tan = frame(TGT_C[t])
            rads.append((centred @ rad).std())
            tans.append((centred @ tan).std())
            evals = np.linalg.eigvalsh(np.cov(centred.T))[::-1]
            stretch.append(np.sqrt(evals[0] / max(evals[1], 1e-12)))
        print(f"  {arm:11s} {np.mean(rads):11.3f} {np.mean(tans):20.3f} "
              f"{np.mean(stretch):14.1f}x")
    for name, pts in [("real_target", real_target)]:
        mode = assign(pts, TGT_C)
        rads = [((pts[mode == t] - pts[mode == t].mean(0)) @ frame(TGT_C[t])[0]).std()
                for t in range(NUM_MODES)]
        tans = [((pts[mode == t] - pts[mode == t].mean(0)) @ frame(TGT_C[t])[1]).std()
                for t in range(NUM_MODES)]
        print(f"  {name:11s} {np.mean(rads):11.3f} {np.mean(tans):20.3f}"
              f"{'':15s}(the answer)")

    print("\n" + "=" * 88)
    print("on-policy: what the frozen source model hands us instead, and what it costs")
    for arm in ["src_t1", "src_t0.5", "src_t0"]:
        fwd, pol = d[f"{arm}_in_forward"], d[f"{arm}_in_onpolicy"]
        of, op = d[f"{arm}_out_forward"], d[f"{arm}_out_onpolicy"]
        # distribution-level, not pointwise: these are independent draws
        def spread(p):
            return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
        print(f"  {arm:9s} input rms radius {spread(fwd):.3f} -> {spread(pol):.3f}   "
              f"input mean shift {np.linalg.norm(fwd.mean(0) - pol.mean(0)):.3f}   "
              f"output rms radius {spread(of):.3f} -> {spread(op):.3f}")
    print("\n  per-mode occupancy of the on-policy input at t=0 (does the source model cover"
          "\n  every source mode evenly? uniform = 0.167):")
    for arm in ["src_t0"]:
        for name in ["in_forward", "in_onpolicy"]:
            share = np.bincount(assign(d[f"{arm}_{name}"], SRC_C),
                                minlength=NUM_MODES) / d[f"{arm}_{name}"].shape[0]
            print(f"    {name:12s} " + " ".join(f"{v:.3f}" for v in share))
        for name in ["in_forward", "in_onpolicy"]:
            pts = d[f"{arm}_{name}"]
            mode = assign(pts, SRC_C)
            dev = np.concatenate([np.linalg.norm(pts[mode == s] - SRC_C[s], axis=1)
                                  for s in range(NUM_MODES)])
            print(f"    {name:12s} distance to its own source centre: "
                  f"mean {dev.mean():.3f}  (a true clean sample averages "
                  f"{SPREAD * np.sqrt(np.pi / 2):.3f})")


if __name__ == "__main__":
    main()
