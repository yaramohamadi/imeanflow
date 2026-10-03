"""Does a smaller entropic blur remove the squash and the bend? (EXP-127)

Section 9 diagnosed the t=0 crescents as two local defects of the map -- a near-singular
Jacobian (squash) and a quadratic residual (bend) -- and blamed the entropic blur, which was
1.13 against a 0.25 blob. EXP-127 swept `eps_rel`. This re-runs section 9's geometry fit at
every swept value, so the diagnosis is tested rather than argued.

Arithmetic on the saved seed-0 clouds. No training, no jax. Writes to stdout; the recorded
output is `data/eps_sweep/geometry.log`.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "data", "eps_sweep")
LEVELS = ["0.05", "0.02", "0.005", "0.0025", "0.001"]

NUM_MODES = 6
RADIUS, SPREAD = 3.0, 0.25
TGT_ROT, TGT_OFFSET, TGT_SCALE = 0.5236, 1.0, 1.2

ANGLES = 2.0 * np.pi * np.arange(NUM_MODES) / NUM_MODES
SRC_C = np.stack([RADIUS * np.cos(ANGLES), RADIUS * np.sin(ANGLES)], 1)
_T = ANGLES + TGT_ROT
TGT_C = np.stack([RADIUS * TGT_SCALE * np.cos(_T) + TGT_OFFSET,
                  RADIUS * TGT_SCALE * np.sin(_T)], 1)


def assign(points, centres):
    return np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), 1)


def frame(centre):
    v = centre - np.array([TGT_OFFSET, 0.0])
    r = v / np.linalg.norm(v)
    return r, np.array([-r[1], r[0]])


def geometry(inputs, outputs, real, eps_rel):
    """blur length, mean Jacobian condition number, mean bend, mean output blob std."""
    cost = ((outputs[:, None, :] - real[None]) ** 2).sum(-1)
    blur = float(np.sqrt(eps_rel * cost.mean()))

    mode = assign(inputs, SRC_C)
    hi, lo, bend = [], [], []
    for s in range(NUM_MODES):
        xc = inputs[mode == s] - inputs[mode == s].mean(0)
        yc = outputs[mode == s] - outputs[mode == s].mean(0)
        fitted, *_ = np.linalg.lstsq(xc, yc, rcond=None)
        sv = np.linalg.svd(fitted.T)[1]
        hi.append(sv[0])
        lo.append(sv[1])
        bend.append((yc - xc @ fitted).std(0).mean())

    out_mode, rad, tan = assign(outputs, TGT_C), [], []
    for t in range(NUM_MODES):
        pts = outputs[out_mode == t]
        if pts.shape[0] < 20:
            continue
        centred = pts - pts.mean(0)
        r, tg = frame(TGT_C[t])
        rad.append((centred @ r).std())
        tan.append((centred @ tg).std())

    return (blur, float(np.mean(hi)), float(np.mean(lo)),
            float(np.mean(np.array(hi) / np.array(lo))), float(np.mean(bend)),
            float(np.mean(rad)), float(np.mean(tan)))


def main():
    print("EXP-127: does a smaller entropic blur remove the squash and the bend?")
    print("src_t0, seed 0, train-matched inputs. Jacobian fitted per source mode, then averaged.")
    print(f"\n{'eps_rel':>8} {'blur':>6} {'J hi':>6} {'J lo':>6} {'cond':>6} {'bend':>6} "
          f"{'rad':>6} {'tan':>6}")
    print(f"{'ideal':>8} {'-':>6} {TGT_SCALE:6.2f} {TGT_SCALE:6.2f} {1.0:6.1f} {0.0:6.3f} "
          f"{SPREAD:6.3f} {SPREAD:6.3f}")
    for level in LEVELS:
        d = np.load(os.path.join(SWEEP, f"eps_{level}_clouds.npz"))
        blur, hi, lo, cond, bend, rad, tan = geometry(
            d["src_t0_in_forward"], d["src_t0_out_forward"], d["real_target"], float(level))
        print(f"{level:>8} {blur:6.2f} {hi:6.2f} {lo:6.2f} {cond:6.1f} {bend:6.3f} "
              f"{rad:6.3f} {tan:6.3f}")
    print(f"\nblur = sqrt(eps_rel * mean cost), in data units; the blob's own std is {SPREAD:.2f}.")
    print("eps 0.001 is the collapsed arm (W2 4.42), so its geometry is not a tightening.")


if __name__ == "__main__":
    main()
