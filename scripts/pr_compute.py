"""k-NN precision / recall (Kynkaenniemi et al. 2019) and density / coverage (Naeem et al. 2020).

Real features are subsampled (seed 0) to the number of generated samples so both sets match.
Usage: python scripts/pr_compute.py <real.npy> <fake.npy> [k=3]   -> prints one JSON line
"""
import json
import sys

import numpy as np


def pairwise_sq(a, b, chunk=2048):
    out = np.empty((len(a), len(b)), dtype=np.float32)
    bb = (b.astype(np.float64) ** 2).sum(1)
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk].astype(np.float64)
        d = (x ** 2).sum(1)[:, None] + bb[None, :] - 2.0 * x @ b.T.astype(np.float64)
        out[i:i + chunk] = np.maximum(d, 0.0)
    return out


def kth_radius(d_self, k):
    # d_self includes the zero self-distance, so the k-th neighbour is index k after sorting.
    return np.sqrt(np.partition(d_self, k, axis=1)[:, k])


def prdc(real, fake, k=3):
    r_rad = kth_radius(pairwise_sq(real, real), k)
    f_rad = kth_radius(pairwise_sq(fake, fake), k)
    d_rf = np.sqrt(pairwise_sq(real, fake))  # [R, F]
    precision = float((d_rf <= r_rad[:, None]).any(axis=0).mean())
    recall = float((d_rf <= f_rad[None, :]).any(axis=1).mean())
    density = float((d_rf <= r_rad[:, None]).sum(axis=0).mean() / k)
    coverage = float((d_rf.min(axis=1) <= r_rad).mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


if __name__ == "__main__":
    real, fake = np.load(sys.argv[1]), np.load(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rng = np.random.default_rng(0)
    if len(real) > len(fake):
        real = real[rng.choice(len(real), len(fake), replace=False)]
    res = prdc(real, fake, k)
    res.update(k=k, n_real=len(real), n_fake=len(fake), real=sys.argv[1], fake=sys.argv[2])
    print(json.dumps(res))
