"""Print the Frechet distance of saved eval features against a reference stats file.

Used to tell which weight set (EMA / online) a dumped feature file came from, by matching
the logged FID / FD-DINO values. Usage: python scripts/pr_identify.py <ref.npz> <feats.npy> [...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils import fid_util

ref = fid_util.get_reference(sys.argv[1])
for path in sys.argv[2:]:
    f = np.load(path).astype(np.float64)
    fd = fid_util.compute_fid(ref["mu"], f.mean(0), ref["sigma"], np.cov(f, rowvar=False))
    print(f"{fd:.4f}  {path}")
