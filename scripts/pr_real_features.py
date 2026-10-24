"""Real-image Inception / DINOv2 features for precision-recall, via the eval's own code path.

Images go through center_crop_arr(256) (same as compute_fid_stats), then fid_util.compute_stats /
compute_dinov2_stats, with FID_DUMP_FEATURES_PREFIX set so the raw features are saved.
Sanity check: FID / FD-DINO of these features against the stored reference stats must be ~0.

Usage: python scripts/pr_real_features.py <image_root_with_class_dirs> <out_prefix> <fid_ref.npz> <fdd_ref.npz>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from torchvision import datasets

from utils import dino_util, fid_util
from utils.input_pipeline import center_crop_arr

image_root, out_prefix, fid_ref, fdd_ref = sys.argv[1:5]
os.environ["FID_DUMP_FEATURES_PREFIX"] = out_prefix
batch = 16

paths = [p for p, _ in datasets.ImageFolder(image_root).samples]
images = np.stack(
    [np.array(center_crop_arr(Image.open(p).convert("RGB"), 256)) for p in paths]
).astype(np.uint8)
print(f"loaded {images.shape} from {image_root}", flush=True)

inception = fid_util.build_jax_inception(batch_size=batch)
stats = fid_util.compute_stats(
    images, inception, batch_size=batch, fid_samples=len(images), num_local_devices=1
)
ref = fid_util.get_reference(fid_ref)
print("FID(real feats, ref stats) =",
      fid_util.compute_fid(ref["mu"], stats["mu"], ref["sigma"], stats["sigma"]), flush=True)

dino = dino_util.build_jax_dinov2(arch="vitb14", batch_size=batch)
dstats = fid_util.compute_dinov2_stats(
    images, dino, batch_size=batch, fid_samples=len(images), num_local_devices=1
)
dref = fid_util.get_reference(fdd_ref)
print("FD-DINO(real feats, ref stats) =",
      fid_util.compute_fid(dref["mu"], dstats["mu"], dref["sigma"], dstats["sigma"]), flush=True)
