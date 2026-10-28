"""CPU smoke for SiT-FT AFM: tiny FlaxSiT, ~20 batches with 2 in-training FID evals, then final eval
(CFG omega 1.5 and no-CFG omega 1). Also loads the REAL XL FT checkpoint into the XL train state.
Run: CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/smoke_sit_ft_afm.py <workdir> [<xl_ft_ckpt>]
"""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from jax import random

from configs.load_config import get_config
from models import imfDiT

imfDiT.flaxSiT_T_test = partial(imfDiT.FlaxSiT, depth=2, hidden_size=64, patch_size=2, num_heads=2, learn_sigma=True)
import train_afm_sit_ft as T  # noqa: E402
from train_sit import _build_plain_sit  # noqa: E402

ADV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workdir = os.path.abspath(sys.argv[1]); os.makedirs(workdir, exist_ok=True)


def tiny_config():
    c = get_config("sit_ft_afm_tab2")
    c.model.model_str = "flaxSiT_T_test"
    c.dataset.root = f"{ADV}/../datasets/cub-200-2011_processed_latents"
    c.dataset.name = "cub200_latent"
    c.dataset.num_classes_from_data = False
    c.dataset.num_classes = c.model.num_classes = c.sampling.num_classes = 200
    c.dataset.num_workers = 0
    c.dataset.prefetch_factor = None
    c.dataset.pin_memory = False
    c.fid.cache_ref = f"{ADV}/files/fid_stats/cub-200-2011_processed-fid_stats.npz"
    c.fd_dino.cache_ref = f"{ADV}/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz"
    c.fid.num_samples = 16
    c.fid.device_batch_size = 4
    c.fid.sample_device_batch_size = 4
    c.afm.max_posttrain_batches = 20
    c.training.fid_schedule = [{"from_step": 10, "every_steps": 10}]
    c.training.sample_per_step = 0
    c.training.log_per_step = 5
    c.training.checkpoint_per_epoch = 1000000
    return c


# 1) tiny "FT" checkpoint in the same Flax layout as the FT runs ({"params": {"net": ...}})
c = tiny_config()
model = _build_plain_sit(c, eval_mode=False)
x = jnp.ones((1, 32, 32, 4)); t = jnp.ones((1,)); y = jnp.zeros((1,), jnp.int32)
params = model.init({"params": random.PRNGKey(0)}, x, t, y)["params"]
ft_dir = os.path.join(workdir, "tiny_ft")
checkpoints.save_checkpoint(ft_dir, {"step": 0, "params": params, "ema_params": None}, 0, overwrite=True)
ft_ckpt = os.path.join(ft_dir, "checkpoint_0")
print("SMOKE tiny FT ckpt:", ft_ckpt, flush=True)

# 2) train ~20 batches (1:1 D:G) with FID at batch 10 and 20 (CFG omega 1.5, 4 Euler steps)
c.load_from = ft_ckpt
train_wd = os.path.join(workdir, "train")
state = T.train_and_evaluate(c, train_wd)
print("SMOKE trained: batch", int(state.step), "G", int(state.gen_step), "D", int(state.dis_step), flush=True)

# 3) final eval of the best_fid checkpoint: omega 1.5 (4x2) and omega 1 (4 NFE)
best = sorted(d for d in os.listdir(os.path.join(train_wd, "best_fid")) if d.startswith("checkpoint_"))[-1]
rows = T.final_evaluate(tiny_config(), train_wd, os.path.join(train_wd, "best_fid", best))
for r in rows:
    print("SMOKE final", {k: r[k] for k in ("omega", "sampling_num_steps", "fid", "fd_dino", "training_step")}, flush=True)

# 4) optional: real XL FT checkpoint -> XL plain-SiT train state (partial restore must skip 0 tensors)
if len(sys.argv) > 2:
    from utils.ckpt_util import restore_partial_checkpoint
    from utils.sit_trainstate_util import create_train_state
    cx = get_config("sit_ft_afm_tab2")
    cx.model.num_classes = cx.dataset.num_classes = 200
    mx = _build_plain_sit(cx, eval_mode=False)
    sx = create_train_state(random.PRNGKey(0), cx, mx, 32, lambda _: 1e-5)
    before = jax.tree_util.tree_leaves(sx.params)[3]
    sx = restore_partial_checkpoint(sx, sys.argv[2], prefer_ema=False, target_model_config=cx.model)
    after = jax.tree_util.tree_leaves(sx.params)[3]
    n = sum(v.size for v in jax.tree_util.tree_leaves(sx.params))
    print("SMOKE XL params:", n, "tensor changed by load:", bool(jnp.any(before != after)), flush=True)
print("SMOKE DONE", flush=True)
