"""Class-conditional samples from an iMF / MF-T+CAMF checkpoint, for qualitative figures.

Same path as train.just_evaluate (restore_eval_checkpoint, iMeanFlow.sample_one_step,
LatentManager.decode); the only change is that labels are chosen, not arange/random.
Each class gets its own fixed noise: fold_in(PRNGKey(seed), class_id).

Usage (run from the repo root):
  python scripts/sample_imf_classes.py --config=configs/load_config.py:caltech_imf_caimf_posttrain \
    --config.load_from=<checkpoint_dir> --config.dataset.num_classes=200 ... \
    --classes=16,13,72 --n_per_class=4 --num_steps=4 --out=<file.npz>
"""
import dataclasses
import os
import sys

# REPO selects whose imf.py / ckpt utils to use (JiT evals run from the main imeanflow clone).
sys.path.insert(0, os.environ.get("REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from imf import iMeanFlow
from utils.ckpt_util import restore_eval_checkpoint
from utils.vae_util import LatentManager

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Config file.", lock_config=True)
flags.DEFINE_string("classes", None, "Comma-separated class ids.")
flags.DEFINE_integer("n_per_class", 4, "Samples per class.")
flags.DEFINE_integer("num_steps", 4, "Sampling steps (NFE).")
flags.DEFINE_integer("seed", 0, "Noise seed.")
flags.DEFINE_string("out", None, "Output .npz path.")
flags.DEFINE_bool("pixel", False, "Pixel-space model (JiT): no VAE decode.")


def main(argv):
    config = FLAGS.config
    classes = [int(c) for c in FLAGS.classes.split(",")]
    n = FLAGS.n_per_class

    model_config = config.model.to_dict()
    valid = {f.name for f in dataclasses.fields(iMeanFlow)}
    model = iMeanFlow(**{k: v for k, v in model_config.items() if k in valid}, eval=True)
    # Same EMA/online choice as train.just_evaluate.
    use_ema = bool(config.training.get("use_ema", True))
    if config.training.get("fid_use_online_only", False):
        use_ema = False
    state = restore_eval_checkpoint(config.load_from, use_ema=use_ema)
    variable = {"params": state.ema_params if use_ema else state.params}
    print("restored step", int(state.step), "from", config.load_from, "ema" if use_ema else "online")

    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels
    # Same time grid as imf.generate.
    eps = 0.0
    if (
        model._uses_dmf_single_head_backbone()
        and model.target_output_prediction_space == "noise"
        and model.target_velocity_map_mode == "transport"
    ):
        default_eps = max(float(model.target_wrapper_eps), 1e-3)
        eps = float(config.sampling.get("target_transport_velocity_eps",
                                        config.sampling.get("transport_velocity_eps", default_eps)))
    if model._uses_auxiliary_v_head() or bool(config.sampling.get("meanflow_reverse_time", False)):
        t_steps = jnp.linspace(1.0, eps, FLAGS.num_steps + 1)
    else:
        t_steps = jnp.linspace(eps, 1.0, FLAGS.num_steps + 1)
    omega, t_min, t_max = config.sampling.omega, config.sampling.t_min, config.sampling.t_max
    print("NFE", FLAGS.num_steps, "omega", omega, "interval", (t_min, t_max))

    @jax.jit
    def sample(variable, z, y):
        def step_fn(i, x):
            return model.apply(variable, x, y, i, t_steps, omega, t_min, t_max,
                               method=model.sample_one_step)
        return jax.lax.fori_loop(0, FLAGS.num_steps, step_fn, z)

    lm = None if FLAGS.pixel else LatentManager(config.dataset.vae, n, img_size, decode_num_local_devices=1)
    images = []
    for c in classes:
        z = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(FLAGS.seed), c),
                              (n, img_size, img_size, img_channels), dtype=model.dtype)
        y = jnp.full((n,), c, dtype=jnp.int32)
        out = sample(variable, z, y)
        if FLAGS.pixel:
            x = jnp.clip(out, -1.0, 1.0)  # PixelImageManager.decode
        else:
            x = lm.decode(out.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        x = np.asarray(jnp.clip(127.5 * x + 128.0, 0, 255).astype(jnp.uint8))
        images.append(x)
        print("class", c, "done", x.shape)
    np.savez(FLAGS.out, images=np.stack(images), classes=np.array(classes),
             num_steps=FLAGS.num_steps, seed=FLAGS.seed, ckpt=config.load_from)
    print("wrote", FLAGS.out)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "classes", "out"])
    app.run(main)
