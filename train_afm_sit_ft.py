"""AFM post-training of a target-finetuned plain SiT (FT) checkpoint: Tab. 2 row "FT + AFM".

Official fixed-NFE AFM (afm_sit_ft.py) on the plain single-time SiT, K = sampling.num_steps jumps.
Training uses NO guidance: the generator is class-conditional, with class dropout to the null
class (model.class_dropout_prob) applied to the labels seen by both G and D. CFG is applied ONLY
at sampling, by the plain-SiT Euler sampler (utils.sit_sample_util, omega = sampling.omega).

Reuses train_afm.train_and_evaluate (optimisers, D:G schedule, losses, EMA, FID/best_fid, full-state
checkpoints) and patches only the model-specific hooks, like train_afm_sit_meft.py does.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from jax import random

import train_afm as _base
from afm_sit_ft import (
    drop_labels,
    fixed_grid_endpoint_terms,
    fixed_grid_times,
    sample_step_indices,
)
from models import imfDiT
from models.sit_ft_discriminator import create_sit_ft_discriminator
from train_sit import _build_plain_sit
from utils.ckpt_util import restore_checkpoint
from utils.eval_csv_util import append_eval_metrics_row
from utils.logging_util import Writer, log_for_0
from utils.sample_util import (
    get_image_metric_evaluator,
    get_sample_device_batch_size,
    get_sample_devices,
    get_sample_local_device_count,
)
from utils.sit_sample_util import _create_sampling_transport
from utils.sit_sample_util import sample_step as _sit_sample_step
from utils.sit_trainstate_util import create_eval_state
from utils.sit_trainstate_util import create_train_state as _sit_create_train_state
from utils.vae_util import LatentManager


def _validate(config):
    if not str(config.model.model_str).startswith("flaxSiT"):
        raise ValueError("SiT-FT AFM needs model.model_str=flaxSiT_*.")
    if int(config.afm.num_steps) != int(config.sampling.num_steps):
        raise ValueError("afm.num_steps must equal sampling.num_steps (train grid = sampling grid).")
    if str(config.sampling.get("method", "euler")).lower() != "euler":
        raise ValueError("AFM jumps are Euler jumps: sampling.method must be 'euler'.")
    if str(config.transport.get("objective", "sit")) != "sit":
        raise ValueError("SiT-FT AFM supports the plain 'sit' transport objective only.")
    if str(config.model.get("sit_output_prediction_space", "velocity")) != "velocity":
        raise ValueError("SiT-FT AFM supports velocity-prediction SiT only.")
    for key in ("lambda_imf", "lambda_anchor"):
        if float(config.afm.get(key, 0.0)) != 0.0:
            raise ValueError(f"afm.{key} > 0 is not supported for SiT-FT AFM.")


def _grid(config):
    """Euler grid of the plain-SiT sampler for this config (identical call to generate())."""
    transport = _create_sampling_transport(config)
    tau0, tau1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    return fixed_grid_times(tau0, tau1, int(config.afm.num_steps))


def _create_models(config):
    _validate(config)
    model = _build_plain_sit(config, eval_mode=False)
    generator_net = getattr(imfDiT, config.model.model_str)(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        eval=False,
    )
    return model, generator_net, create_sit_ft_discriminator(generator_net)


def _make_endpoint_terms(model, params, images, labels, rng, config):
    rng_index, rng_noise = random.split(rng)
    times = _grid(config)
    step_index = sample_step_indices(rng_index, images.shape[0], int(config.afm.num_steps))
    noise = random.normal(rng_noise, images.shape, dtype=images.dtype)

    def velocity_fn(x, s, y):
        # PlainSiT.__call__ = transport velocity of the FT SiT, no guidance.
        return model.apply({"params": params}, x, s, y)

    return fixed_grid_endpoint_terms(velocity_fn, images, labels, noise, step_index, times)


_BASE_GET_IMAGES_AND_LABELS = _base._get_images_and_labels


def _make_get_images_and_labels(config):
    """Base loader + class dropout. Both G and D steps get the SAME dropped labels."""
    base = _BASE_GET_IMAGES_AND_LABELS
    num_classes = int(config.model.num_classes)
    drop_prob = float(config.model.class_dropout_prob) if bool(config.model.target_use_null_class) else 0.0

    def get_images_and_labels(batch, rng_vae, distributed):
        rng_vae, rng_drop = random.split(rng_vae)
        images, labels = base(batch, rng_vae, distributed)
        return images, drop_labels(rng_drop, labels, num_classes, drop_prob)

    return get_images_and_labels


def _patch(config):
    _base._create_models = _create_models
    _base._make_endpoint_terms = _make_endpoint_terms
    _base._get_images_and_labels = _make_get_images_and_labels(config)
    _base.create_train_state = _sit_create_train_state
    _base._create_sampling_model = lambda cfg: _build_plain_sit(cfg, eval_mode=True)
    _base._sample_step = _sit_sample_step


def train_and_evaluate(config, workdir):
    _validate(config)
    if not config.model.get("uses_classifier_free_guidance", False):
        # train_afm passes sampling.omega to the sampler only for guidance-controllable models.
        raise ValueError("Set model.uses_classifier_free_guidance=True so eval samples with CFG.")
    _patch(config)
    return _base.train_and_evaluate(config, workdir)


def final_evaluate(config, workdir, checkpoint_dir, omegas=(None, 1.0)):
    """10K-sample FID/FDD of one saved generator checkpoint at sampling.num_steps Euler steps.

    omegas: None = config.sampling.omega (CFG, the "K x 2" row); 1.0 = no CFG (K NFE).
    Same evaluator, sampler and EMA choice as the in-training FID. Rows go to
    <workdir>/eval_best_fid_<K>steps_omega<w>/eval_metrics.csv.
    """
    _validate(config)
    _base._set_num_classes_from_data(config)
    num_steps = int(config.sampling.num_steps)
    image_size = int(config.dataset.image_size)
    use_ema = bool(config.training.use_ema) and not bool(config.training.get("fid_use_online_only", False))
    model = _build_plain_sit(config, eval_mode=True)
    state = create_eval_state(random.PRNGKey(0), config, model, image_size, model_label="SiT-FT AFM")
    state = state.replace(ema_params=state.params)
    state = restore_checkpoint(state, os.path.abspath(checkpoint_dir))
    if getattr(state, "ema_params", None) is None:
        state = state.replace(ema_params=state.params)
    step = int(np.asarray(state.step).reshape(-1)[0])
    sample_devices = get_sample_devices(config)
    state = jax_utils.replicate(state, devices=sample_devices)
    sample_bsz = get_sample_device_batch_size(config)
    latent_manager = LatentManager(
        config.dataset.vae, sample_bsz, image_size,
        decode_num_local_devices=get_sample_local_device_count(config),
    )
    p_sample = jax.pmap(
        partial(_sit_sample_step, model=model, rng_init=random.PRNGKey(99),
                device_batch_size=sample_bsz, config=config, num_steps=num_steps),
        "batch", devices=sample_devices,
    )
    results = []
    for omega in omegas:
        omega = float(config.sampling.omega) if omega is None else float(omega)
        out_dir = os.path.join(workdir, f"eval_best_fid_{num_steps}steps_omega{omega:g}")
        os.makedirs(out_dir, exist_ok=True)
        evaluator = get_image_metric_evaluator(config, Writer(config, out_dir), latent_manager)
        kwargs = jax_utils.replicate(
            {"omega": omega, "t_min": float(config.sampling.t_min), "t_max": float(config.sampling.t_max)},
            devices=sample_devices,
        )
        result = evaluator(state, p_sample, step, ema_only=use_ema,
                           metric_suffix=f"steps_{num_steps}_omega{omega:g}", **kwargs)
        fdd = result.get("fd_dino")
        row = {
            "eval_phase": "final_eval",
            "metric_mode": "ema" if use_ema else "online",
            "training_step": step,
            "sampling_num_steps": num_steps,
            "omega": omega,
            "t_min": float(config.sampling.t_min),
            "t_max": float(config.sampling.t_max),
            "fid": float(result["fid"]),
            "inception_score": float(result["is"]),
            "fd_dino": "" if fdd is None else float(fdd),
            "is_best_fid": "",
            "is_best_fd_dino": "",
            "checkpoint_path": os.path.abspath(checkpoint_dir),
        }
        append_eval_metrics_row(out_dir, row)
        log_for_0("FINAL EVAL omega=%g NFE=%d x %d: FID %.4f FDD %s", omega, num_steps,
                  1 if omega == 1.0 else 2, row["fid"], row["fd_dino"])
        results.append(row)
    return results
