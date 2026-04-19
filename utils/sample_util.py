import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from utils import dino_util
from utils import fid_util
from utils.logging_util import log_for_0


def has_controllable_sampling_guidance(model_config):
    if model_config.get("use_auxiliary_v_head", True):
        return True
    if model_config.get("use_context_guidance_conditioning", False):
        return True
    if model_config.get("use_adaln_guidance_scale_conditioning", False):
        return True
    return not (
        model_config.get("use_training_guidance", True)
        and model_config.get("guidance_scale_strategy", "sampled") == "fixed"
    )


def run_p_sample_step(
    p_sample_step, state, sample_idx, latent_manager, ema=True, **kwargs
):
    """
    Run one p_sample_step to get samples from the model.
    """
    params = state.ema_params if ema else state.params

    variable = {"params": params}
    latent = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
    latent = latent.reshape(-1, *latent.shape[2:])

    samples = latent_manager.decode(latent)
    assert not jnp.any(
        jnp.isnan(samples)
    ), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"

    samples = samples.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

    jax.random.normal(random.key(0), ()).block_until_ready()  # dist sync
    return samples


def get_sample_device_batch_size(config):
    sample_device_batch_size = int(config.fid.get("sample_device_batch_size", -1))
    if sample_device_batch_size <= 0:
        sample_device_batch_size = config.fid.device_batch_size
    return int(sample_device_batch_size)


def _get_half_precision_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if dtype_name in ("fp16", "float16"):
        return jnp.float16
    raise ValueError(
        "half precision dtype must be one of: bfloat16, bf16, float16, fp16. "
        f"Got {dtype_name!r}."
    )


def get_training_param_dtype(config):
    if not config.training.get("half_precision", False):
        return None

    return _get_half_precision_dtype(config.training.get("half_precision_dtype", "float16"))


def get_sampling_param_dtype(config):
    sampling = config.get("sampling", {})
    if not sampling.get("half_precision", False):
        return None

    return _get_half_precision_dtype(
        sampling.get(
            "half_precision_dtype",
            config.training.get("half_precision_dtype", "float16"),
        )
    )


def _cast_floating_tree(tree, dtype):
    if tree is None or dtype is None:
        return tree

    def maybe_cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(maybe_cast, tree)


def maybe_cast_state_for_sampling(state, config):
    dtype = get_sampling_param_dtype(config)
    if dtype is None:
        return state

    log_for_0("Casting sampling params to %s.", dtype)
    return state.replace(
        params=_cast_floating_tree(state.params, dtype),
        ema_params=_cast_floating_tree(state.ema_params, dtype),
    )


def generate_fid_samples(
    state, config, p_sample_step, run_p_sample_step, ema=True, num_samples=None, **kwargs
):
    """
    Generate samples for FID evaluation or preview logging.
    """
    target_num_samples = config.fid.num_samples if num_samples is None else num_samples
    sample_device_batch_size = get_sample_device_batch_size(config)
    state = maybe_cast_state_for_sampling(state, config)
    num_steps = np.ceil(
        target_num_samples / sample_device_batch_size / jax.device_count()
    ).astype(int)
    log_sample_every = max(1, int(config.fid.get("sample_log_every", 1)))

    samples_all = []

    log_for_0("Note: the first sample may be significant slower")
    log_for_0(
        "Generating %d samples with sample_device_batch_size=%d, global_batch_size=%d, batches=%d",
        target_num_samples,
        sample_device_batch_size,
        sample_device_batch_size * jax.device_count(),
        num_steps,
    )
    for step in range(num_steps):
        sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
            jax.local_device_count()
        )
        sample_idx = jax.device_count() * step + sample_idx
        should_log = step == 0 or step + 1 == num_steps or step % log_sample_every == 0
        if should_log:
            log_for_0(f"Sampling step {step} / {num_steps}...")
        sample_start = time.time()
        samples = run_p_sample_step(
            p_sample_step, state, sample_idx=sample_idx, ema=ema, **kwargs
        )
        samples.block_until_ready()
        device_seconds = time.time() - sample_start
        transfer_start = time.time()
        samples = jax.device_get(samples)
        transfer_seconds = time.time() - transfer_start
        if should_log:
            log_for_0(
                "Sampling batch %d / %d timing: device %.2fs, device_get %.2fs",
                step,
                num_steps,
                device_seconds,
                transfer_seconds,
            )
        samples_all.append(samples)

    samples_all = np.concatenate(samples_all, axis=0)

    return samples_all[:target_num_samples]


def _get_eval_descriptor(kwargs, mode_str, guidance_controllable=True):
    omega = kwargs.get("omega", None)[0]
    t_min = kwargs.get("t_min", None)[0]
    t_max = kwargs.get("t_max", None)[0]
    if guidance_controllable:
        descriptor = f"omega_{omega:.2f}_tmin_{t_min:.2f}_tmax_{t_max:.2f}_{mode_str}"
    else:
        descriptor = f"single_head_{mode_str}"
    return descriptor, omega, t_min, t_max


def get_image_metric_evaluator(config, writer, latent_manager):
    """
    Create a single evaluator that logs FID, Inception Score, and optionally FD-DINO.
    """
    inception_batch_size = config.fid.device_batch_size * jax.local_device_count()
    inception_net = fid_util.build_jax_inception(batch_size=inception_batch_size)
    fid_stats_ref = fid_util.get_reference(config.fid.cache_ref)

    fd_dino_config = config.get("fd_dino", None)
    fd_dino_enabled = bool(fd_dino_config and fd_dino_config.get("cache_ref", ""))
    dino_net = None
    fd_dino_stats_ref = None
    if fd_dino_enabled:
        dino_net = dino_util.build_jax_dinov2(
            arch=fd_dino_config.get("arch", "vitb14"),
            model_name=fd_dino_config.get("model_name", None),
            batch_size=config.fid.device_batch_size * jax.device_count(),
        )
        fd_dino_stats_ref = fid_util.get_reference(fd_dino_config.cache_ref)

    run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)
    use_ema = config.training.get("use_ema", True)
    guidance_controllable = has_controllable_sampling_guidance(config.model)

    def _evaluate_one_mode(state, p_sample_step, ema, **kwargs):
        samples_all = generate_fid_samples(
            state, config, p_sample_step, run_p_sample_step_inner, ema, **kwargs
        )

        fid_stats = fid_util.compute_stats(
            samples_all,
            inception_net,
            batch_size=config.fid.device_batch_size,
            fid_samples=config.fid.num_samples,
        )
        metric = {}
        result = {}

        mode_str = "ema" if ema else "online"
        descriptor, omega, t_min, t_max = _get_eval_descriptor(
            kwargs, mode_str, guidance_controllable=guidance_controllable
        )
        if guidance_controllable:
            log_for_0(
                f"Computing image metrics at omega={omega:.2f}, t_min={t_min:.2f}, "
                f"t_max={t_max:.2f}, mode={mode_str}..."
            )
        else:
            log_for_0(
                f"Computing image metrics for single-head evaluation without "
                f"sampling-time guidance control, mode={mode_str}..."
            )

        fid = fid_util.compute_fid(
            fid_stats_ref["mu"],
            fid_stats["mu"],
            fid_stats_ref["sigma"],
            fid_stats["sigma"],
        )
        is_score, _ = fid_util.compute_inception_score(fid_stats["logits"])

        metric[f"FID_{descriptor}"] = fid
        metric[f"IS_{descriptor}"] = is_score
        result["fid"] = fid
        result["is"] = is_score

        if fd_dino_enabled:
            dino_stats = fid_util.compute_dinov2_stats(
                samples_all,
                dino_net,
                batch_size=config.fid.device_batch_size,
                fid_samples=config.fid.num_samples,
            )
            fd_dino = fid_util.compute_fid(
                fd_dino_stats_ref["mu"],
                dino_stats["mu"],
                fd_dino_stats_ref["sigma"],
                dino_stats["sigma"],
            )
            metric[f"FD_DINO_{descriptor}"] = fd_dino
            result["fd_dino"] = fd_dino

        return metric, result

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        metric_dict = {}
        primary_result = None
        if use_ema:
            metric, primary_result = _evaluate_one_mode(
                state, p_sample_step, True, **kwargs
            )
            metric_dict.update(metric)
            if not ema_only:
                metric, _ = _evaluate_one_mode(
                    state, p_sample_step, False, **kwargs
                )
                metric_dict.update(metric)
        else:
            metric, primary_result = _evaluate_one_mode(
                state, p_sample_step, False, **kwargs
            )
            metric_dict.update(metric)

        writer.write_scalars(step + 1, metric_dict)
        return primary_result

    return evaluator


def get_fid_evaluator(config, writer, latent_manager):
    """
    Backward-compatible wrapper that returns FID and IS.
    """
    metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        result = metric_evaluator(
            state, p_sample_step, step, ema_only=ema_only, **kwargs
        )
        return result["fid"], result["is"]

    return evaluator


def get_fd_dino_evaluator(config, writer, latent_manager):
    """
    Backward-compatible wrapper that returns only FD-DINO.
    """
    fd_dino_config = config.get("fd_dino", None)
    if fd_dino_config is None or not fd_dino_config.get("cache_ref", ""):
        raise ValueError("config.fd_dino.cache_ref must be set to use FD-DINO evaluation.")

    metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        result = metric_evaluator(
            state, p_sample_step, step, ema_only=ema_only, **kwargs
        )
        return result["fd_dino"]

    return evaluator
