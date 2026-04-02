"""
Training and evaluation for improved MeanFlow.
"""

import jax
import jax.numpy as jnp
import ml_collections
import os
import torch
from flax import jax_utils
from jax import lax, random
from functools import partial
from optax._src.alias import *

from imf import iMeanFlow, generate

import utils.input_pipeline as input_pipeline
from utils.ckpt_util import (
    load_checkpoint_params,
    save_best_checkpoint,
    save_checkpoint,
    restore_checkpoint,
    restore_eval_checkpoint,
    restore_partial_checkpoint,
)
from utils.ema_util import ema_schedules, update_ema
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.vae_util import LatentManager
from utils.lr_utils import lr_schedules
from utils.sample_util import (
    generate_fid_samples,
    get_image_metric_evaluator,
    run_p_sample_step,
)
from utils.trainstate_util import create_train_state, TrainState

#######################################################
#                    Train Step                       #
#######################################################


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def train_step_with_vae(
    state, batch, rng_init, ema_fn, lr_fn, latent_manager, use_ema, grad_accum_steps
):
    """
    Perform a single training step.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]  # [B, H, W, C]
    labels = batch["label"]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)  # [B, H, W, C] sample latent

    def loss_fn(params):
        """loss function used for training."""
        outputs = state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            source_params=state.source_params,
            rngs=dict(
                gen=rng_base,
            ),
        )
        return outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    lr_value = lr_fn(state.step)
    dict_losses = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr_value
    new_grad_accum = jax.tree_util.tree_map(
        lambda acc, g: acc + g, state.grad_accum, grads
    )
    new_accum_step = state.grad_accum_step + 1
    should_apply = new_accum_step >= grad_accum_steps

    def apply_update(args):
        current_state, accum_grads = args
        mean_grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum_steps, accum_grads
        )
        updated_state = current_state.apply_gradients(grads=mean_grads)
        if use_ema:
            ema_value = ema_fn(current_state.step)
            new_ema = update_ema(
                updated_state.ema_params, updated_state.params, ema_value
            )
            updated_state = updated_state.replace(ema_params=new_ema)
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        updated_state = updated_state.replace(
            grad_accum=zero_accum,
            grad_accum_step=jnp.array(0, dtype=jnp.int32),
        )
        return updated_state

    def keep_accumulating(args):
        current_state, accum_grads = args
        return current_state.replace(
            grad_accum=accum_grads,
            grad_accum_step=new_accum_step,
        )

    new_state = jax.lax.cond(
        should_apply,
        apply_update,
        keep_accumulating,
        (state, new_grad_accum),
    )
    metrics["did_update"] = should_apply.astype(jnp.float32)
    return new_state, metrics


def _cosine_similarity(a, b, eps=1e-8):
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    num = jnp.sum(a * b, axis=1)
    den = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1) + eps
    return jnp.mean(num / den)


def debug_step_with_vae(state, batch, rng_init, latent_manager, model):
    """
    Run a single debug forward pass and expose intermediate tensors.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]
    labels = batch["label"]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)

    outputs = model.apply(
        {"params": state.params},
        images=images,
        labels=labels,
        source_params=state.source_params,
        rngs=dict(gen=rng_base),
        method=model.debug_forward,
    )

    v_u = outputs["v_u"]
    v_c = outputs["v_c"]
    v_g = outputs["v_g"]
    v_pred = outputs["v_pred"]
    V = outputs["V"]

    metrics = {
        "debug/omega_mean": jnp.mean(outputs["omega"]),
        "debug/t_mean": jnp.mean(outputs["t"]),
        "debug/r_mean": jnp.mean(outputs["r"]),
        "debug/fm_fraction": jnp.mean(outputs["fm_mask"]),
        "debug/v_u_mean": jnp.mean(v_u),
        "debug/v_u_abs_mean": jnp.mean(jnp.abs(v_u)),
        "debug/v_c_mean": jnp.mean(v_c),
        "debug/v_c_abs_mean": jnp.mean(jnp.abs(v_c)),
        "debug/v_pred_mean": jnp.mean(v_pred),
        "debug/v_pred_abs_mean": jnp.mean(jnp.abs(v_pred)),
        "debug/V_mean": jnp.mean(V),
        "debug/V_abs_mean": jnp.mean(jnp.abs(V)),
        "debug/v_g_mean": jnp.mean(v_g),
        "debug/v_g_abs_mean": jnp.mean(jnp.abs(v_g)),
        "debug/v_u_to_v_c_cosine": _cosine_similarity(v_u, v_c),
        "debug/v_pred_to_v_g_cosine": _cosine_similarity(v_pred, v_g),
        "debug/V_to_v_g_cosine": _cosine_similarity(V, v_g),
        "debug/v_pred_to_v_g_mse": jnp.mean((v_pred - v_g) ** 2),
        "debug/V_to_v_g_mse": jnp.mean((V - v_g) ** 2),
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return outputs, metrics


def _latents_to_uint8_images(latent_manager, latents_bhwc):
    num_images = latents_bhwc.shape[0]
    decode_total = latent_manager.batch_size * jax.local_device_count()
    if num_images > decode_total:
        raise ValueError(
            f"Debug decode received {num_images} images, but the compiled VAE decoder "
            f"supports at most {decode_total} images per call."
        )

    if num_images < decode_total:
        pad_shape = (decode_total - num_images,) + latents_bhwc.shape[1:]
        latents_bhwc = jnp.concatenate(
            [latents_bhwc, jnp.zeros(pad_shape, dtype=latents_bhwc.dtype)],
            axis=0,
        )

    latents_bchw = latents_bhwc.transpose(0, 3, 1, 2)
    samples = latent_manager.decode(latents_bchw)
    samples = samples[:num_images]
    samples = samples.transpose(0, 2, 3, 1)
    samples = 127.5 * samples + 128.0
    return jnp.clip(samples, 0, 255).astype(jnp.uint8)


def _velocity_step_to_uint8_images(latent_manager, z_t_bhwc, velocity_bhwc, step_scale):
    stepped_latents = z_t_bhwc - step_scale * velocity_bhwc
    return _latents_to_uint8_images(latent_manager, stepped_latents)


#######################################################
#               Sampling and Metrics                  #
#######################################################


def sample_step(variable, sample_idx, model, rng_init, device_batch_size, 
                config, num_steps, omega, t_min, t_max):
    """
    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold

    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max, sample_idx=sample_idx)

    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images


def infer_num_classes_from_latents(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        raise ValueError(f"Latent train directory not found: {train_root}")

    max_label = -1
    num_files = 0
    for filename in os.listdir(train_root):
        if not filename.endswith(".pt"):
            continue
        sample = torch.load(
            os.path.join(train_root, filename),
            map_location="cpu",
            weights_only=False,
        )
        max_label = max(max_label, int(sample["label"]))
        num_files += 1

    if num_files == 0:
        raise ValueError(f"No latent .pt files found under: {train_root}")

    return max_label + 1


def _get_eval_sampling_configs(config):
    """
    Normalize evaluation CFG settings from either eval-style list configs or
    train-style scalar sampling fields.
    """
    sampling = config.sampling

    intervals = sampling.get("interval", None)
    omegas = sampling.get("omegas", None)
    if intervals is not None and omegas is not None:
        configs = []
        for interval in intervals:
            t_min, t_max = interval
            for omega in omegas:
                configs.append((omega, t_min, t_max))
        return configs

    omega = sampling.get("omega", None)
    t_min = sampling.get("t_min", None)
    t_max = sampling.get("t_max", None)
    if omega is not None and t_min is not None and t_max is not None:
        return [(omega, t_min, t_max)]

    raise ValueError(
        "Evaluation requires either sampling.interval + sampling.omegas "
        "or sampling.omega + sampling.t_min + sampling.t_max in the config."
    )


#######################################################
#                       Main                          #
#######################################################


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ########### Initialize ###########
    writer = Writer(config, workdir)

    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0("Inferred dataset.num_classes from latent data: {}".format(inferred_num_classes))

    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size
    use_ema = config.training.get("use_ema", True)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = config.training.get("grad_accum_steps", 1)

    log_for_0("config.training.batch_size: {}".format(config.training.batch_size))
    log_for_0("config.training.use_ema: {}".format(use_ema))
    log_for_0("config.training.max_train_steps: {}".format(max_train_steps))
    log_for_0("config.training.grad_accum_steps: {}".format(grad_accum_steps))
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    ########### Create DataLoaders ###########
    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: {}".format(steps_per_epoch))

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model = iMeanFlow(**model_config)

    ########### Create Train State ###########
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)

    if config.load_from != "":
        if config.get("partial_load", False):
            state = restore_partial_checkpoint(state, config.load_from)
        else:
            state = restore_checkpoint(state, config.load_from)
        if config.training.get("capture_source_from_load", False):
            source_ckpt_path = config.load_from
            source_params = load_checkpoint_params(source_ckpt_path, prefer_ema=True)
            state = state.replace(source_params=source_params)
            log_for_0(
                "Loaded frozen source_params for DogFit from %s.",
                source_ckpt_path,
            )

    step = int(state.step)
    epoch_offset = step // steps_per_epoch

    state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(config.dataset.vae, device_bsz, image_size)

    ########### Create train and sample pmap ###########

    p_train_step = jax.pmap(
        partial(
            train_step_with_vae,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            latent_manager=latent_manager,
            use_ema=use_ema,
            grad_accum_steps=grad_accum_steps,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    p_debug_step = jax.pmap(
        partial(
            debug_step_with_vae,
            rng_init=rng,
            latent_manager=latent_manager,
            model=model,
        ),
        axis_name="batch",
    )

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
    )

    sample_kwargs = {
        "omega": config.sampling.omega,
        "t_min": config.sampling.t_min,
        "t_max": config.sampling.t_max,
    }
    sample_kwargs = jax_utils.replicate(sample_kwargs)

    image_metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)
    best_fid = float("inf")
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )
    save_best_fid_only = config.training.get("save_best_fid_only", False)

    ########### Training Loop ###########
    metrics_tracker = MetricsTracker()
    log_for_0("Initial compilation, this might take some minutes...")

    should_stop = False
    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        ########### Train ###########
        timer = Timer()
        log_for_0("epoch {}...".format(epoch))
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            micro_step = epoch * steps_per_epoch + n_batch

            batch = input_pipeline.prepare_batch_data(batch)
            state, metrics = p_train_step(state, batch)
            current_step = int(jax.device_get(state.step)[0])
            did_update = bool(jax.device_get(metrics["did_update"])[0])

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                compilation_time = timer.elapse_with_reset()
                log_for_0("p_train_step compiled in {:.2f}s".format(compilation_time))

            ########### Metrics ###########
            metrics_tracker.update(metrics)  # stream one step in
            should_log = did_update and current_step > 0 and (
                current_step == 1 or current_step % config.training.log_per_step == 0
            )
            if should_log:
                summary = metrics_tracker.finalize()
                logged_steps = 1 if current_step == 1 else config.training.log_per_step
                summary["steps_per_second"] = (
                    logged_steps / timer.elapse_with_reset()
                )
                summary.pop("did_update", None)
                writer.write_scalars(current_step, summary)

                if config.training.get("debug_log_during_train", False):
                    log_for_0("Running debug logging on current batch at step %d.", current_step)
                    debug_outputs, debug_metrics = p_debug_step(state, batch)
                    debug_metrics = {
                        k: float(jnp.asarray(v).mean())
                        for k, v in jax.device_get(debug_metrics).items()
                    }
                    writer.write_scalars(current_step, debug_metrics)

                    debug_num_images = int(config.training.get("debug_num_images", 4))
                    debug_velocity_decode_scale = float(
                        config.training.get("debug_velocity_decode_scale", 0.1)
                    )
                    debug_grid_size = int(debug_num_images ** 0.5)
                    if debug_grid_size ** 2 != debug_num_images:
                        raise ValueError(
                            f"config.training.debug_num_images must be a perfect square, got {debug_num_images}"
                        )
                    x = jax.device_get(debug_outputs["x"]).reshape(-1, *debug_outputs["x"].shape[2:])[:debug_num_images]
                    z_t = jax.device_get(debug_outputs["z_t"]).reshape(-1, *debug_outputs["z_t"].shape[2:])[:debug_num_images]
                    v_u = jax.device_get(debug_outputs["v_u"]).reshape(-1, *debug_outputs["v_u"].shape[2:])[:debug_num_images]
                    v_c = jax.device_get(debug_outputs["v_c"]).reshape(-1, *debug_outputs["v_c"].shape[2:])[:debug_num_images]
                    v_pred = jax.device_get(debug_outputs["v_pred"]).reshape(-1, *debug_outputs["v_pred"].shape[2:])[:debug_num_images]
                    V = jax.device_get(debug_outputs["V"]).reshape(-1, *debug_outputs["V"].shape[2:])[:debug_num_images]
                    v_g = jax.device_get(debug_outputs["v_g"]).reshape(-1, *debug_outputs["v_g"].shape[2:])[:debug_num_images]

                    writer.write_image_grid(
                        current_step,
                        jax.device_get(_latents_to_uint8_images(latent_manager, jnp.asarray(x))),
                        debug_grid_size,
                        key="debug_clean_latents",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(_latents_to_uint8_images(latent_manager, jnp.asarray(z_t))),
                        debug_grid_size,
                        key="debug_noisy_latents",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(
                            _velocity_step_to_uint8_images(
                                latent_manager,
                                jnp.asarray(z_t),
                                jnp.asarray(v_u),
                                debug_velocity_decode_scale,
                            )
                        ),
                        debug_grid_size,
                        key="debug_v_u_step",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(
                            _velocity_step_to_uint8_images(
                                latent_manager,
                                jnp.asarray(z_t),
                                jnp.asarray(v_c),
                                debug_velocity_decode_scale,
                            )
                        ),
                        debug_grid_size,
                        key="debug_v_c_step",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(
                            _velocity_step_to_uint8_images(
                                latent_manager,
                                jnp.asarray(z_t),
                                jnp.asarray(v_pred),
                                debug_velocity_decode_scale,
                            )
                        ),
                        debug_grid_size,
                        key="debug_v_pred_step",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(
                            _velocity_step_to_uint8_images(
                                latent_manager,
                                jnp.asarray(z_t),
                                jnp.asarray(V),
                                debug_velocity_decode_scale,
                            )
                        ),
                        debug_grid_size,
                        key="debug_V_step",
                    )
                    writer.write_image_grid(
                        current_step,
                        jax.device_get(
                            _velocity_step_to_uint8_images(
                                latent_manager,
                                jnp.asarray(z_t),
                                jnp.asarray(v_g),
                                debug_velocity_decode_scale,
                            )
                        ),
                        debug_grid_size,
                        key="debug_v_g_step",
                    )

            ########### Sampling ###########
            if did_update and current_step > 0 and current_step % config.training.sample_per_step == 0:
                num_images = config.fid.num_images_to_log
                grid_size = int(num_images ** 0.5)
                if grid_size ** 2 != num_images:
                    raise ValueError(
                        f"config.fid.num_images_to_log must be a perfect square, got {num_images}"
                    )
                log_for_0("Logging %d preview samples at step %d.", num_images, current_step)
                samples = generate_fid_samples(
                    state,
                    config,
                    p_sample_step,
                    partial(run_p_sample_step, latent_manager=latent_manager),
                    use_ema,
                    num_samples=num_images,
                    **sample_kwargs,
                )
                writer.write_image_grid(current_step, samples, grid_size)

            ########### FID ###########
            if did_update and current_step > 0 and current_step % config.training.fid_per_step == 0:
                result = image_metric_evaluator(
                    state, p_sample_step, current_step - 1, **sample_kwargs
                )
                fid = result["fid"]
                if fid < best_fid:
                    best_fid = fid
                    log_for_0(
                        "New best FID %.4f at step %d. Saving best checkpoint to %s.",
                        best_fid,
                        current_step,
                        best_fid_ckpt_dir,
                    )
                    save_best_checkpoint(state, best_fid_ckpt_dir)

            if max_train_steps is not None and current_step >= max_train_steps:
                should_stop = True
                break

        ########### Save Checkpoint ###########
        if (
            not save_best_fid_only
            and (
                should_stop
                or (epoch + 1) % config.training.checkpoint_per_epoch == 0
                or (epoch + 1) == config.training.num_epochs
            )
        ):
            save_checkpoint(state, workdir)

        if should_stop:
            log_for_0("Reached max_train_steps=%d at step %d.", max_train_steps, current_step)
            break

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state

########################################################
#                    Evaluation                        #
########################################################

def just_evaluate(config: ml_collections.ConfigDict, workdir: str):

    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert (
        config.load_from != ""
    ), "config.load_from must be specified for just_evaluate"

    ########### Initialize ###########
    writer = Writer(config, workdir)

    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        log_for_0("Inferred dataset.num_classes from latent data: {}".format(inferred_num_classes))

    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size
    use_ema = config.training.get("use_ema", True)

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model = iMeanFlow(**model_config, eval=True)

    ########### Restore lightweight Eval State ###########
    state = restore_eval_checkpoint(config.load_from, use_ema=use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(config.dataset.vae, device_bsz, image_size)

    ########### Create sample pmap ###########

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
    )

    image_metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    ############ Evaluate over CFG configs ###########
    best_fid = float("inf")
    best_is = float("-inf")
    best_fd_dino = float("inf")
    best_config = None
    best_fd_dino_config = None
    best_fd_dino_at_best_fid = None
    for omega, t_min, t_max in _get_eval_sampling_configs(config):
        kwargs = {"omega": omega, "t_min": t_min, "t_max": t_max}
        kwargs = jax_utils.replicate(kwargs)
        result = image_metric_evaluator(
            state, p_sample_step, step, not use_ema, **kwargs
        )
        fid = result["fid"]
        is_score = result["is"]
        fd_dino = result.get("fd_dino", None)

        if fid < best_fid:
            best_fid, best_is, best_config = fid, is_score, (omega, t_min, t_max)
            best_fd_dino_at_best_fid = fd_dino
        if fd_dino is not None and fd_dino < best_fd_dino:
            best_fd_dino = fd_dino
            best_fd_dino_config = (omega, t_min, t_max)

    summary = {
        'best_fid': best_fid,
        'best_is': best_is,
        'omega': best_config[0],
        't_min': best_config[1],
        't_max': best_config[2],
    }
    log_message = (
        f"Best FID achieved: {best_fid:.2f}, \n"
        f"IS achieved: {best_is:.2f}, \n"
        f"omega: {best_config[0]:.2f}, t_min: {best_config[1]:.2f}, t_max: {best_config[2]:.2f}"
    )
    if best_fd_dino_at_best_fid is not None:
        summary['best_fd_dino_at_best_fid'] = best_fd_dino_at_best_fid
        log_message += f", \nFD-DINO at best FID config: {best_fd_dino_at_best_fid:.2f}"
    if best_fd_dino_config is not None:
        summary['best_fd_dino'] = best_fd_dino
        summary['best_fd_dino_omega'] = best_fd_dino_config[0]
        summary['best_fd_dino_t_min'] = best_fd_dino_config[1]
        summary['best_fd_dino_t_max'] = best_fd_dino_config[2]
        log_message += (
            f", \nBest FD-DINO achieved: {best_fd_dino:.2f}, "
            f"omega: {best_fd_dino_config[0]:.2f}, "
            f"t_min: {best_fd_dino_config[1]:.2f}, "
            f"t_max: {best_fd_dino_config[2]:.2f}"
        )
    log_for_0(log_message)
    writer.write_scalars(step, summary)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
