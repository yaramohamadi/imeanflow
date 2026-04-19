"""JAX training loop for original-objective DiT on cached latents."""

import os
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import torch
from flax import jax_utils
from jax import lax, random

from dit import PlainDiT
import utils.input_pipeline as input_pipeline
from utils.ckpt_util import (
    restore_checkpoint,
    restore_eval_checkpoint,
    restore_partial_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from utils.ema_util import ema_schedules, update_ema
from utils.eval_csv_util import append_eval_metrics_row
from utils.logging_util import MetricsTracker, Timer, Writer, log_for_0
from utils.lr_utils import lr_schedules
from utils.preview_util import (
    format_preview_guidance_label,
    generate_preview_samples_first_device,
    make_side_by_side_preview_panel,
    make_stacked_grid_panel,
)
from utils.sample_util import (
    get_image_metric_evaluator,
    get_sample_device_batch_size,
    get_sample_devices,
    get_sample_local_device_count,
    get_sampling_param_dtype,
    get_training_param_dtype,
)
from utils.dit_sample_util import get_default_cfg_scale, sample_step
from utils.sit_trainstate_util import EvalState, TrainState, create_eval_state, create_train_state
from utils.vae_util import DiTLatentManager


class CachedDiTLatentEncoder:
    """Samples cached VAE mean/std latents and applies the official DiT scale."""

    def __init__(self, latent_scale=0.18215):
        self.latent_scale = float(latent_scale)

    def cached_encode(self, cached_value, rng):
        mean, std = jnp.split(cached_value, 2, axis=-1)
        latent = mean + std * jax.random.normal(rng, mean.shape, dtype=mean.dtype)
        return latent * self.latent_scale


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    return lax.pmean(metrics, axis_name="batch")


def train_step_with_vae(
    state, batch, rng_init, ema_fn, lr_fn, latent_manager, use_ema, grad_accum_steps
):
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]
    labels = batch["label"]
    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)

    def loss_fn(params):
        return state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            rngs=dict(gen=rng_base),
        )

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    lr_value = lr_fn(state.step)
    metrics = compute_metrics(aux[1])
    metrics["lr"] = lr_value

    new_grad_accum = jax.tree_util.tree_map(
        lambda acc, g: acc + g, state.grad_accum, grads
    )
    new_accum_step = state.grad_accum_step + 1
    should_apply = new_accum_step >= grad_accum_steps

    def apply_update(args):
        current_state, accum_grads = args
        mean_grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, accum_grads)
        updated_state = current_state.apply_gradients(grads=mean_grads)
        if use_ema:
            ema_value = ema_fn(current_state.step)
            new_ema = update_ema(updated_state.ema_params, updated_state.params, ema_value)
            updated_state = updated_state.replace(ema_params=new_ema)
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return updated_state.replace(
            grad_accum=zero_accum,
            grad_accum_step=jnp.array(0, dtype=jnp.int32),
        )

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


def infer_num_classes_from_latents(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        raise ValueError(f"Latent train directory not found: {train_root}")
    max_label = -1
    num_files = 0
    for filename in os.listdir(train_root):
        if not filename.endswith(".pt"):
            continue
        sample = torch.load(os.path.join(train_root, filename), map_location="cpu")
        max_label = max(max_label, int(sample["label"]))
        num_files += 1
    if num_files == 0:
        raise ValueError(f"No latent .pt files found under: {train_root}")
    return max_label + 1


def _build_plain_dit(config, *, eval_mode=False):
    dtype = (
        get_sampling_param_dtype(config)
        if eval_mode
        else get_training_param_dtype(config)
    )
    dtype = dtype or jnp.float32
    return PlainDiT(
        model_str=config.model.model_str,
        dtype=dtype,
        num_classes=config.model.num_classes,
        class_dropout_prob=float(config.model.get("class_dropout_prob", 0.1)),
        target_use_null_class=bool(config.model.get("target_use_null_class", True)),
        diffusion_steps=int(config.diffusion.diffusion_steps),
        noise_schedule=config.diffusion.noise_schedule,
        learn_sigma=bool(config.diffusion.learn_sigma),
        predict_xstart=bool(config.diffusion.predict_xstart),
        rescale_learned_sigmas=bool(config.diffusion.rescale_learned_sigmas),
        eval=eval_mode,
    )


def _load_initial_state(state, config):
    if not config.load_from:
        return state
    load_path = os.path.abspath(config.load_from)
    is_torch_ckpt = os.path.isfile(load_path) and load_path.endswith(
        (".pt", ".pth", ".pth.tar")
    )
    if is_torch_ckpt or config.get("partial_load", False):
        return restore_partial_checkpoint(
            state,
            load_path,
            target_model_config=config.model,
        )
    return restore_checkpoint(state, load_path)


def _restore_eval_state(config, model, image_size, use_ema):
    load_path = os.path.abspath(config.load_from)
    is_torch_ckpt = os.path.isfile(load_path) and load_path.endswith(
        (".pt", ".pth", ".pth.tar")
    )
    if is_torch_ckpt or config.get("partial_load", False):
        state = create_eval_state(random.key(config.training.seed), config, model, image_size)
        state = restore_partial_checkpoint(
            state,
            load_path,
            target_model_config=config.model,
        )
        if use_ema:
            state = state.replace(ema_params=state.params)
        return state
    return restore_eval_checkpoint(load_path, use_ema=use_ema)


def _should_run_fid(current_step, training_config):
    fid_schedule = training_config.get("fid_schedule", [])
    if fid_schedule:
        for schedule_item in fid_schedule:
            from_step = int(schedule_item.get("from_step", 0))
            until_step = schedule_item.get("until_step", None)
            every_steps = int(schedule_item["every_steps"])
            if current_step < from_step:
                continue
            if until_step is not None and current_step >= int(until_step):
                continue
            if (current_step - from_step) % every_steps == 0:
                return True
        return False
    fid_per_step = int(training_config.get("fid_per_step", 0))
    return fid_per_step > 0 and current_step % fid_per_step == 0


def _get_metric_num_steps(config):
    configured_steps = config.training.get("metric_num_steps", ())
    if configured_steps:
        num_steps = [int(step) for step in configured_steps]
    else:
        num_steps = [int(config.sampling.num_steps)]

    primary_steps = int(config.sampling.num_steps)
    ordered_steps = []
    for step in [primary_steps] + num_steps:
        if step < 1:
            raise ValueError("Metric sampling steps must be >= 1.")
        if step not in ordered_steps:
            ordered_steps.append(step)
    return tuple(ordered_steps)


def _primary_metric_mode(use_ema):
    return "ema" if use_ema else "online"


def _write_eval_metrics_csv(workdir, **row):
    append_eval_metrics_row(workdir, row)


def _set_num_classes_from_data(config):
    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0("Inferred dataset.num_classes from latent data: %s", inferred_num_classes)


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> EvalState:
    assert config.eval_only
    assert config.load_from != ""
    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)

    image_size = config.dataset.image_size
    metric_device_bsz = int(config.fid.device_batch_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    use_ema = config.training.get("use_ema", True)
    log_for_0("config.fid.device_batch_size: %s", metric_device_bsz)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)

    model = _build_plain_dit(config, eval_mode=True)
    state = _restore_eval_state(config, model, image_size, use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    latent_manager = DiTLatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=sample_local_device_count,
    )
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=sample_device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
        devices=sample_devices,
    )
    evaluator = get_image_metric_evaluator(config, writer, latent_manager)
    kwargs = jax_utils.replicate(
        {
            "omega": get_default_cfg_scale(config),
            "t_min": 0.0,
            "t_max": 1.0,
        },
        devices=sample_devices,
    )
    result = evaluator(state, p_sample_step, step, not use_ema, **kwargs)
    _write_eval_metrics_csv(
        workdir,
        eval_phase="eval_only",
        metric_mode=_primary_metric_mode(use_ema),
        training_step=step,
        sampling_num_steps=config.sampling.num_steps,
        omega=get_default_cfg_scale(config),
        t_min=0.0,
        t_max=1.0,
        fid=float(result["fid"]),
        inception_score=float(result["is"]),
        fd_dino="" if result.get("fd_dino") is None else float(result["fd_dino"]),
        is_best_fid=1,
        is_best_fd_dino=1 if result.get("fd_dino") is not None else 0,
        checkpoint_path=os.path.abspath(config.load_from),
    )
    writer.write_scalars(step, {"fid": result["fid"], "is": result["is"]})
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    writer = Writer(config, workdir)
    if config.eval_only:
        raise ValueError("Use just_evaluate() for eval_only DiT runs.")
    _set_num_classes_from_data(config)

    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    metric_device_bsz = int(config.fid.device_batch_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    use_ema = config.training.get("use_ema", True)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = config.training.get("grad_accum_steps", 1)

    log_for_0("config.training.batch_size: %s", config.training.batch_size)
    log_for_0("config.training.use_ema: %s", use_ema)
    log_for_0("config.training.max_train_steps: %s", max_train_steps)
    log_for_0("config.training.grad_accum_steps: %s", grad_accum_steps)
    log_for_0("config.fid.device_batch_size: %s", metric_device_bsz)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)

    local_batch_size = config.training.batch_size // jax.process_count()
    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: %s", steps_per_epoch)

    model = _build_plain_dit(config, eval_mode=False)
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)
    state = _load_initial_state(state, config)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch
    state = jax_utils.replicate(state)

    latent_encoder = CachedDiTLatentEncoder()
    p_train_step = jax.pmap(
        partial(
            train_step_with_vae,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            latent_manager=latent_encoder,
            use_ema=use_ema,
            grad_accum_steps=grad_accum_steps,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    sample_latent_manager = DiTLatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=sample_local_device_count,
    )
    sample_model = _build_plain_dit(config, eval_mode=True)

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=sample_model,
                rng_init=random.PRNGKey(99),
                config=config,
                device_batch_size=sample_device_bsz,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in metric_num_steps
    }
    log_for_0("Metric evaluation sampling steps: %s", metric_num_steps)
    preview_num_steps = config.training.get("preview_num_steps", (50, 250))
    preview_num_steps = tuple(int(num) for num in preview_num_steps) if preview_num_steps else (250,)
    p_preview_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in preview_num_steps
    }
    preview_guidance_scales = config.training.get("preview_guidance_scales", [])
    if preview_guidance_scales:
        preview_guidance_scales = [float(omega) for omega in preview_guidance_scales]
    else:
        preview_guidance_scales = [get_default_cfg_scale(config)]

    def log_preview_samples(state_for_logging, step_for_logging):
        num_images = min(int(config.fid.num_images_to_log), int(sample_latent_manager.batch_size))
        grid_size = int(num_images ** 0.5)
        num_images = grid_size ** 2
        if num_images <= 0:
            return
        preview_image_groups = {}
        for cfg_scale in preview_guidance_scales:
            preview_kwargs = jax_utils.replicate(
                {"omega": cfg_scale, "t_min": 0.0, "t_max": 1.0},
                devices=sample_devices,
            )
            preview_images = {}
            for num_steps, p_preview_step in p_preview_sample_steps.items():
                preview_images[num_steps] = generate_preview_samples_first_device(
                    state_for_logging,
                    p_preview_step,
                    sample_latent_manager,
                    use_ema,
                    num_samples=num_images,
                    param_dtype=get_sampling_param_dtype(config),
                    sample_local_device_count=sample_local_device_count,
                    **preview_kwargs,
                )
            preview_panel = make_side_by_side_preview_panel(preview_images, grid_size)
            preview_image_groups[format_preview_guidance_label(cfg_scale, 0.0, 1.0)] = [
                preview_panel
            ]
        writer.write_images(
            step_for_logging,
            {"image_grid": make_stacked_grid_panel(preview_image_groups, 1)},
        )

    image_metric_evaluator = get_image_metric_evaluator(config, writer, sample_latent_manager)
    best_fid_by_steps = {num_steps: float("inf") for num_steps in metric_num_steps}
    best_fd_dino_by_steps = {
        num_steps: float("inf") for num_steps in metric_num_steps
    }
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )
    eval_ckpt_dir = os.path.join(
        workdir,
        config.training.get("eval_checkpoint_dir", "latest_eval"),
    )
    metric_mode = _primary_metric_mode(use_ema)
    metrics_tracker = MetricsTracker()
    timer = Timer()
    should_stop = False
    log_for_0("Initial compilation, this might take some minutes...")

    initial_step = int(jax.device_get(state.step)[0])
    if initial_step == 0 and config.training.sample_per_step > 0:
        log_preview_samples(state, 0)

    sample_kwargs = jax_utils.replicate(
        {"omega": get_default_cfg_scale(config), "t_min": 0.0, "t_max": 1.0},
        devices=sample_devices,
    )

    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch %s...", epoch)
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            batch = input_pipeline.prepare_batch_data(batch)
            state, metrics = p_train_step(state, batch)
            current_step = int(jax.device_get(state.step)[0])
            did_update = bool(jax.device_get(metrics["did_update"])[0])

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                log_for_0("p_train_step compiled in %.2fs", timer.elapse_with_reset())

            metrics_tracker.update(metrics)
            should_log = did_update and current_step > 0 and (
                current_step == 1 or current_step % config.training.log_per_step == 0
            )
            if should_log:
                summary = metrics_tracker.finalize()
                logged_steps = 1 if current_step == 1 else config.training.log_per_step
                summary["steps_per_second"] = logged_steps / timer.elapse_with_reset()
                summary.pop("did_update", None)
                writer.write_scalars(current_step, summary)

            if (
                did_update
                and config.training.sample_per_step > 0
                and current_step > 0
                and current_step % config.training.sample_per_step == 0
            ):
                log_preview_samples(state, current_step)

            if did_update and current_step > 0 and _should_run_fid(current_step, config.training):
                checkpoint_path_for_csv = ""
                if config.training.get("save_eval_checkpoint_per_fid", False):
                    save_best_checkpoint(state, eval_ckpt_dir)
                    checkpoint_path_for_csv = eval_ckpt_dir

                for metric_num_steps, p_metric_sample_step in p_metric_sample_steps.items():
                    log_for_0(
                        "Running metric evaluation at step %d with %d sampling steps.",
                        current_step,
                        metric_num_steps,
                    )
                    result = image_metric_evaluator(
                        state,
                        p_metric_sample_step,
                        current_step - 1,
                        metric_suffix=f"steps_{metric_num_steps}",
                        **sample_kwargs,
                    )
                    fid = result["fid"]
                    fd_dino = result.get("fd_dino", None)
                    is_best_fid = fid < best_fid_by_steps[metric_num_steps]
                    is_best_fd_dino = (
                        fd_dino is not None
                        and fd_dino < best_fd_dino_by_steps[metric_num_steps]
                    )
                    if is_best_fid:
                        best_fid_by_steps[metric_num_steps] = fid
                    if is_best_fd_dino:
                        best_fd_dino_by_steps[metric_num_steps] = fd_dino

                    row_checkpoint_path = checkpoint_path_for_csv
                    if (
                        metric_num_steps == int(config.sampling.num_steps)
                        and is_best_fid
                    ):
                        save_best_checkpoint(state, best_fid_ckpt_dir)
                        row_checkpoint_path = best_fid_ckpt_dir

                    _write_eval_metrics_csv(
                        workdir,
                        eval_phase="train",
                        metric_mode=metric_mode,
                        training_step=current_step,
                        sampling_num_steps=metric_num_steps,
                        omega=get_default_cfg_scale(config),
                        t_min=0.0,
                        t_max=1.0,
                        fid=float(fid),
                        inception_score=float(result["is"]),
                        fd_dino="" if fd_dino is None else float(fd_dino),
                        is_best_fid=int(is_best_fid),
                        is_best_fd_dino=int(is_best_fd_dino),
                        checkpoint_path=(
                            os.path.abspath(row_checkpoint_path)
                            if row_checkpoint_path
                            else ""
                        ),
                    )

            if max_train_steps is not None and current_step >= max_train_steps:
                should_stop = True
                break

        if (
            not config.training.get("save_best_fid_only", False)
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

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
