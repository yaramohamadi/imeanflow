"""Training loop for the dedicated plain SiT path."""

import os

import jax
import jax.numpy as jnp
import ml_collections
import torch
from flax import jax_utils
from functools import partial
from jax import lax, random

from sit import PlainSiT
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
from utils.sit_sample_util import sample_step
from utils.sit_trainstate_util import EvalState, TrainState, create_eval_state, create_train_state
from utils.vae_util import LatentDist, LatentManager


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def train_step_with_vae(
    state, batch, rng_init, ema_fn, lr_fn, latent_manager, use_ema, grad_accum_steps
):
    """Perform one plain-SiT training step."""
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


class CachedLatentEncoder:
    """Lightweight latent sampler for cached mean/std latents."""

    def __init__(self):
        self.mean = jnp.array(
            [0.86488, -0.27787343, 0.21616915, 0.3738409], dtype=jnp.float32
        ).reshape(1, 4, 1, 1)
        self.std = jnp.array(
            [4.85503674, 5.31922414, 3.93725398, 3.9870003], dtype=jnp.float32
        ).reshape(1, 4, 1, 1)

    def cached_encode(self, cached_value, rng):
        latent = LatentDist(cached_value).sample(key=rng).transpose((0, 3, 1, 2))
        latent = (latent - self.mean) / self.std
        return latent.transpose((0, 2, 3, 1))


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


def _build_plain_sit(config, *, eval_mode=False):
    dtype = (
        get_sampling_param_dtype(config)
        if eval_mode
        else get_training_param_dtype(config)
    )
    dtype = dtype or jnp.float32
    return PlainSiT(
        model_str=config.model.model_str,
        dtype=dtype,
        num_classes=config.model.num_classes,
        class_dropout_prob=float(config.model.get("class_dropout_prob", 0.1)),
        target_use_null_class=bool(
            config.model.get("target_use_null_class", True)
        ),
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
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
        state = create_eval_state(
            random.key(config.training.seed),
            config,
            model,
            image_size,
        )
        state = restore_partial_checkpoint(
            state,
            load_path,
            target_model_config=config.model,
        )
        if use_ema:
            state = state.replace(ema_params=state.params)
        return state

    return restore_eval_checkpoint(load_path, use_ema=use_ema)


def _get_eval_sampling_configs(config):
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


def _write_eval_metrics_csv(
    workdir,
    *,
    eval_phase,
    metric_mode,
    training_step,
    sampling_num_steps,
    omega,
    t_min,
    t_max,
    fid,
    is_score,
    fd_dino,
    checkpoint_path,
    is_best_fid,
    is_best_fd_dino,
):
    append_eval_metrics_row(
        workdir,
        {
            "eval_phase": eval_phase,
            "metric_mode": metric_mode,
            "training_step": int(training_step),
            "sampling_num_steps": int(sampling_num_steps),
            "omega": float(omega),
            "t_min": float(t_min),
            "t_max": float(t_max),
            "fid": float(fid),
            "inception_score": float(is_score),
            "fd_dino": "" if fd_dino is None else float(fd_dino),
            "is_best_fid": int(bool(is_best_fid)),
            "is_best_fd_dino": int(bool(is_best_fd_dino)),
            "checkpoint_path": checkpoint_path,
        },
    )


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> EvalState:
    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert config.load_from != "", "config.load_from must be specified for just_evaluate"

    writer = Writer(config, workdir)

    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0(
            "Inferred dataset.num_classes from latent data: %s",
            inferred_num_classes,
        )

    image_size = config.dataset.image_size
    metric_device_bsz = int(config.fid.device_batch_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    use_ema = config.training.get("use_ema", True)
    log_for_0("config.fid.device_batch_size: %s", metric_device_bsz)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)

    model = _build_plain_sit(config, eval_mode=True)
    state = _restore_eval_state(config, model, image_size, use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    latent_manager = LatentManager(
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

    image_metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    best_fid = float("inf")
    best_is = float("-inf")
    best_fd_dino = float("inf")
    best_config = None
    best_fd_dino_config = None
    best_fd_dino_at_best_fid = None
    csv_rows = []
    metric_mode = _primary_metric_mode(use_ema)

    for omega, t_min, t_max in _get_eval_sampling_configs(config):
        kwargs = jax_utils.replicate(
            {"omega": omega, "t_min": t_min, "t_max": t_max},
            devices=sample_devices,
        )
        result = image_metric_evaluator(
            state,
            p_sample_step,
            step,
            not use_ema,
            **kwargs,
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
        csv_rows.append(
            {
                "omega": omega,
                "t_min": t_min,
                "t_max": t_max,
                "fid": fid,
                "is": is_score,
                "fd_dino": fd_dino,
            }
        )

    for row in csv_rows:
        row_config = (row["omega"], row["t_min"], row["t_max"])
        _write_eval_metrics_csv(
            workdir,
            eval_phase="eval_only",
            metric_mode=metric_mode,
            training_step=step,
            sampling_num_steps=config.sampling.num_steps,
            omega=row["omega"],
            t_min=row["t_min"],
            t_max=row["t_max"],
            fid=row["fid"],
            is_score=row["is"],
            fd_dino=row["fd_dino"],
            checkpoint_path=os.path.abspath(config.load_from),
            is_best_fid=(row_config == best_config),
            is_best_fd_dino=(
                best_fd_dino_config is not None and row_config == best_fd_dino_config
            ),
        )

    summary = {
        "best_fid": best_fid,
        "best_is": best_is,
        "omega": best_config[0],
        "t_min": best_config[1],
        "t_max": best_config[2],
    }
    log_message = (
        f"Best FID achieved: {best_fid:.2f}, \n"
        f"IS achieved: {best_is:.2f}, \n"
        f"omega: {best_config[0]:.2f}, t_min: {best_config[1]:.2f}, t_max: {best_config[2]:.2f}"
    )
    if best_fd_dino_at_best_fid is not None:
        summary["best_fd_dino_at_best_fid"] = best_fd_dino_at_best_fid
        log_message += f", \nFD-DINO at best FID config: {best_fd_dino_at_best_fid:.2f}"
    if best_fd_dino_config is not None:
        summary["best_fd_dino"] = best_fd_dino
        summary["best_fd_dino_omega"] = best_fd_dino_config[0]
        summary["best_fd_dino_t_min"] = best_fd_dino_config[1]
        summary["best_fd_dino_t_max"] = best_fd_dino_config[2]
        log_message += (
            f", \nBest FD-DINO achieved: {best_fd_dino:.2f}, "
            f"omega: {best_fd_dino_config[0]:.2f}, "
            f"t_min: {best_fd_dino_config[1]:.2f}, "
            f"t_max: {best_fd_dino_config[2]:.2f}"
        )
    log_for_0(log_message)
    writer.write_scalars(step, summary)

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    writer = Writer(config, workdir)

    if config.eval_only:
        raise ValueError(
            "train_and_evaluate() only handles training runs. "
            "Use main_sit.py with --config.eval_only=True to route into just_evaluate()."
        )

    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0(
            "Inferred dataset.num_classes from latent data: %s",
            inferred_num_classes,
        )

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
    log_for_0("config.training.sample_per_step: %s", config.training.sample_per_step)
    log_for_0("config.training.fid_per_step: %s", config.training.fid_per_step)
    log_for_0("config.fid.device_batch_size: %s", metric_device_bsz)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)

    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0("local_batch_size: %s", local_batch_size)
    log_for_0("jax.local_device_count: %s", jax.local_device_count())

    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: %s", steps_per_epoch)

    model = _build_plain_sit(config, eval_mode=False)
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)
    state = _load_initial_state(state, config)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch
    state = jax_utils.replicate(state)

    latent_encoder = CachedLatentEncoder()
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

    metrics_tracker = MetricsTracker()
    timer = Timer()
    should_stop = False
    log_for_0("Initial compilation, this might take some minutes...")

    sample_latent_manager = LatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=sample_local_device_count,
    )
    sample_model = _build_plain_sit(config, eval_mode=True)

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
    preview_num_steps = config.training.get("preview_num_steps", (1, 2, 4))
    preview_num_steps = (
        tuple(int(num) for num in preview_num_steps)
        if preview_num_steps
        else (1, 2, 4)
    )
    if any(num_steps < 1 for num_steps in preview_num_steps):
        raise ValueError("Plain SiT preview sampling requires all preview_num_steps >= 1.")
    p_preview_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in preview_num_steps
    }
    preview_guidance_scales = config.training.get("preview_guidance_scales", [])
    if preview_guidance_scales:
        preview_guidance_scales = [float(omega) for omega in preview_guidance_scales]
    else:
        preview_guidance_scales = [float(config.sampling.omega)]
    preview_t_min = float(config.sampling.t_min)
    preview_t_max = float(config.sampling.t_max)
    sample_kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.omega),
            "t_min": preview_t_min,
            "t_max": preview_t_max,
        },
        devices=sample_devices,
    )

    def log_preview_samples(state_for_logging, step_for_logging):
        num_images = min(
            int(config.fid.num_images_to_log),
            int(sample_latent_manager.batch_size),
        )
        grid_size = int(num_images ** 0.5)
        num_images = grid_size ** 2
        if num_images <= 0:
            raise ValueError(
                "Preview logging requires at least one square grid image on the first device."
            )

        log_for_0(
            "Logging %d preview samples at step %d for %s steps and omegas %s.",
            num_images,
            step_for_logging,
            preview_num_steps,
            preview_guidance_scales,
        )
        preview_image_groups = {}
        for omega in preview_guidance_scales:
            preview_kwargs = jax_utils.replicate(
                {
                    "omega": omega,
                    "t_min": preview_t_min,
                    "t_max": preview_t_max,
                },
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
            preview_image_groups[
                format_preview_guidance_label(omega, preview_t_min, preview_t_max)
            ] = [preview_panel]

        stacked_preview_panel = make_stacked_grid_panel(preview_image_groups, 1)
        writer.write_images(step_for_logging, {"image_grid": stacked_preview_panel})

    image_metric_evaluator = get_image_metric_evaluator(
        config,
        writer,
        sample_latent_manager,
    )
    best_fid_by_steps = {num_steps: float("inf") for num_steps in metric_num_steps}
    best_fd_dino_by_steps = {
        num_steps: float("inf") for num_steps in metric_num_steps
    }
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )
    save_best_fid_only = config.training.get("save_best_fid_only", False)
    save_eval_checkpoint_per_fid = config.training.get(
        "save_eval_checkpoint_per_fid", False
    )
    eval_ckpt_dir = os.path.join(
        workdir,
        config.training.get("eval_checkpoint_dir", "latest_eval"),
    )
    metric_mode = _primary_metric_mode(use_ema)

    initial_step = int(jax.device_get(state.step)[0])
    if initial_step == 0 and config.training.sample_per_step > 0:
        log_preview_samples(state, 0)

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
                compilation_time = timer.elapse_with_reset()
                log_for_0("p_train_step compiled in %.2fs", compilation_time)

            metrics_tracker.update(metrics)
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

            if (
                did_update
                and config.training.sample_per_step > 0
                and current_step > 0
                and current_step % config.training.sample_per_step == 0
            ):
                log_preview_samples(state, current_step)

            if did_update and current_step > 0 and _should_run_fid(current_step, config.training):
                checkpoint_path_for_csv = ""
                if save_eval_checkpoint_per_fid:
                    log_for_0(
                        "Saving latest evaluation checkpoint at step %d to %s.",
                        current_step,
                        eval_ckpt_dir,
                    )
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
                        log_for_0(
                            "New best %d-step FID %.4f at step %d. Saving best checkpoint to %s.",
                            metric_num_steps,
                            fid,
                            current_step,
                            best_fid_ckpt_dir,
                        )
                        save_best_checkpoint(state, best_fid_ckpt_dir)
                        row_checkpoint_path = best_fid_ckpt_dir

                    _write_eval_metrics_csv(
                        workdir,
                        eval_phase="train",
                        metric_mode=metric_mode,
                        training_step=current_step,
                        sampling_num_steps=metric_num_steps,
                        omega=float(config.sampling.omega),
                        t_min=float(config.sampling.t_min),
                        t_max=float(config.sampling.t_max),
                        fid=fid,
                        is_score=result["is"],
                        fd_dino=fd_dino,
                        checkpoint_path=(
                            os.path.abspath(row_checkpoint_path)
                            if row_checkpoint_path
                            else ""
                        ),
                        is_best_fid=is_best_fid,
                        is_best_fd_dino=is_best_fd_dino,
                    )

            if max_train_steps is not None and current_step >= max_train_steps:
                should_stop = True
                break

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
            log_for_0(
                "Reached max_train_steps=%d at step %d.",
                max_train_steps,
                current_step,
            )
            break

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
