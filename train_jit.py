"""JAX training loop for pixel-space JiT fine-tuning."""

import os
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import jax_utils
from jax import lax, random

from plain_jit import PlainJiT
import utils.input_pipeline as input_pipeline
from utils.ckpt_util import (
    restore_checkpoint,
    restore_eval_checkpoint,
    restore_partial_checkpoint,
    save_checkpoint,
)
from utils.ema_util import ema_schedules, update_ema
from utils.logging_util import MetricsTracker, Timer, Writer, log_for_0
from utils.lr_utils import lr_schedules
from utils.preview_util import make_uint8_image_grid
from utils.sample_util import get_sampling_param_dtype, get_training_param_dtype
from utils.sit_trainstate_util import EvalState, TrainState


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    return lax.pmean(metrics, axis_name="batch")


def infer_num_classes_from_images(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        raise ValueError(f"Image train directory not found: {train_root}")
    class_dirs = [
        name
        for name in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, name))
    ]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {train_root}")
    return len(class_dirs)


def initialized(key, config, model):
    image_size = int(config.dataset.image_size)
    image_channels = int(config.dataset.image_channels)
    x = jnp.ones((1, image_size, image_size, image_channels), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    y = jnp.ones((1,), dtype=jnp.int32)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing JiT params...")
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing JiT params done.")
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: %s", param_count)
    return variables, variables["params"]


def create_jit_train_state(rng, config, model, lr_fn):
    rng, rng_init = random.split(rng)
    _, params = initialized(rng_init, config, model)
    use_ema = config.training.get("use_ema", True)
    ema_params = deepcopy(params)
    if use_ema:
        ema_params = update_ema(ema_params, params, 0)

    grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad_accum_step = jnp.array(0, dtype=jnp.int32)
    tx = optax.adamw(
        learning_rate=lr_fn,
        weight_decay=0,
        b2=config.training.adam_b2,
    )
    return TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        grad_accum=grad_accum,
        grad_accum_step=grad_accum_step,
        tx=tx,
    )


def create_jit_eval_state(rng, config, model):
    rng, rng_init = random.split(rng)
    _, params = initialized(rng_init, config, model)
    return EvalState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        ema_params=None,
    )


def train_step(state, batch, rng_init, ema_fn, lr_fn, use_ema, grad_accum_steps):
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    images = batch["image"]
    labels = batch["label"]

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
        lambda acc, g: acc + g,
        state.grad_accum,
        grads,
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


def sample_step(
    variable,
    sample_idx,
    *,
    model,
    rng_init,
    config,
    device_batch_size,
    num_steps,
    omega,
    t_min,
    t_max,
):
    sample_idx = jnp.asarray(sample_idx, dtype=jnp.int32)
    labels = sample_idx * device_batch_size + jnp.arange(device_batch_size)
    labels = labels % int(config.dataset.num_classes)

    sample_shape = (
        labels.shape[0],
        int(config.dataset.image_size),
        int(config.dataset.image_size),
        int(config.dataset.image_channels),
    )
    rng = random.fold_in(rng_init, lax.axis_index(axis_name="batch"))
    rng = random.fold_in(rng, sample_idx)
    z = random.normal(rng, sample_shape, dtype=model.dtype)
    t_steps = jnp.linspace(0.0, 1.0, int(num_steps) + 1, dtype=model.dtype)

    method = config.sampling.get("method", "euler")

    def body_fn(i, current):
        t = jnp.full((current.shape[0],), t_steps[i], dtype=model.dtype)
        t_next = jnp.full((current.shape[0],), t_steps[i + 1], dtype=model.dtype)
        if method == "heun":
            return model.apply(
                variable,
                current,
                t,
                t_next,
                labels,
                omega,
                t_min,
                t_max,
                method=model.heun_step,
            )
        return model.apply(
            variable,
            current,
            t,
            t_next,
            labels,
            omega,
            t_min,
            t_max,
            method=model.euler_step,
        )

    return jax.lax.fori_loop(0, int(num_steps), body_fn, z)


def _build_plain_jit(config, *, eval_mode=False):
    dtype = (
        get_sampling_param_dtype(config)
        if eval_mode
        else get_training_param_dtype(config)
    )
    dtype = dtype or jnp.float32
    return PlainJiT(
        model_str=config.model.model_str,
        dtype=dtype,
        input_size=int(config.dataset.image_size),
        in_channels=int(config.dataset.image_channels),
        num_classes=int(config.model.num_classes),
        class_dropout_prob=float(config.model.get("class_dropout_prob", 0.1)),
        target_use_null_class=bool(config.model.get("target_use_null_class", True)),
        P_mean=float(config.model.get("P_mean", -0.8)),
        P_std=float(config.model.get("P_std", 0.8)),
        t_eps=float(config.model.get("t_eps", 1e-5)),
        noise_scale=float(config.model.get("noise_scale", 1.0)),
        attn_drop=float(config.model.get("attn_drop", 0.0)),
        proj_drop=float(config.model.get("proj_drop", 0.0)),
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


def _restore_eval_state(config, model, use_ema):
    load_path = os.path.abspath(config.load_from)
    is_torch_ckpt = os.path.isfile(load_path) and load_path.endswith(
        (".pt", ".pth", ".pth.tar")
    )
    if is_torch_ckpt or config.get("partial_load", False):
        state = create_jit_eval_state(random.key(config.training.seed), config, model)
        state = restore_partial_checkpoint(
            state,
            load_path,
            target_model_config=config.model,
        )
        if use_ema:
            state = state.replace(ema_params=state.params)
        return state
    return restore_eval_checkpoint(load_path, use_ema=use_ema)


def _set_num_classes_from_data(config):
    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_images(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0("Inferred dataset.num_classes from image data: %s", inferred_num_classes)


def _samples_to_uint8(samples):
    samples = jnp.clip(samples, -1.0, 1.0)
    samples = 127.5 * samples + 128.0
    return jnp.clip(samples, 0, 255).astype(jnp.uint8)


def _preview_samples(state, p_sample_step, config, model, use_ema, num_images):
    grid_size = int(num_images ** 0.5)
    num_images = grid_size ** 2
    if num_images <= 0:
        return None
    sample_device_batch_size = int(config.fid.get("sample_device_batch_size", 1))
    sample_idx = jnp.arange(jax.local_device_count(), dtype=jnp.int32)
    params = state.ema_params if use_ema else state.params
    variable = {"params": params}
    kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.get("omega", 1.0)),
            "t_min": float(config.sampling.get("t_min", 0.0)),
            "t_max": float(config.sampling.get("t_max", 1.0)),
        }
    )
    samples = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
    samples = samples.reshape(-1, *samples.shape[2:])
    samples = _samples_to_uint8(samples[:num_images])
    return make_uint8_image_grid(jax.device_get(samples), grid_size)


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> EvalState:
    assert config.eval_only
    assert config.load_from != ""
    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)
    use_ema = config.training.get("use_ema", True)
    model = _build_plain_jit(config, eval_mode=True)
    state = _restore_eval_state(config, model, use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=int(config.fid.get("sample_device_batch_size", 1)),
            num_steps=int(config.sampling.num_steps),
        ),
        axis_name="batch",
    )
    preview = _preview_samples(
        state,
        p_sample_step,
        config,
        model,
        use_ema,
        int(config.fid.get("num_images_to_log", 16)),
    )
    if preview is not None:
        writer.write_images(step, {"image_grid": preview})
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    writer = Writer(config, workdir)
    if config.eval_only:
        raise ValueError("Use just_evaluate() for eval_only JiT runs.")
    _set_num_classes_from_data(config)

    rng = random.key(config.training.seed)
    use_ema = config.training.get("use_ema", True)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = int(config.training.get("grad_accum_steps", 1))

    log_for_0("config.training.batch_size: %s", config.training.batch_size)
    log_for_0("config.training.use_ema: %s", use_ema)
    log_for_0("config.training.max_train_steps: %s", max_train_steps)
    log_for_0("config.training.grad_accum_steps: %s", grad_accum_steps)

    local_batch_size = int(config.training.batch_size) // jax.process_count()
    train_loader, steps_per_epoch = input_pipeline.create_image_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: %s", steps_per_epoch)

    model = _build_plain_jit(config, eval_mode=False)
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_jit_train_state(rng, config, model, lr_fn)
    state = _load_initial_state(state, config)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            use_ema=use_ema,
            grad_accum_steps=grad_accum_steps,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    sample_model = _build_plain_jit(config, eval_mode=True)
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=sample_model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=int(config.fid.get("sample_device_batch_size", 1)),
            num_steps=int(config.sampling.num_steps),
        ),
        axis_name="batch",
    )

    metrics_tracker = MetricsTracker()
    timer = Timer()
    should_stop = False
    log_for_0("Initial compilation, this might take some minutes...")

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
                preview = _preview_samples(
                    state,
                    p_sample_step,
                    config,
                    sample_model,
                    use_ema,
                    int(config.fid.get("num_images_to_log", 16)),
                )
                if preview is not None:
                    writer.write_images(current_step, {"image_grid": preview})

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
