"""JAX training loop for pixel-space pMF fine-tuning."""

import os
from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import jax_utils, serialization, struct
from flax.training import checkpoints, train_state
from jax import lax, random

from plain_pmf import generate, pixelMeanFlow
import utils.input_pipeline as input_pipeline
from utils.auxloss_util import init_auxloss
from utils.ckpt_util import save_best_checkpoint, save_checkpoint
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
)
from utils.muon_util import muon as local_muon


class TrainState(train_state.TrainState):
    ema_params: Any
    grad_accum: Any
    grad_accum_step: Any


@struct.dataclass
class EvalState:
    step: Any
    params: Any
    ema_params: Any


class PixelImageManager:
    """Decode helper for pMF samples, which are already BHWC pixels in [-1, 1]."""

    def __init__(self, batch_size, decode_num_local_devices=None):
        self.batch_size = int(batch_size)
        self.decode_num_local_devices = (
            jax.local_device_count()
            if decode_num_local_devices is None
            else int(decode_num_local_devices)
        )

    def decode(self, images_bhwc):
        images_bhwc = jnp.clip(images_bhwc, -1.0, 1.0)
        return jnp.transpose(images_bhwc, (0, 3, 1, 2))


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

    log_for_0("Initializing pMF params...")
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing pMF params done.")
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: %s", param_count)
    return variables, variables["params"]


def create_pmf_train_state(rng, config, model, lr_fn):
    rng, rng_init = random.split(rng)
    _, params = initialized(rng_init, config, model)
    use_ema = config.training.get("use_ema", False)
    ema_params = deepcopy(params) if use_ema else None
    if use_ema:
        ema_params = update_ema(ema_params, params, 0)

    grad_accum_steps = int(config.training.get("grad_accum_steps", 1))
    grad_accum = (
        jax.tree_util.tree_map(jnp.zeros_like, params)
        if grad_accum_steps > 1
        else None
    )
    grad_accum_step = jnp.array(0, dtype=jnp.int32)
    tx = _create_optimizer(config, lr_fn)
    return TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        grad_accum=grad_accum,
        grad_accum_step=grad_accum_step,
        tx=tx,
    )


def _optional_optimizer_dtype(dtype_name):
    if dtype_name is None or dtype_name == "":
        return None
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("float16", "fp16"):
        return jnp.float16
    if dtype_name in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if dtype_name in ("float32", "fp32"):
        return jnp.float32
    raise ValueError(
        "optimizer state dtype must be one of float16/fp16, bfloat16/bf16, "
        f"float32/fp32, or empty. Got {dtype_name!r}."
    )


def _create_optimizer(config, lr_fn):
    optimizer = str(config.training.get("optimizer", "lion")).lower()
    mu_dtype = _optional_optimizer_dtype(
        config.training.get("optimizer_mu_dtype", None)
    )
    if optimizer == "adamw":
        return optax.adamw(
            learning_rate=lr_fn,
            b2=float(config.training.get("adam_b2", 0.95)),
            weight_decay=float(config.training.get("weight_decay", 0.0)),
            mu_dtype=mu_dtype,
        )
    if optimizer == "lion":
        return optax.lion(
            learning_rate=lr_fn,
            b2=float(config.training.get("lion_b2", 0.99)),
            weight_decay=float(config.training.get("weight_decay", 0.0)),
            mu_dtype=mu_dtype,
        )
    if optimizer == "muon":
        upstream_muon = getattr(getattr(optax, "contrib", None), "muon", None)
        muon_fn = upstream_muon or local_muon
        return muon_fn(
            learning_rate=lr_fn,
            beta=float(config.training.get("adam_b2", 0.95)),
            mu_dtype=mu_dtype,
            adam_b2=float(config.training.get("adam_b2", 0.95)),
            weight_decay=float(config.training.get("weight_decay", 0.0)),
        )
    if optimizer == "sgd":
        momentum = config.training.get("sgd_momentum", None)
        return optax.sgd(
            learning_rate=lr_fn,
            momentum=None if momentum is None else float(momentum),
            accumulator_dtype=mu_dtype,
        )
    raise ValueError(f"Unsupported pMF optimizer: {optimizer!r}")


def create_pmf_eval_state(rng, config, model):
    rng, rng_init = random.split(rng)
    _, params = initialized(rng_init, config, model)
    return EvalState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        ema_params=None,
    )


def train_step(state, batch, rng_init, ema_fn, lr_fn, use_ema, grad_accum_steps, aux_fn=None):
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    images = batch["image"]
    labels = batch["label"]

    def loss_fn(params):
        return state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            aux_fn=aux_fn,
            rngs=dict(gen=rng_base),
        )

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    lr_value = lr_fn(state.step)
    metrics = compute_metrics(aux[1])
    metrics["lr"] = lr_value

    if grad_accum_steps <= 1:
        new_state = state.apply_gradients(grads=grads)
        if use_ema:
            ema_value = ema_fn(state.step)
            new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
            new_state = new_state.replace(ema_params=new_ema)
        metrics["did_update"] = jnp.array(1.0, dtype=jnp.float32)
        return new_state, metrics

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
    rng_sample = random.fold_in(rng_init, lax.axis_index(axis_name="batch"))
    rng_sample = random.fold_in(rng_sample, sample_idx)
    return generate(
        variable,
        model,
        rng_sample,
        int(device_batch_size),
        config,
        int(num_steps),
        omega,
        t_min,
        t_max,
        sample_idx=sample_idx,
    )


_PMF_MODEL_KEYS = {
    "model_str",
    "dtype",
    "num_classes",
    "P_mean",
    "P_std",
    "cfg_max",
    "noise_scale",
    "data_proportion",
    "cfg_beta",
    "class_dropout_prob",
    "norm_p",
    "norm_eps",
    "lpips",
    "lpips_lambda",
    "convnext",
    "convnext_lambda",
    "perceptual_max_t",
    "tr_uniform",
}


def _build_plain_pmf(config, *, eval_mode=False):
    model_config = {
        key: value
        for key, value in config.model.to_dict().items()
        if key in _PMF_MODEL_KEYS
    }
    return pixelMeanFlow(**model_config, eval=eval_mode)


def _set_pmf_num_classes(config, num_classes):
    config.dataset.num_classes = num_classes
    config.model.num_classes = num_classes
    config.sampling.num_classes = num_classes


def _set_num_classes_from_data(config):
    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_images(config.dataset.root)
        _set_pmf_num_classes(config, inferred_num_classes)
        log_for_0("Inferred dataset.num_classes from image data: %s", inferred_num_classes)


def _shape_matches(target_value, source_value):
    return (
        hasattr(target_value, "shape")
        and hasattr(source_value, "shape")
        and target_value.shape == source_value.shape
    )


def _select_source_params(restored, ema_key):
    ema_params = restored.get("ema_params") if isinstance(restored, dict) else None
    if isinstance(ema_params, dict):
        for key in (str(ema_key), str(int(float(ema_key))), ema_key):
            if key in ema_params:
                log_for_0("Using source pMF ema_params[%r].", key)
                return ema_params[key]
        first_key = next(iter(ema_params))
        log_for_0("Using source pMF ema_params[%r].", first_key)
        return ema_params[first_key]
    if isinstance(restored, dict) and "params" in restored:
        log_for_0("Using source pMF params.")
        return restored["params"]
    raise ValueError("Could not find params/ema_params in pMF checkpoint.")


def _restore_pmf_source_params(config, eval_mode=False):
    source_config = deepcopy(config)
    _set_pmf_num_classes(source_config, 1000)
    source_model = _build_plain_pmf(source_config, eval_mode=eval_mode)
    source_state = create_pmf_eval_state(
        random.PRNGKey(int(config.training.seed)),
        source_config,
        source_model,
    )
    target = {
        "step": source_state.step,
        "params": source_state.params,
        "ema_params": {
            "500": source_state.params,
            "1000": source_state.params,
            "2000": source_state.params,
        },
    }
    restored = checkpoints.restore_checkpoint(config.load_from, target=target)
    return _select_source_params(restored, config.get("load_ema", "1000"))


def _merge_shape_compatible_params(target_params, source_params):
    target_state = serialization.to_state_dict(target_params)
    source_state = serialization.to_state_dict(source_params)
    loaded_count = 0
    skipped_count = 0
    skipped_examples = []

    def merge_state(target_subtree, source_subtree, key_path=()):
        nonlocal loaded_count, skipped_count, skipped_examples
        if isinstance(target_subtree, dict):
            merged = {}
            source_subtree = source_subtree if isinstance(source_subtree, dict) else {}
            for key, target_value in target_subtree.items():
                merged[key] = merge_state(
                    target_value,
                    source_subtree.get(key),
                    key_path + (key,),
                )
            return merged

        if source_subtree is not None and _shape_matches(target_subtree, source_subtree):
            loaded_count += 1
            return source_subtree

        skipped_count += 1
        if len(skipped_examples) < 10:
            skipped_examples.append("/".join(key_path))
        return target_subtree

    merged_state = merge_state(target_state, source_state)
    merged_params = serialization.from_state_dict(target_params, merged_state)
    log_for_0(
        "Partially restored pMF checkpoint: loaded %d tensors, skipped %d tensors.",
        loaded_count,
        skipped_count,
    )
    if skipped_examples:
        log_for_0("Skipped tensor examples: %s", ", ".join(skipped_examples))
    return merged_params


def _load_initial_state(state, config):
    if not config.load_from:
        return state
    if config.get("partial_load", False):
        source_params = _restore_pmf_source_params(config, eval_mode=False)
        merged_params = _merge_shape_compatible_params(state.params, source_params)
        ema_params = None
        if state.ema_params is not None:
            ema_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), merged_params)
        return state.replace(params=merged_params, ema_params=ema_params)
    restored = checkpoints.restore_checkpoint(config.load_from, target=state)
    log_for_0("Restored pMF checkpoint from %s.", config.load_from)
    return restored


def _restore_eval_state(config, model, use_ema):
    state = create_pmf_eval_state(
        random.PRNGKey(int(config.training.seed)),
        config,
        model,
    )
    target = {
        "step": state.step,
        "params": state.params,
        "ema_params": state.ema_params,
        "grad_accum": None,
        "grad_accum_step": None,
        "opt_state": None,
    }
    restored = checkpoints.restore_checkpoint(config.load_from, target=target)
    params = restored.get("params")
    ema_params = restored.get("ema_params")
    if params is None and ema_params is None:
        raise ValueError(f"No params/ema_params found in checkpoint: {config.load_from}")
    if params is None:
        params = ema_params
    if use_ema and ema_params is None:
        ema_params = params
    if not use_ema:
        ema_params = None
    return EvalState(
        step=jnp.asarray(restored.get("step", 0), dtype=jnp.int32),
        params=params,
        ema_params=ema_params,
    )


def _parse_int_steps(value, *, fallback=()):
    if value is None or value == "":
        return tuple(fallback)
    if isinstance(value, str):
        return tuple(int(step) for step in value.replace(",", " ").split())
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(step) for step in value)


def _get_metric_num_steps(config):
    forced_steps = str(config.training.get("force_metric_num_steps", "") or "").strip()
    if forced_steps:
        return _parse_int_steps(forced_steps)
    configured_steps = config.training.get("metric_num_steps", ())
    if configured_steps:
        num_steps = list(_parse_int_steps(configured_steps))
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


def _should_run_fid(current_step, training_config):
    force_fid_per_step = int(training_config.get("force_fid_per_step", 0) or 0)
    if force_fid_per_step > 0:
        return current_step % force_fid_per_step == 0
    fid_schedule = training_config.get("fid_schedule", [])
    if fid_schedule:
        for schedule_item in fid_schedule:
            from_step = int(schedule_item.get("from_step", 0))
            until_step = schedule_item.get("until_step", None)
            if current_step < from_step:
                continue
            if until_step is not None and current_step >= int(until_step):
                continue
            every_steps = int(schedule_item["every_steps"])
            return every_steps > 0 and current_step % every_steps == 0
        return False
    fid_per_step = int(training_config.get("fid_per_step", 0))
    return fid_per_step > 0 and current_step % fid_per_step == 0


def _metrics_enabled(training_config):
    if training_config.get("fid_schedule", []):
        return True
    return int(training_config.get("fid_per_step", 0)) > 0


def _primary_metric_mode(use_ema):
    return "ema" if use_ema else "online"


def _write_eval_metrics_csv(workdir, **row):
    append_eval_metrics_row(workdir, row)


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> EvalState:
    assert config.eval_only
    assert config.load_from != ""
    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)
    use_ema = config.training.get("use_ema", False)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    model = _build_plain_pmf(config, eval_mode=True)
    state = _restore_eval_state(config, model, use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=sample_device_bsz,
            num_steps=int(config.sampling.num_steps),
        ),
        axis_name="batch",
        devices=sample_devices,
    )
    pixel_manager = PixelImageManager(
        sample_device_bsz,
        decode_num_local_devices=sample_local_device_count,
    )
    sample_kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.get("omega", 1.0)),
            "t_min": float(config.sampling.get("t_min", 0.0)),
            "t_max": float(config.sampling.get("t_max", 1.0)),
        },
        devices=sample_devices,
    )
    num_preview_images = int(config.fid.get("num_images_to_log", 16))
    preview_grid_size = int(num_preview_images ** 0.5)
    num_preview_images = preview_grid_size ** 2
    preview = generate_preview_samples_first_device(
        state,
        p_sample_step,
        pixel_manager,
        use_ema,
        num_samples=num_preview_images,
        param_dtype=get_sampling_param_dtype(config),
        sample_local_device_count=sample_local_device_count,
        **sample_kwargs,
    )
    preview_grid = make_side_by_side_preview_panel(
        {int(config.sampling.num_steps): preview},
        preview_grid_size,
    )
    writer.write_images(step, {"image_grid": preview_grid})

    evaluator = get_image_metric_evaluator(config, writer, pixel_manager)
    result = evaluator(state, p_sample_step, step, not use_ema, **sample_kwargs)
    _write_eval_metrics_csv(
        workdir,
        eval_phase="eval_only",
        metric_mode=_primary_metric_mode(use_ema),
        training_step=step,
        sampling_num_steps=int(config.sampling.num_steps),
        omega=float(config.sampling.get("omega", 1.0)),
        t_min=float(config.sampling.get("t_min", 0.0)),
        t_max=float(config.sampling.get("t_max", 1.0)),
        fid=float(result["fid"]),
        inception_score=float(result["is"]),
        fd_dino="" if result.get("fd_dino") is None else float(result["fd_dino"]),
        is_best_fid=1,
        is_best_fd_dino=1 if result.get("fd_dino") is not None else 0,
        checkpoint_path=os.path.abspath(config.load_from),
    )
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    writer = Writer(config, workdir)
    if config.eval_only:
        raise ValueError("Use just_evaluate() for eval_only pMF runs.")
    _set_num_classes_from_data(config)

    rng = random.key(config.training.seed)
    use_ema = config.training.get("use_ema", False)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = int(config.training.get("grad_accum_steps", 1))
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)

    log_for_0("config.training.batch_size: %s", config.training.batch_size)
    log_for_0("config.training.use_ema: %s", use_ema)
    log_for_0("config.training.max_train_steps: %s", max_train_steps)
    log_for_0("config.training.grad_accum_steps: %s", grad_accum_steps)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)

    local_batch_size = int(config.training.batch_size) // jax.process_count()
    if local_batch_size < jax.local_device_count():
        raise ValueError(
            "config.training.batch_size is too small for pmap: "
            f"global batch {config.training.batch_size}, process_count "
            f"{jax.process_count()}, local batch {local_batch_size}, local devices "
            f"{jax.local_device_count()}."
        )
    if local_batch_size % jax.local_device_count() != 0:
        raise ValueError(
            "config.training.batch_size must make the per-host batch divisible by "
            f"local devices: global batch {config.training.batch_size}, "
            f"process_count {jax.process_count()}, local batch {local_batch_size}, "
            f"local devices {jax.local_device_count()}."
        )
    train_loader, steps_per_epoch = input_pipeline.create_image_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: %s", steps_per_epoch)

    model = _build_plain_pmf(config, eval_mode=False)
    if config.model.get("convnext", False) or config.model.get("lpips", False):
        log_for_0("Using pMF perceptual auxiliary loss.")
        aux_fn = init_auxloss(config)
    else:
        log_for_0("Not using pMF perceptual auxiliary loss.")
        aux_fn = None
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_pmf_train_state(rng, config, model, lr_fn)
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
            aux_fn=aux_fn,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    sample_model = _build_plain_pmf(config, eval_mode=True)

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

    pixel_manager = PixelImageManager(
        sample_device_bsz,
        decode_num_local_devices=sample_local_device_count,
    )
    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in metric_num_steps
    }
    log_for_0("Metric evaluation sampling steps: %s", metric_num_steps)

    preview_num_steps = _parse_int_steps(
        config.training.get("preview_num_steps", ()),
        fallback=(int(config.sampling.num_steps),),
    )
    p_preview_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in preview_num_steps
    }
    preview_guidance_scales = config.training.get("preview_guidance_scales", [])
    preview_guidance_scales = (
        [float(omega) for omega in preview_guidance_scales]
        if preview_guidance_scales
        else [float(config.sampling.get("omega", 1.0))]
    )

    def log_preview_samples(state_for_logging, step_for_logging):
        num_images = int(config.fid.num_images_to_log)
        grid_size = int(num_images ** 0.5)
        num_images = grid_size ** 2
        if num_images <= 0:
            return
        preview_image_groups = {}
        for cfg_scale in preview_guidance_scales:
            preview_kwargs = jax_utils.replicate(
                {
                    "omega": cfg_scale,
                    "t_min": float(config.sampling.get("t_min", 0.0)),
                    "t_max": float(config.sampling.get("t_max", 1.0)),
                },
                devices=sample_devices,
            )
            preview_images = {}
            for num_steps, p_preview_step in p_preview_sample_steps.items():
                preview_images[num_steps] = generate_preview_samples_first_device(
                    state_for_logging,
                    p_preview_step,
                    pixel_manager,
                    use_ema,
                    num_samples=num_images,
                    param_dtype=get_sampling_param_dtype(config),
                    sample_local_device_count=sample_local_device_count,
                    **preview_kwargs,
                )
            preview_panel = make_side_by_side_preview_panel(preview_images, grid_size)
            preview_image_groups[
                format_preview_guidance_label(
                    cfg_scale,
                    float(config.sampling.get("t_min", 0.0)),
                    float(config.sampling.get("t_max", 1.0)),
                )
            ] = [preview_panel]
        writer.write_images(
            step_for_logging,
            {"image_grid": make_stacked_grid_panel(preview_image_groups, 1)},
        )

    image_metric_evaluator = (
        get_image_metric_evaluator(config, writer, pixel_manager)
        if _metrics_enabled(config.training)
        else None
    )
    best_fid_by_steps = {num_steps: float("inf") for num_steps in metric_num_steps}
    best_fd_dino_by_steps = {num_steps: float("inf") for num_steps in metric_num_steps}
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
    if (
        initial_step == 0
        and config.training.sample_per_step > 0
        and config.training.get("preview_at_step0", False)
    ):
        log_preview_samples(state, 0)

    sample_kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.get("omega", 1.0)),
            "t_min": float(config.sampling.get("t_min", 0.0)),
            "t_max": float(config.sampling.get("t_max", 1.0)),
        },
        devices=sample_devices,
    )

    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch %s...", epoch)
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            batch = input_pipeline.prepare_batch_data(batch, batch_size=local_batch_size)
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

            if (
                image_metric_evaluator is not None
                and did_update
                and current_step > 0
                and _should_run_fid(current_step, config.training)
            ):
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
                        omega=float(config.sampling.get("omega", 1.0)),
                        t_min=float(config.sampling.get("t_min", 0.0)),
                        t_max=float(config.sampling.get("t_max", 1.0)),
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
