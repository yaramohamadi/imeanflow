#!/usr/bin/env python3
import argparse
import json
import subprocess
import threading
import time

import jax
import jax.numpy as jnp

from imf import iMeanFlow


def tree_num_bytes(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(leaf.size * leaf.dtype.itemsize for leaf in leaves))


def mib(num_bytes):
    return float(num_bytes) / (1024.0 ** 2)


class GpuMemoryPoller:
    def __init__(self, interval_s=0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _query_used_mib(self):
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return max(int(line.strip()) for line in out.splitlines() if line.strip())

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._query_used_mib())
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples = []
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=1.0)
        return max(self.samples) if self.samples else None


def build_model(variant, num_classes, source_num_classes, use_dogfit, use_ema_vc):
    common = dict(
        num_classes=num_classes,
        cfg_beta=1.0,
        training_mode="imf_jvp",
        use_dogfit=use_dogfit,
        target_use_null_class=False,
        class_dropout_prob=0.0,
        source_prediction_space="v",
        source_num_classes=source_num_classes,
        use_ema_vc=use_ema_vc,
        guidance_scale_strategy="fixed",
        fixed_guidance_scale=1.5,
        training_guidance_interval_strategy="fixed",
        training_guidance_t_min=0.0,
        training_guidance_t_max=1.0,
        training_guidance_start_step=0,
    )

    if variant == "sit_dual":
        return iMeanFlow(
            model_str="imfSiT_XL_2",
            use_auxiliary_v_head=True,
            **common,
        )
    if variant == "sit_dmf":
        return iMeanFlow(
            model_str="imfSiT_DMF_XL_2",
            use_auxiliary_v_head=False,
            use_context_guidance_conditioning=False,
            use_adaln_guidance_scale_conditioning=False,
            adaln_guidance_scale_init="timestep",
            time_conditioning_mode="split",
            **common,
        )
    raise ValueError(f"Unsupported variant: {variant}")


def initialize_model(model, batch_size, image_size, num_classes, source_num_classes, seed):
    x = jnp.ones((batch_size, image_size, image_size, 4), dtype=jnp.float32)
    t = jnp.ones((batch_size,), dtype=jnp.int32)
    y = jnp.arange(batch_size, dtype=jnp.int32) % num_classes
    y_source = jnp.full((batch_size,), source_num_classes, dtype=jnp.int32)

    rng = jax.random.key(seed)
    rng, init_rng, source_rng = jax.random.split(rng, 3)
    params = model.init({"params": init_rng}, x, t, y)["params"]

    source_params = None
    if model.use_dogfit:
        source_params = model.init(
            {"params": source_rng},
            x,
            t,
            y_source,
            method=model.init_source,
        )["params"]["source_net"]

    return params, source_params


def compile_and_run(model, params, source_params, batch_size, image_size, num_classes, seed):
    x = jax.random.normal(
        jax.random.key(seed + 100),
        (batch_size, image_size, image_size, 4),
        dtype=jnp.float32,
    )
    y = jnp.arange(batch_size, dtype=jnp.int32) % num_classes
    teacher_params = params
    step = jnp.array(0, dtype=jnp.int32)

    def loss_fn(trainable_params, rng):
        loss, _ = model.apply(
            {"params": trainable_params},
            images=x,
            labels=y,
            source_params=source_params,
            teacher_params=teacher_params,
            current_step=step,
            rngs={"gen": rng},
            method=model.forward,
        )
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    warm_rng = jax.random.key(seed + 200)
    loss, grads = grad_fn(params, warm_rng)
    loss.block_until_ready()
    jax.tree_util.tree_map(lambda z: z.block_until_ready(), grads)

    poller = GpuMemoryPoller()
    poller.start()
    run_rng = jax.random.key(seed + 201)
    started = time.time()
    loss, grads = grad_fn(params, run_rng)
    loss.block_until_ready()
    jax.tree_util.tree_map(lambda z: z.block_until_ready(), grads)
    elapsed_s = time.time() - started
    peak_used_mib = poller.stop()

    grad_bytes = tree_num_bytes(grads)
    return {
        "loss": float(loss),
        "grad_mib": mib(grad_bytes),
        "elapsed_s": elapsed_s,
        "peak_gpu_used_mib": peak_used_mib,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["sit_dual", "sit_dmf"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--source-num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-dogfit", action="store_true")
    parser.add_argument("--use-ema-vc", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    model = build_model(
        args.variant,
        args.num_classes,
        args.source_num_classes,
        args.use_dogfit,
        args.use_ema_vc,
    )
    params, source_params = initialize_model(
        model,
        args.batch_size,
        args.image_size,
        args.num_classes,
        args.source_num_classes,
        args.seed,
    )

    params_bytes = tree_num_bytes(params)
    source_bytes = tree_num_bytes(source_params) if source_params is not None else 0

    persistent_training_bytes = (
        params_bytes
        + params_bytes  # ema_params
        + source_bytes  # frozen source for DogFit
        + 2 * params_bytes  # Adam m and v
        + params_bytes  # grad accumulation / gradient-sized buffer
    )

    result = {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "use_dogfit": args.use_dogfit,
        "use_ema_vc": args.use_ema_vc,
        "params_mib": mib(params_bytes),
        "ema_params_mib": mib(params_bytes),
        "source_params_mib": mib(source_bytes),
        "persistent_training_state_mib_estimate": mib(persistent_training_bytes),
    }
    if not args.init_only:
        runtime = compile_and_run(
            model,
            params,
            source_params,
            args.batch_size,
            args.image_size,
            args.num_classes,
            args.seed,
        )
        result.update(runtime)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
