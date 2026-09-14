"""Correctness checks for the GT-anchored on-policy loss.

Three checks, none of which needs the real 675M-param checkpoint:

1. ``_velocity_to_data`` inverts the interpolant: feeding it the exact target
   velocity ``ut`` at ``xt`` recovers ``x1``.
2. The corrective target degenerates to the standard FM target when the model's
   data prediction is perfect: with ``x1_hat = x1``,
   ``(x1 - x_hat_t') / (1 - t') == x1 - eps'`` exactly. This is the statement
   that the new loss is the old loss plus a reconstruction-error correction.
3. ``forward_gt_on_policy`` runs end to end on a small backbone, its loss is
   finite, its gradients are finite, and at ``lambda = 1`` it reproduces
   ``forward``'s FM loss.

Run with:
    JAX_PLATFORMS=cpu python scripts/check_gt_on_policy_target.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from models import imfDiT
from models.imfDiT import FlaxSiT
from sit import PlainSiT

IMAGE_SIZE = 16

# A tiny backbone so the checks run on CPU in seconds.
imfDiT.flaxSiT_TINY_8 = partial(
    FlaxSiT,
    input_size=IMAGE_SIZE,
    depth=2,
    hidden_size=64,
    patch_size=8,
    num_heads=4,
    learn_sigma=True,
)

CHANNELS = 4
BATCH = 8
NUM_CLASSES = 7


def make_model(**overrides):
    kwargs = dict(
        model_str="flaxSiT_TINY_8",
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.0,
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        output_prediction_space="velocity",
        sit_wrapper_eps=1e-3,
    )
    kwargs.pop("sit_wrapper_eps")
    kwargs["wrapper_eps"] = 1e-3
    kwargs.update(overrides)
    return PlainSiT(**kwargs)


def init(model, rng):
    x = jnp.zeros((BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), jnp.float32)
    t = jnp.zeros((BATCH,), jnp.float32)
    y = jnp.zeros((BATCH,), jnp.int32)
    return model.init({"params": rng, "gen": rng}, x, t, y)["params"]


def check_1_velocity_to_data():
    model = make_model()
    rng = jax.random.key(0)
    params = init(model, rng)

    k1, k2, k3 = jax.random.split(rng, 3)
    x1 = jax.random.normal(k1, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    x0 = jax.random.normal(k2, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    t = jax.random.uniform(k3, (BATCH,), minval=0.01, maxval=0.99)

    def body(module, x1, x0, t):
        _, xt, ut = module.transport.path_sampler.plan(t, x0, x1)
        return module._velocity_to_data(ut, xt, t)

    x1_rec = model.apply({"params": params}, x1, x0, t, method=body)
    err = float(jnp.max(jnp.abs(x1_rec - x1)))
    print(f"[1] _velocity_to_data inversion max abs err = {err:.3e}")
    assert err < 1e-4, err


def check_2_target_degenerates():
    """With a perfect data prediction the corrective target IS the FM target."""
    model = make_model()
    rng = jax.random.key(1)
    params = init(model, rng)

    k1, k2, k3 = jax.random.split(rng, 3)
    x1 = jax.random.normal(k1, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    eps = jax.random.normal(k2, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    # t' as the loss draws it: uniform on [0, 1] squeezed by (1 - delta).
    delta = 0.2
    t_prime = jax.random.uniform(k3, (BATCH,)) * (1.0 - delta)

    def body(module, x1, eps, t_prime):
        # x1_hat := x1 (a perfect stage-2 prediction)
        _, xt_hat, _ = module.transport.path_sampler.plan(t_prime, eps, x1)
        t_b = module._broadcast_scalar(t_prime, xt_hat)
        u_gt_on = (x1 - xt_hat) / (1.0 - t_b)
        return u_gt_on, xt_hat

    u_gt_on, xt_hat = model.apply(
        {"params": params}, x1, eps, t_prime, method=body
    )
    u_fm = x1 - eps
    err = float(jnp.max(jnp.abs(u_gt_on - u_fm)))
    print(f"[2] target vs standard FM target max abs err = {err:.3e}")
    assert err < 1e-4, err

    # And the divisor really is bounded: 1 - t' >= delta.
    min_divisor = float(jnp.min(1.0 - t_prime))
    print(f"[2] min divisor (1 - t') = {min_divisor:.4f} (delta = {delta})")
    assert min_divisor >= delta - 1e-6, min_divisor


def check_3_forward_runs():
    rng = jax.random.key(2)
    images = jax.random.normal(rng, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    labels = jnp.arange(BATCH) % NUM_CLASSES

    plain = make_model()
    params = init(plain, rng)

    def loss_of(model):
        def loss_fn(p):
            return model.apply(
                {"params": p}, images, labels, rngs=dict(gen=jax.random.key(3))
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        gnorm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )
        return float(loss), aux, float(gnorm)

    plain_loss, _, plain_gnorm = loss_of(plain)
    print(f"[3] plain forward: loss = {plain_loss:.6f}  |grad| = {plain_gnorm:.4f}")

    gton = make_model(gt_on_lambda=0.5, gt_on_t_delta=0.2)
    gton_loss, aux, gton_gnorm = loss_of(gton)
    print(f"[3] gt-on (lambda=0.5): loss = {gton_loss:.6f}  |grad| = {gton_gnorm:.4f}")
    for key in sorted(aux):
        print(f"      {key} = {float(aux[key]):.6f}")
    assert np.isfinite(gton_loss) and np.isfinite(gton_gnorm)

    # lambda = 1 must reproduce the plain FM loss exactly: same rng, same
    # transport call, corrective term weighted to zero.
    anchor = make_model(gt_on_lambda=1.0, gt_on_t_delta=0.2)
    anchor_loss, anchor_aux, _ = loss_of(anchor)
    err = abs(anchor_loss - plain_loss)
    print(f"[3] lambda=1 vs plain forward: |diff| = {err:.3e}")
    assert err < 1e-5, (anchor_loss, plain_loss)

    self_arm = make_model(gt_on_lambda=0.5, gt_on_target="self")
    self_loss, _, _ = loss_of(self_arm)
    print(f"[3] gt-on (target=self): loss = {self_loss:.6f}")
    assert np.isfinite(self_loss)


if __name__ == "__main__":
    check_1_velocity_to_data()
    check_2_target_degenerates()
    check_3_forward_runs()
    print("ALL CHECKS PASSED")
