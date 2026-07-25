"""Original CAFM JVP post-training from a target-finetuned iMF checkpoint.

The mature checkpoint/evaluation loop is shared with ``train_caimf``, but this
module installs separate model, sample, loss, and discriminator-JVP functions.
No finite-interval CA-iMF function is used by the resulting training steps.
"""

import dataclasses

import jax
import jax.numpy as jnp

import train_caimf as _harness
from cafm_imf import (
    discriminator_jvp_logits,
    discriminator_loss,
    generator_jvp_logit,
    generator_loss,
)
from imf import iMeanFlow
from models import imfDiT
from models.cafm_imf_discriminator import create_cafm_imf_discriminator
from utils.logging_util import log_for_0

_BASE_CREATE_STATE = _harness._create_state


class CAFMImprovedMeanFlow(iMeanFlow):
    """Expose iMF's instantaneous velocity as an original CAFM generator."""

    def _cafm_samples(self, images, labels):
        if not self._uses_plain_imf_dit_backbone():
            raise ValueError("CAFM-on-iMF requires a plain imfDiT backbone.")
        if not self.use_auxiliary_v_head:
            raise ValueError(
                "CAFM-on-iMF requires the trained auxiliary instantaneous "
                "velocity head (model.use_auxiliary_v_head=True)."
            )

        x_0 = images.astype(self.dtype)
        batch_size = x_0.shape[0]
        t = jax.random.uniform(
            self.make_rng("gen"),
            (batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        noise = jax.random.normal(
            self.make_rng("gen"), x_0.shape, dtype=self.dtype
        )
        x_t = (1.0 - t) * x_0 + t * noise
        velocity_real = noise - x_0

        # Match the official CAFM class-dropout setup. Unit guidance and a
        # fixed full CFG interval make the velocity a function of (x_t,t,y).
        labels, _ = self.cond_drop(velocity_real, velocity_real, labels)
        velocity_fake = self.v_cond_fn(
            x_t,
            t,
            jnp.ones_like(t),
            labels,
        )
        return {
            "x_t": x_t,
            "t": t.reshape(batch_size),
            "labels": labels,
            "velocity_real": jax.lax.stop_gradient(velocity_real),
            "velocity_fake": velocity_fake,
            # Compatibility key for the shared OT-aware generator harness.
            "u": velocity_fake,
        }

    def sample_cafm_one_step(
        self, x_t, labels, step_index, time_steps, omega, t_min, t_max
    ):
        """Euler step using the instantaneous velocity trained by CAFM."""
        t = jnp.take(time_steps, step_index)
        r = jnp.take(time_steps, step_index + 1)
        batch_size = x_t.shape[0]
        t_batch = jnp.broadcast_to(t, (batch_size,))
        r_batch = jnp.broadcast_to(r, (batch_size,))
        omega_batch = jnp.broadcast_to(omega, (batch_size,))
        t_min_batch = jnp.broadcast_to(t_min, (batch_size,))
        t_max_batch = jnp.broadcast_to(t_max, (batch_size,))

        velocity_cond, velocity_uncond = self.v_fn(x_t, t_batch, labels)
        effective_omega = jnp.where(
            (t_batch >= t_min_batch) & (t_batch <= t_max_batch),
            omega_batch,
            jnp.ones_like(omega_batch),
        ).reshape((batch_size, 1, 1, 1))
        guided_rgb = velocity_uncond[..., :3] + effective_omega * (
            velocity_cond[..., :3] - velocity_uncond[..., :3]
        )
        velocity = jnp.concatenate([guided_rgb, velocity_cond[..., 3:]], axis=-1)
        return x_t + (r_batch - t_batch).reshape(
            (batch_size, 1, 1, 1)
        ) * velocity

    def forward_caimf_discriminator_samples(
        self,
        images,
        labels,
        current_step=None,
        interval_eps=0.0,
    ):
        del current_step, interval_eps
        return self._cafm_samples(images, labels)

    def forward_caimf_generator_terms(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
        interval_eps=0.0,
    ):
        del source_params, teacher_params, current_step, interval_eps
        terms = self._cafm_samples(images, labels)
        zero = jnp.asarray(0.0, dtype=terms["velocity_fake"].dtype)
        terms.update(loss_imf=zero, loss_u=zero, loss_v=zero)
        return terms


def _create_models(config):
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model = CAFMImprovedMeanFlow(
        **{key: value for key, value in model_config.items() if key in valid_keys}
    )
    if not str(config.model.model_str).startswith("imfDiT_"):
        raise ValueError("CAFM-on-iMF currently supports model_str=imfDiT_*.")
    net_fn = getattr(imfDiT, config.model.model_str)
    generator_net = net_fn(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        use_auxiliary_v_head=bool(config.model.use_auxiliary_v_head),
        eval=False,
    )
    return model, create_cafm_imf_discriminator(generator_net)


def _create_sampling_model(config):
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model_config = {
        key: value for key, value in model_config.items() if key in valid_keys
    }
    # CAFM generates from the full trained auxiliary-v branch. The base iMF
    # eval mode intentionally removes v refinement blocks because ordinary iMF
    # sampling uses u; that optimization is invalid for this CAFM sampler.
    model_config["eval"] = False
    return CAFMImprovedMeanFlow(**model_config)


def _sample_step(
    variable,
    sample_idx,
    *,
    model,
    rng_init,
    device_batch_size,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
):
    """Generate with CAFM's instantaneous velocity rather than average u."""
    rng = jax.random.fold_in(rng_init, sample_idx)
    rng, rng_noise = jax.random.split(rng)
    shape = (
        device_batch_size,
        int(config.dataset.image_size),
        int(config.dataset.image_size),
        int(config.dataset.image_channels),
    )
    x_t = jax.random.normal(rng_noise, shape, dtype=model.dtype)
    labels = (
        jnp.arange(device_batch_size, dtype=jnp.int32)
        + sample_idx * device_batch_size
    ) % int(config.dataset.num_classes)
    time_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

    def step_fn(index, value):
        return model.apply(
            variable,
            value,
            labels,
            index,
            time_steps,
            omega,
            t_min,
            t_max,
            method=model.sample_cafm_one_step,
        )

    images = jax.lax.fori_loop(0, num_steps, step_fn, x_t)
    return images.transpose(0, 3, 1, 2)


def _adapt_discriminator_init_for_point_time():
    """Make the shared state loader initialize D(x,t,y), not interval D."""
    def create_state(config, model, discriminator, rng):
        original_init = discriminator.init

        class InitAdapter:
            def init(self, rngs, x, point_time, interval_r, interval_t, labels):
                del interval_r, interval_t
                return original_init(rngs, x, point_time, labels)

        # The harness only requires ``init`` here; train steps retain the real D.
        return _BASE_CREATE_STATE(config, model, InitAdapter(), rng)

    return create_state


def train_and_evaluate(config, workdir):
    """Run original CAFM while reusing CA-iMF's tested I/O/evaluation harness."""
    if float(config.caimf.lambda_imf) != 0.0:
        raise ValueError(
            "Original CAFM has no iMF regression loss; set caimf.lambda_imf=0."
        )
    if not bool(config.model.use_auxiliary_v_head):
        raise ValueError("Original CAFM requires iMF's auxiliary v head.")

    log_for_0(
        "ORIGINAL CAFM-on-iMF: JVP discriminator at (x_t,t), "
        "instantaneous auxiliary-v generator; no finite intervals."
    )

    # These names are resolved dynamically by the already-tested train steps.
    _harness._create_models = _create_models
    _harness._create_state = _adapt_discriminator_init_for_point_time()
    _harness._create_sampling_model = _create_sampling_model
    _harness._sample_step = _sample_step
    _harness.finite_interval_logits = discriminator_jvp_logits
    _harness.finite_fake_logit = generator_jvp_logit
    _harness.discriminator_loss = discriminator_loss
    _harness.generator_loss = generator_loss
    return _harness.train_and_evaluate(config, workdir)
