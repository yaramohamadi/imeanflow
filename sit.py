"""Plain SiT wrapper for official transport-style training."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import imfDiT
from utils.sit_transport_jax import create_transport


class PlainSiT(nn.Module):
    """Dedicated plain SiT training wrapper around the exact Flax SiT backbone."""

    model_str: str
    dtype: jnp.dtype = jnp.float32
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    path_type: str = "Linear"
    prediction: str = "velocity"
    loss_weight: str = None
    train_eps: float = None
    sample_eps: float = None
    objective: str = "sit"
    path_power_k: float = 1.0
    P_mean: float = -0.4
    P_std: float = 1.0
    data_proportion: float = 0.5
    output_prediction_space: str = "velocity"
    wrapper_eps: float = 1e-6
    eval: bool = False

    def setup(self):
        if not (
            self.model_str.startswith("flaxSiT")
            or self.model_str.startswith("flaxDiT")
        ):
            raise ValueError(
                "PlainSiT expects a flaxSiT_* or flaxDiT_* backbone, got "
                f"{self.model_str!r}."
            )

        net_fn = getattr(imfDiT, self.model_str)
        self.net = net_fn(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            use_r_conditioning=(self.objective == "power_meanflow"),
            eval=self.eval,
        )
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
        )
        if self.objective not in {"sit", "power_meanflow"}:
            raise ValueError(
                "PlainSiT objective must be one of ['sit', 'power_meanflow'], got "
                f"{self.objective!r}."
            )

    def _validate_output_prediction_space(self):
        if self.output_prediction_space not in {"velocity", "data", "noise"}:
            raise ValueError(
                "PlainSiT output_prediction_space must be one of "
                "['velocity', 'data', 'noise'], got "
                f"{self.output_prediction_space!r}."
            )
        if self.output_prediction_space != "velocity" and self.objective != "sit":
            raise ValueError(
                "PlainSiT non-velocity output wrappers are only supported for "
                f"objective='sit', got objective={self.objective!r}."
            )

    def _compute_wrapped_velocity(self, raw_output, xt, t):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "velocity":
            return raw_output

        t_expanded = t.reshape((t.shape[0],) + (1,) * (xt.ndim - 1))
        alpha_t, d_alpha_t = self.transport.path_sampler.compute_alpha_t(t_expanded)
        sigma_t, d_sigma_t = self.transport.path_sampler.compute_sigma_t(t_expanded)
        sigma_safe = jnp.where(
            jnp.abs(sigma_t) > self.wrapper_eps,
            sigma_t,
            jnp.where(sigma_t >= 0.0, self.wrapper_eps, -self.wrapper_eps),
        )

        if self.output_prediction_space == "data":
            x1_hat = raw_output
            x0_hat = (xt - alpha_t * x1_hat) / sigma_safe
        else:
            x0_hat = raw_output
            x1_hat = (xt - sigma_t * x0_hat) / jnp.maximum(alpha_t, self.wrapper_eps)

        return d_alpha_t * x1_hat + d_sigma_t * x0_hat

    def _predict_transport_output(self, x, t, y, r=None):
        raw_output = self.net(
            x.astype(self.dtype),
            t.astype(self.dtype),
            y,
            r=None if r is None else r.astype(self.dtype),
        )
        return self._compute_wrapped_velocity(raw_output, x.astype(self.dtype), t.astype(self.dtype))

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)
        return t, r, fm_mask

    def _drop_labels(self, labels, rng):
        if (
            (not self.target_use_null_class)
            or self.class_dropout_prob <= 0.0
            or self.eval
        ):
            return labels

        drop_mask = jax.random.uniform(rng, labels.shape, dtype=jnp.float32)
        drop_mask = drop_mask < self.class_dropout_prob
        null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
        return jnp.where(drop_mask, null_labels, labels)

    def forward(self, images, labels):
        """Compute the official SiT transport loss."""
        if self.objective == "power_meanflow":
            return self.forward_power_meanflow(images, labels)

        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_loss = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)

        def model_fn(xt, t, y):
            return self._predict_transport_output(xt, t, y)

        terms = self.transport.training_losses(
            model_fn,
            x,
            rng=rng_loss,
            model_kwargs={"y": labels},
        )
        loss = jnp.mean(terms["loss"])
        dict_losses = {
            "loss": loss,
            "loss_transport": jnp.mean(terms["loss"]),
            "t_mean": jnp.mean(terms["t"]),
        }
        return loss, dict_losses

    def forward_power_meanflow(self, images, labels):
        """Compute the experimental power-geometry mean-flow loss."""
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)
        bz = x.shape[0]

        rng_drop, rng_eps = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)
        eps = jax.random.normal(rng_eps, x.shape, dtype=self.dtype)

        t, r, fm_mask = self.sample_tr(bz)
        t_scalar = t.reshape((bz,))
        r_scalar = r.reshape((bz,))
        one_minus_t = jnp.clip(1.0 - t, 1e-6, 1.0)
        one_minus_r = jnp.clip(1.0 - r, 1e-6, 1.0)
        t_clamped = jnp.clip(t, 1e-6, 1.0)
        r_clamped = jnp.clip(r, 1e-6, 1.0)
        k = jnp.asarray(self.path_power_k, dtype=self.dtype)

        a_t = one_minus_t**k
        b_t = t_clamped**k
        z_t = a_t * x + b_t * eps

        inst_target = (
            -k * (one_minus_t ** (k - 1.0)) * x
            + k * (t_clamped ** (k - 1.0)) * eps
        )

        a_r = one_minus_r**k
        b_r = r_clamped**k
        z_r = a_r * x + b_r * eps
        denom = jnp.maximum(jnp.abs(t - r), 1e-6)
        mf_target = (z_t - z_r) / denom
        target = jnp.where(fm_mask, inst_target, mf_target)

        pred = self.net(
            z_t.astype(self.dtype),
            t_scalar.astype(self.dtype),
            labels,
            r=r_scalar.astype(self.dtype),
        )
        sq_error = (pred - target.astype(pred.dtype)) ** 2
        per_example_loss = jnp.mean(sq_error.reshape((bz, -1)), axis=1)
        inst_mask = fm_mask.reshape((bz,)).astype(self.dtype)
        mf_mask = 1.0 - inst_mask
        inst_denom = jnp.maximum(jnp.sum(inst_mask), 1.0)
        mf_denom = jnp.maximum(jnp.sum(mf_mask), 1.0)
        inst_loss = jnp.sum(per_example_loss * inst_mask) / inst_denom
        mf_loss = jnp.sum(per_example_loss * mf_mask) / mf_denom
        loss = jnp.mean(per_example_loss)

        dict_losses = {
            "loss": loss,
            "loss_power_meanflow": loss,
            "loss_instantaneous": inst_loss,
            "loss_meanflow": mf_loss,
            "diag_fraction": jnp.mean(inst_mask),
            "t_mean": jnp.mean(t_scalar),
            "r_mean": jnp.mean(r_scalar),
            "interval_mean": jnp.mean(jnp.abs(t_scalar - r_scalar)),
        }
        return loss, dict_losses

    def __call__(self, x, t, y, r=None):
        """Initialization-only forward that mirrors the exact SiT backbone."""
        return self._predict_transport_output(x, t, y, r=r)
