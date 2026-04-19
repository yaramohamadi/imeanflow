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
            eval=self.eval,
        )
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
        )

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
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_loss = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)

        def model_fn(xt, t, y):
            return self.net(xt.astype(self.dtype), t.astype(self.dtype), y)

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

    def __call__(self, x, t, y):
        """Initialization-only forward that mirrors the exact SiT backbone."""
        return self.net(x.astype(self.dtype), t.astype(self.dtype), y)
