"""Plain DiT wrapper using the original diffusion objective."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import imfDiT
from utils.dit_diffusion import create_diffusion


class PlainDiT(nn.Module):
    """JAX/Flax DiT backbone trained with the original DiT DDPM objective."""

    model_str: str
    dtype: jnp.dtype = jnp.float32
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    learn_sigma: bool = True
    predict_xstart: bool = False
    rescale_learned_sigmas: bool = False
    eval: bool = False

    def setup(self):
        if not self.model_str.startswith("flaxDiT"):
            raise ValueError(f"PlainDiT expects a flaxDiT_* backbone, got {self.model_str!r}.")

        net_fn = getattr(imfDiT, self.model_str)
        self.net = net_fn(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            learn_sigma=self.learn_sigma,
            return_learned_sigma=True,
            eval=self.eval,
        )
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=self.noise_schedule,
            learn_sigma=self.learn_sigma,
            predict_xstart=self.predict_xstart,
            rescale_learned_sigmas=self.rescale_learned_sigmas,
            diffusion_steps=self.diffusion_steps,
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
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_t, rng_noise = jax.random.split(self.make_rng("gen"), 3)
        labels = self._drop_labels(labels, rng_drop)
        t = jax.random.randint(
            rng_t,
            (x.shape[0],),
            minval=0,
            maxval=self.diffusion.num_timesteps,
            dtype=jnp.int32,
        )

        def model_fn(xt, timesteps):
            return self.net(xt.astype(self.dtype), timesteps.astype(jnp.float32), labels)

        terms = self.diffusion.training_losses(model_fn, x, t, rng_noise)
        loss = jnp.mean(terms["loss"])
        dict_losses = {
            "loss": loss,
            "loss_diffusion": loss,
            "mse": jnp.mean(terms["mse"]),
            "vb": jnp.mean(terms["vb"]),
            "t_mean": jnp.mean(terms["t"].astype(jnp.float32)),
        }
        return loss, dict_losses

    def __call__(self, x, t, y):
        return self.net(x.astype(self.dtype), t.astype(jnp.float32), y)
