"""Generator-initialised scalar discriminator D(x_s, s, y) for a plain Flax SiT."""

import jax.numpy as jnp

from models.imfDiT import FlaxSiT
from models.torch_models import RMSNorm, TorchLinear


class SiTFTDiscriminator(FlaxSiT):
    """FlaxSiT backbone (same module names, so G tensors copy over) with a scalar head.

    The final image layer is never called, so it gets no parameters. Tokens are mean-pooled
    (FlaxSiT has no class token) and mapped to one logit by a zero-initialised head.
    """

    def setup(self):
        super().setup()
        self.dis_norm = RMSNorm(self.hidden_size)
        self.dis_head = TorchLinear(
            self.hidden_size, 1, bias=True, weight_init="zeros", bias_init="zeros"
        )

    def __call__(self, x, point_time, y=None):
        batch_size = x.shape[0]
        point_time = jnp.asarray(point_time, dtype=x.dtype).reshape(batch_size)
        if y is None:
            y = jnp.zeros((batch_size,), dtype=jnp.int32)
        y = jnp.asarray(y, dtype=jnp.int32).reshape(batch_size)
        h = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(point_time).astype(h.dtype) + self.y_embedder(y).astype(h.dtype)
        for block in self.blocks:
            h = block(h, c)
        pooled = jnp.mean(h, axis=1)
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_sit_ft_discriminator(generator_net):
    """Discriminator with the exact architecture of the FlaxSiT generator net."""
    if not isinstance(generator_net, FlaxSiT):
        raise ValueError("SiT-FT AFM needs a FlaxSiT (flaxSiT_*) generator backbone.")
    return SiTFTDiscriminator(
        input_size=generator_net.input_size,
        patch_size=generator_net.patch_size,
        in_channels=generator_net.in_channels,
        hidden_size=generator_net.hidden_size,
        depth=generator_net.depth,
        num_heads=generator_net.num_heads,
        mlp_ratio=generator_net.mlp_ratio,
        num_classes=generator_net.num_classes,
        use_null_class=generator_net.use_null_class,
        learn_sigma=generator_net.learn_sigma,
        eval=False,
    )
