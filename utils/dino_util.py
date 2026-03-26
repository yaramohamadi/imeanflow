import time
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from absl import logging

from .logging_util import log_for_0


VALID_DINOV2_ARCHITECTURES = (
    "vits14",
    "vitb14",
    "vitl14",
    "vitg14",
)

DINOV2_MODEL_NAMES = {
    "vits14": "facebook/dinov2-small",
    "vitb14": "facebook/dinov2-base",
    "vitl14": "facebook/dinov2-large",
    "vitg14": "facebook/dinov2-giant",
}

DINO_IMAGE_SIZE = 224
DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def build_jax_dinov2(
    arch: str = "vitb14",
    model_name: Optional[str] = None,
    batch_size: int = 200,
    clean_resize: bool = False,
) -> Dict[str, object]:
    if arch not in VALID_DINOV2_ARCHITECTURES:
        raise ValueError(
            f"Unsupported DINOv2 architecture: {arch}. "
            f"Choose from {VALID_DINOV2_ARCHITECTURES}."
        )

    if clean_resize:
        raise NotImplementedError(
            "clean_resize=True is not implemented yet. "
            "The default dgm-eval path uses standard bicubic resize."
        )

    from transformers import FlaxDinov2Model

    resolved_model_name = model_name or DINOV2_MODEL_NAMES[arch]
    logging.info("Initializing Flax DINOv2 model '%s'", resolved_model_name)
    model = FlaxDinov2Model.from_pretrained(
        resolved_model_name,
        dtype=jnp.float32,
    )
    params = model.params

    def dinov2_apply(model_params, pixel_values):
        outputs = model(
            pixel_values=pixel_values,
            params=model_params,
            train=False,
        )
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    dino_fn = jax.jit(dinov2_apply)
    fake_x = jnp.zeros((batch_size, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), dtype=jnp.float32)
    logging.info("Start compiling DINOv2 function...")
    t_start = time.time()
    _ = dino_fn(params, fake_x)
    logging.info("End compiling: %.4f seconds.", time.time() - t_start)

    return {
        "params": params,
        "fn": dino_fn,
        "model": model,
        "arch": arch,
        "model_name": resolved_model_name,
        "clean_resize": clean_resize,
        "image_size": DINO_IMAGE_SIZE,
    }


def preprocess_dinov2_batch(
    batch_images: np.ndarray,
    clean_resize: bool = False,
) -> jnp.ndarray:
    if clean_resize:
        raise NotImplementedError(
            "clean_resize=True is not implemented yet. "
            "The default dgm-eval path uses standard bicubic resize."
        )

    if batch_images.ndim != 4 or batch_images.shape[-1] != 3:
        raise ValueError(
            "Expected batch_images to have shape [B, H, W, 3] in uint8 RGB format."
        )

    processed = []
    for image in batch_images:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        resized = pil_image.resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), resample=_BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - DINO_MEAN) / DINO_STD
        processed.append(np.transpose(image_np, (2, 0, 1)))

    batch = np.stack(processed, axis=0)
    return jnp.asarray(batch, dtype=jnp.float32)


def compute_dinov2_features(
    batch_images: np.ndarray,
    dino_net: Dict[str, object],
) -> jnp.ndarray:
    batch = preprocess_dinov2_batch(
        batch_images,
        clean_resize=dino_net["clean_resize"],
    )
    features = dino_net["fn"](dino_net["params"], batch)
    log_for_0(f"DINOv2 feature batch: {tuple(features.shape)}")
    return features
