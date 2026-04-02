import jax
import jax.numpy as jnp
from jax import random
from typing import Any
from functools import partial
from copy import deepcopy
import ml_collections
import optax

from flax.training import train_state
from flax import struct
from utils.logging_util import log_for_0
from utils.state_util import print_params
from utils.ema_util import update_ema


#######################################################
#                    Initialize                       #
#######################################################


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 4)
    x = jnp.ones(input_shape)
    t = jnp.ones((1,), dtype=int)
    y = jnp.ones((1,), dtype=int)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing params done.")

    param_count = sum(x.size for x in jax.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: " + str(param_count))
    return variables, variables["params"]


#######################################################
#                     Train State                     #
#######################################################


class TrainState(train_state.TrainState):
    ema_params: Any
    source_params: Any
    grad_accum: Any
    grad_accum_step: Any


@struct.dataclass
class EvalState:
    step: Any
    params: Any
    ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_fn
):
    """
    Create initial training state.
    ---
    apply_fn: output a dict, with key 'loss', 'mse'
    """

    rng, rng_init = random.split(rng)

    _, params = initialized(rng_init, image_size, model)
    use_ema = config.training.get("use_ema", True)
    ema_params = deepcopy(params)
    if use_ema:
        ema_params = update_ema(ema_params, params, 0)
    needs_source_params = (
        config.model.get("use_dogfit", False)
        or config.training.get("capture_source_from_load", False)
    )
    if needs_source_params:
        source_params = deepcopy(params)
    else:
        source_params = None
    grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad_accum_step = jnp.array(0, dtype=jnp.int32)
    print_params(params["net"])

    tx = optax.adamw(
        learning_rate=lr_fn,
        weight_decay=0,
        b2=config.training.adam_b2,
    )
    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        source_params=source_params,
        grad_accum=grad_accum,
        grad_accum_step=grad_accum_step,
        tx=tx,
    )
    return state
