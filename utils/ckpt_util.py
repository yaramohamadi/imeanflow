import jax
from flax.training import checkpoints
from flax import serialization

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    state = checkpoints.restore_checkpoint(workdir, state)
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def _shape_matches(target_value, source_value):
    if not hasattr(target_value, "shape") or not hasattr(source_value, "shape"):
        return False
    return target_value.shape == source_value.shape


def restore_partial_checkpoint(state, workdir, prefer_ema=True):
    """
    Restore shape-compatible model parameters from a checkpoint.

    This is useful for fine-tuning on a new dataset where some parameter shapes
    differ, such as the class embedding table after changing num_classes.
    Optimizer state and training step are intentionally left fresh.
    """
    restored = checkpoints.restore_checkpoint(workdir, target=None)
    log_for_0("Loaded raw checkpoint from {}".format(workdir))

    source_tree = restored.get("ema_params") if prefer_ema else None
    if source_tree is None:
        source_tree = restored.get("params")
    if source_tree is None:
        raise ValueError(f"No params/ema_params found in checkpoint: {workdir}")

    target_state = serialization.to_state_dict(state.params)
    source_state = serialization.to_state_dict(source_tree)

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
    merged_params = serialization.from_state_dict(state.params, merged_state)
    new_state = state.replace(params=merged_params, ema_params=merged_params)

    log_for_0(
        "Partially restored checkpoint: loaded %d tensors, skipped %d tensors.",
        loaded_count,
        skipped_count,
    )
    if skipped_examples:
        log_for_0("Skipped tensor examples: %s", ", ".join(skipped_examples))

    return new_state


def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    # Save only one copy from device 0.
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    log_for_0("Checkpoint step %d saved.", step)
