import os
import shutil

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import serialization

from utils.logging_util import log_for_0

# The PyTorch model files in models/torch_SiT.py, models/torch_SiT_MF.py,
# and models/torch_DiT.py are not directly imported by the Flax training or
# evaluation code here. This loader only consumes a serialized PyTorch
# checkpoint (.pt/.pth) and converts its SiT state_dict tensors into the
# Flax parameter tree expected by the new imfSiT_MF model.


def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    workdir = os.path.abspath(workdir)
    state = checkpoints.restore_checkpoint(workdir, state)
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .pt checkpoints. Install torch or use a Flax checkpoint instead."
        ) from exc
    return torch


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    try:
        torch = _import_torch()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _set_param(tree, path, value):
    node = tree
    parts = path.split("/")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _transpose_linear(weight):
    return _to_numpy(weight).T


def _convert_qkv(weight, bias):
    w = _to_numpy(weight)
    b = _to_numpy(bias)
    hidden = w.shape[1]
    q, k, v = np.split(w, 3, axis=0)
    qb, kb, vb = np.split(b, 3, axis=0)
    return (
        q.T,
        qb,
        k.T,
        kb,
        v.T,
        vb,
    )


def _load_torch_checkpoint_state_dict(workdir):
    torch = _import_torch()
    raw = torch.load(workdir, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported torch checkpoint format: {workdir}")
    return raw


def _convert_torch_sit_state_dict_to_flax(source_dict):
    """Convert a SiT PyTorch state_dict into a Flax-compatible param tree."""
    state = {}

    def set_linear(module_path, torch_weight, torch_bias):
        _set_param(state, f"{module_path}/_flax_linear/kernel", _transpose_linear(torch_weight))
        _set_param(state, f"{module_path}/_flax_linear/bias", _to_numpy(torch_bias))

    def set_dense(module_path, torch_weight, torch_bias):
        _set_param(state, f"{module_path}/_flax_linear/kernel", _transpose_linear(torch_weight))
        _set_param(state, f"{module_path}/_flax_linear/bias", _to_numpy(torch_bias))

    # Patch embedding
    if "x_embedder.proj.weight" in source_dict:
        _set_param(
            state,
            "x_embedder/proj/kernel",
            np.transpose(_to_numpy(source_dict["x_embedder.proj.weight"]), (2, 3, 1, 0)),
        )
    if "x_embedder.proj.bias" in source_dict:
        _set_param(state, "x_embedder/proj/bias", _to_numpy(source_dict["x_embedder.proj.bias"]))

    # Positional embedding
    if "pos_embed" in source_dict:
        _set_param(state, "pos_embed", _to_numpy(source_dict["pos_embed"]))

    # Time / interval / omega embedder reuse
    # The pretrained SiT timestep embedder is reused for all scalar embedders,
    # including both t_embedder and h_embedder (the r-related signal).
    # That means both t and r conditioning are initialized from the same SiT weights.
    if "t_embedder.mlp.0.weight" in source_dict:
        fc1_w = _transpose_linear(source_dict["t_embedder.mlp.0.weight"])
        fc1_b = _to_numpy(source_dict["t_embedder.mlp.0.bias"])
        fc2_w = _transpose_linear(source_dict["t_embedder.mlp.2.weight"])
        fc2_b = _to_numpy(source_dict["t_embedder.mlp.2.bias"])
        for name in [
            "t_embedder",
            "h_embedder",
            "omega_embedder",
            "t_min_embedder",
            "t_max_embedder",
        ]:
            _set_param(state, f"{name}/fc1/_flax_linear/kernel", fc1_w)
            _set_param(state, f"{name}/fc1/_flax_linear/bias", fc1_b)
            _set_param(state, f"{name}/fc2/_flax_linear/kernel", fc2_w)
            _set_param(state, f"{name}/fc2/_flax_linear/bias", fc2_b)

    # Label embedding
    if "y_embedder.embedding_table.weight" in source_dict:
        _set_param(
            state,
            "y_embedder/embedding_table/_flax_embedding/embedding",
            _to_numpy(source_dict["y_embedder.embedding_table.weight"]),
        )

    # Transformer blocks
    blocks = [
        key for key in source_dict.keys() if key.startswith("blocks.")
    ]
    block_indices = sorted({int(k.split(".")[1]) for k in blocks})
    if block_indices:
        total_blocks = max(block_indices) + 1
        shared_depth = total_blocks - 8
        for i in range(total_blocks):
            path_prefix = f"blocks.{i}"
            qkv_weight = source_dict[f"{path_prefix}.attn.qkv.weight"]
            qkv_bias = source_dict[f"{path_prefix}.attn.qkv.bias"]
            q_w, q_b, k_w, k_b, v_w, v_b = _convert_qkv(qkv_weight, qkv_bias)
            out_w = _transpose_linear(source_dict[f"{path_prefix}.attn.proj.weight"])
            out_b = _to_numpy(source_dict[f"{path_prefix}.attn.proj.bias"])
            mlp_fc1_w = _transpose_linear(source_dict[f"{path_prefix}.mlp.fc1.weight"])
            mlp_fc1_b = _to_numpy(source_dict[f"{path_prefix}.mlp.fc1.bias"])
            mlp_fc2_w = _transpose_linear(source_dict[f"{path_prefix}.mlp.fc2.weight"])
            mlp_fc2_b = _to_numpy(source_dict[f"{path_prefix}.mlp.fc2.bias"])
            ada_w = _transpose_linear(source_dict[f"{path_prefix}.adaLN_modulation.1.weight"])
            ada_b = _to_numpy(source_dict[f"{path_prefix}.adaLN_modulation.1.bias"])

            def assign_block(flax_prefix):
                _set_param(state, f"{flax_prefix}/attn/q_proj/_flax_linear/kernel", q_w)
                _set_param(state, f"{flax_prefix}/attn/q_proj/_flax_linear/bias", q_b)
                _set_param(state, f"{flax_prefix}/attn/k_proj/_flax_linear/kernel", k_w)
                _set_param(state, f"{flax_prefix}/attn/k_proj/_flax_linear/bias", k_b)
                _set_param(state, f"{flax_prefix}/attn/v_proj/_flax_linear/kernel", v_w)
                _set_param(state, f"{flax_prefix}/attn/v_proj/_flax_linear/bias", v_b)
                _set_param(state, f"{flax_prefix}/attn/out_proj/_flax_linear/kernel", out_w)
                _set_param(state, f"{flax_prefix}/attn/out_proj/_flax_linear/bias", out_b)
                _set_param(state, f"{flax_prefix}/mlp/fc1/_flax_linear/kernel", mlp_fc1_w)
                _set_param(state, f"{flax_prefix}/mlp/fc1/_flax_linear/bias", mlp_fc1_b)
                _set_param(state, f"{flax_prefix}/mlp/fc2/_flax_linear/kernel", mlp_fc2_w)
                _set_param(state, f"{flax_prefix}/mlp/fc2/_flax_linear/bias", mlp_fc2_b)
                _set_param(state, f"{flax_prefix}/adaLN_modulation/_flax_linear/kernel", ada_w)
                _set_param(state, f"{flax_prefix}/adaLN_modulation/_flax_linear/bias", ada_b)

            if i < shared_depth:
                assign_block(f"shared_blocks_{i}")
            else:
                head_idx = i - shared_depth
                assign_block(f"u_heads_{head_idx}")
                assign_block(f"v_heads_{head_idx}")

    # Final output layers
    if "final_layer.linear.weight" in source_dict:
        final_w = _transpose_linear(source_dict["final_layer.linear.weight"])
        final_b = _to_numpy(source_dict["final_layer.linear.bias"])
        final_ada_w = _transpose_linear(source_dict["final_layer.adaLN_modulation.1.weight"])
        final_ada_b = _to_numpy(source_dict["final_layer.adaLN_modulation.1.bias"])
        for head in ["u_final_layer", "v_final_layer"]:
            _set_param(state, f"{head}/linear/_flax_linear/kernel", final_w)
            _set_param(state, f"{head}/linear/_flax_linear/bias", final_b)
            _set_param(state, f"{head}/adaLN_modulation/_flax_linear/kernel", final_ada_w)
            _set_param(state, f"{head}/adaLN_modulation/_flax_linear/bias", final_ada_b)

    return state


def _shape_matches(target_value, source_value):
    if not hasattr(target_value, "shape") or not hasattr(source_value, "shape"):
        return False
    return target_value.shape == source_value.shape


def load_checkpoint_params(workdir, prefer_ema=True):
    """
    Load params or ema_params from a checkpoint without shape adaptation.
    Supports Flax checkpoints and PyTorch .pt checkpoints.
    """
    workdir = os.path.abspath(workdir)
    if os.path.isfile(workdir) and workdir.endswith((".pt", ".pth", ".pth.tar")):
        source_tree = _load_torch_checkpoint_state_dict(workdir)
        log_for_0("Loaded PyTorch checkpoint from {}".format(workdir))
        return _convert_torch_sit_state_dict_to_flax(source_tree)

    restored = checkpoints.restore_checkpoint(workdir, target=None)
    log_for_0("Loaded raw checkpoint from {}".format(workdir))

    source_tree = restored.get("ema_params") if prefer_ema else None
    if source_tree is None:
        source_tree = restored.get("params")
    if source_tree is None:
        raise ValueError(f"No params/ema_params found in checkpoint: {workdir}")
    return source_tree


def restore_partial_checkpoint(state, workdir, prefer_ema=True):
    """
    Restore shape-compatible model parameters from a checkpoint.

    This is useful for fine-tuning on a new dataset where some parameter shapes
    differ, such as the class embedding table after changing num_classes.
    Optimizer state and training step are intentionally left fresh.
    """
    workdir = os.path.abspath(workdir)
    source_tree = load_checkpoint_params(workdir, prefer_ema=prefer_ema)

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
    ema_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x, copy=True),
        merged_params,
    )
    new_state = state.replace(params=merged_params, ema_params=ema_params)

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
    workdir = os.path.abspath(workdir)
    # Save only one copy from device 0.
    state = jax.device_get(
        jax.tree_util.tree_map(
            lambda x: x if x is None else x[0],
            state,
            is_leaf=lambda x: x is None,
        )
    )
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    log_for_0("Checkpoint step %d saved.", step)


def save_best_checkpoint(state, workdir, step=None, keep=1):
    """
    Saves the model state to a best-checkpoint directory.

    When keep=1, this also removes any previously saved best checkpoint
    directories so only the latest best checkpoint remains.
    """
    workdir = os.path.abspath(workdir)
    state = jax.device_get(
        jax.tree_util.tree_map(
            lambda x: x if x is None else x[0],
            state,
            is_leaf=lambda x: x is None,
        )
    )
    ckpt_step = int(state.step) if step is None else int(step)
    log_for_0("Saving best checkpoint step %d to %s.", ckpt_step, workdir)
    checkpoints.save_checkpoint_multiprocess(
        workdir,
        state,
        ckpt_step,
        keep=keep,
        overwrite=True,
    )
    log_for_0("Best checkpoint step %d saved.", ckpt_step)

    if keep == 1:
        for entry in os.listdir(workdir):
            if not entry.startswith("checkpoint_"):
                continue
            if entry == f"checkpoint_{ckpt_step}":
                continue
            path = os.path.join(workdir, entry)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    log_for_0("Removed old best checkpoint directory %s.", path)
                except Exception as exc:
                    log_for_0(
                        "Failed to remove old best checkpoint directory %s: %s",
                        path,
                        exc,
                    )
