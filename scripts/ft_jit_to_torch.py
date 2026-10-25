"""Convert a Flax plain-JiT checkpoint (flaxJiT_H_16, e.g. a target-domain FT run) to a torch JiT state_dict.

Output uses the official JiT key layout, so it loads through the same torch -> Flax converters as
JiT-H-16-256.pth (including the DMF split). Inverse of utils.ckpt_util._convert_torch_jit_state_dict_to_flax.
Uses `params` (the FT runs train without EMA).

Usage: python scripts/ft_jit_to_torch.py <flax_checkpoint_dir> <out.pt> [--check]
  --check: re-convert the .pt with the repo's torch->Flax JiT converter and assert bit-equality.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from flax import traverse_util
from flax.training import checkpoints


def flax_jit_to_torch(net):
    g = lambda path: np.asarray(net[path], dtype=np.float32)
    sd = {
        "pos_embed": g("pos_embed"),
        "in_context_posemb": g("in_context_posemb"),
        "t_embedder.mlp.0.weight": g("t_embedder/mlp/layers_0/_flax_linear/kernel").T,
        "t_embedder.mlp.0.bias": g("t_embedder/mlp/layers_0/_flax_linear/bias"),
        "t_embedder.mlp.2.weight": g("t_embedder/mlp/layers_2/_flax_linear/kernel").T,
        "t_embedder.mlp.2.bias": g("t_embedder/mlp/layers_2/_flax_linear/bias"),
        "y_embedder.embedding_table.weight": g("y_embedder/embedding_table/_flax_embedding/embedding"),
        "x_embedder.proj1.weight": np.transpose(g("x_embedder/proj1/kernel"), (3, 2, 0, 1)),
        "x_embedder.proj2.weight": np.transpose(g("x_embedder/proj2/kernel"), (3, 2, 0, 1)),
        "x_embedder.proj2.bias": g("x_embedder/proj2/bias"),
        "final_layer.norm_final.weight": g("final_layer/norm_final/kernel"),
        "final_layer.linear.weight": g("final_layer/linear/_flax_linear/kernel").T,
        "final_layer.linear.bias": g("final_layer/linear/_flax_linear/bias"),
        "final_layer.adaLN_modulation.1.weight": g("final_layer/adaLN_modulation/_flax_linear/kernel").T,
        "final_layer.adaLN_modulation.1.bias": g("final_layer/adaLN_modulation/_flax_linear/bias"),
    }
    n_blocks = 1 + max(int(k.split("/")[0].split("_")[1]) for k in net if k.startswith("blocks_"))
    for i in range(n_blocks):
        b, t = f"blocks_{i}", f"blocks.{i}"
        lin = lambda name, part: g(f"{b}/{name}/_flax_linear/{part}")
        sd[f"{t}.norm1.weight"] = g(f"{b}/norm1/kernel")
        sd[f"{t}.attn.q_norm.weight"] = g(f"{b}/attn/q_norm/kernel")
        sd[f"{t}.attn.k_norm.weight"] = g(f"{b}/attn/k_norm/kernel")
        sd[f"{t}.attn.qkv.weight"] = lin("attn/qkv", "kernel").T
        sd[f"{t}.attn.qkv.bias"] = lin("attn/qkv", "bias")
        sd[f"{t}.attn.proj.weight"] = lin("attn/proj", "kernel").T
        sd[f"{t}.attn.proj.bias"] = lin("attn/proj", "bias")
        sd[f"{t}.norm2.weight"] = g(f"{b}/norm2/kernel")
        for fc in ("w12", "w3"):
            sd[f"{t}.mlp.{fc}.weight"] = lin(f"mlp/{fc}", "kernel").T
            sd[f"{t}.mlp.{fc}.bias"] = lin(f"mlp/{fc}", "bias")
        sd[f"{t}.adaLN_modulation.1.weight"] = lin("adaLN_modulation", "kernel").T
        sd[f"{t}.adaLN_modulation.1.bias"] = lin("adaLN_modulation", "bias")
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


if __name__ == "__main__":
    src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    restored = checkpoints.restore_checkpoint(src, target=None)
    net = traverse_util.flatten_dict(restored["params"]["net"], sep="/")
    sd = flax_jit_to_torch(net)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(sd, out)
    print(f"saved {len(sd)} tensors from {src} (step {int(np.asarray(restored['step']).reshape(-1)[0])}) to {out}")
    if "--check" in sys.argv:
        from utils.ckpt_util import _convert_torch_jit_state_dict_to_flax
        back = traverse_util.flatten_dict(_convert_torch_jit_state_dict_to_flax(torch.load(out))["net"], sep="/")
        assert set(back) == set(net), sorted(set(back) ^ set(net))[:10]
        bad = [k for k in net if not np.array_equal(np.asarray(net[k], np.float32), np.asarray(back[k], np.float32))]
        assert not bad, bad[:10]
        print(f"round-trip check OK: {len(net)} tensors bit-identical")
