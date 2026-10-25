"""Convert a Flax plain-SiT checkpoint (flaxSiT_XL_2, e.g. a target-domain FT run) to a torch SiT state_dict.

The output uses the official SiT key layout, so it loads through the same torch -> Flax converters as
SiT-XL-2-256.pt (including the DMF split into encoder/decoder blocks). Inverse of
utils.ckpt_util._convert_torch_sit_state_dict_to_flax_exact. Uses `params` (the FT runs train without EMA).

Usage: python scripts/ft_sit_to_torch.py <flax_checkpoint_dir> <out.pt> [--check]
  --check: re-convert the .pt with the repo's torch->Flax exact converter and assert bit-equality.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from flax import traverse_util
from flax.training import checkpoints


def flax_sit_to_torch(net):
    g = lambda path: np.asarray(net[path], dtype=np.float32)
    sd = {}
    sd["x_embedder.proj.weight"] = np.transpose(g("x_embedder/proj/kernel"), (3, 2, 0, 1))
    sd["x_embedder.proj.bias"] = g("x_embedder/proj/bias")
    sd["pos_embed"] = g("pos_embed")[None]  # torch keeps a leading batch dim
    sd["y_embedder.embedding_table.weight"] = g("y_embedder/embedding_table/_flax_embedding/embedding")
    for i, fc in ((0, "fc1"), (2, "fc2")):
        sd[f"t_embedder.mlp.{i}.weight"] = g(f"t_embedder/{fc}/_flax_linear/kernel").T
        sd[f"t_embedder.mlp.{i}.bias"] = g(f"t_embedder/{fc}/_flax_linear/bias")
    n_blocks = 1 + max(int(k.split("/")[0].split("_")[1]) for k in net if k.startswith("blocks_"))
    for i in range(n_blocks):
        b = f"blocks_{i}"
        lin = lambda name, part: g(f"{b}/{name}/_flax_linear/{part}")
        sd[f"blocks.{i}.attn.qkv.weight"] = np.concatenate(
            [lin(f"attn/{x}_proj", "kernel").T for x in "qkv"], axis=0)
        sd[f"blocks.{i}.attn.qkv.bias"] = np.concatenate([lin(f"attn/{x}_proj", "bias") for x in "qkv"])
        sd[f"blocks.{i}.attn.proj.weight"] = lin("attn/out_proj", "kernel").T
        sd[f"blocks.{i}.attn.proj.bias"] = lin("attn/out_proj", "bias")
        for fc in ("fc1", "fc2"):
            sd[f"blocks.{i}.mlp.{fc}.weight"] = lin(f"mlp/{fc}", "kernel").T
            sd[f"blocks.{i}.mlp.{fc}.bias"] = lin(f"mlp/{fc}", "bias")
        sd[f"blocks.{i}.adaLN_modulation.1.weight"] = lin("adaLN_modulation", "kernel").T
        sd[f"blocks.{i}.adaLN_modulation.1.bias"] = lin("adaLN_modulation", "bias")
    sd["final_layer.linear.weight"] = g("final_layer/linear/_flax_linear/kernel").T
    sd["final_layer.linear.bias"] = g("final_layer/linear/_flax_linear/bias")
    sd["final_layer.adaLN_modulation.1.weight"] = g("final_layer/adaLN_modulation/_flax_linear/kernel").T
    sd["final_layer.adaLN_modulation.1.bias"] = g("final_layer/adaLN_modulation/_flax_linear/bias")
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


if __name__ == "__main__":
    src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    restored = checkpoints.restore_checkpoint(src, target=None)
    net = traverse_util.flatten_dict(restored["params"]["net"], sep="/")
    sd = flax_sit_to_torch(net)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(sd, out)
    print(f"saved {len(sd)} tensors from {src} (step {int(np.asarray(restored['step']).reshape(-1)[0])}) to {out}")
    if "--check" in sys.argv:
        from utils.ckpt_util import _convert_torch_sit_state_dict_to_flax_exact
        back = traverse_util.flatten_dict(
            _convert_torch_sit_state_dict_to_flax_exact(torch.load(out))["net"], sep="/")
        back["pos_embed"] = np.asarray(back["pos_embed"])[0]
        assert set(back) == set(net), sorted(set(back) ^ set(net))[:10]
        bad = [k for k in net if not np.array_equal(np.asarray(net[k], np.float32), np.asarray(back[k], np.float32))]
        assert not bad, bad[:10]
        print(f"round-trip check OK: {len(net)} tensors bit-identical")
