"""Checks for SiT-FT AFM (afm_sit_ft.py, train_afm_sit_ft.py). Run: python tests/test_afm_sit_ft.py"""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from afm import discriminator_loss, generator_loss
from afm_sit_ft import drop_labels, fixed_grid_endpoint_terms, fixed_grid_times, sample_step_indices
from configs.load_config import get_config
from models import imfDiT
from models.sit_ft_discriminator import create_sit_ft_discriminator
from train_caimf import _copy_matching_generator_params

imfDiT.flaxSiT_T_test = partial(imfDiT.FlaxSiT, depth=2, hidden_size=64, patch_size=2, num_heads=2, learn_sigma=True)
import train_afm_sit_ft as T  # noqa: E402


def _cfg(model_str="flaxSiT_T_test"):
    config = get_config("sit_ft_afm_tab2")
    config.model.model_str = model_str
    config.model.num_classes = 7
    config.dataset.num_classes = 7
    return config


def _toy(batch=8, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    images = jax.random.normal(k1, (batch, 32, 32, 4))
    noise = jax.random.normal(k2, (batch, 32, 32, 4))
    labels = jax.random.randint(k3, (batch,), 0, 7)
    return images, noise, labels


def test_grid_is_the_sampler_grid():
    times = np.asarray(T._grid(_cfg()))
    assert times.shape == (5,), times
    np.testing.assert_allclose(np.diff(times), np.diff(times)[0], atol=1e-6)
    # Plain-SiT Linear/velocity ODE: noise at the first time, data at the last (SiT convention).
    assert times[0] < times[-1]
    print("   grid", times.tolist())


def test_straight_line_flow_gives_exact_endpoints():
    images, noise, labels = _toy()
    times = fixed_grid_times(0.0, 1.0, 4)
    oracle = lambda x, s, y: images - noise          # true velocity of the straight SiT path
    for k in range(4):
        idx = jnp.full((images.shape[0],), k, jnp.int32)
        terms = fixed_grid_endpoint_terms(oracle, images, labels, noise, idx, times)
        err = float(jnp.max(jnp.abs(terms["x_r_fake"] - terms["x_r"])))
        assert err < 1e-5, (k, err)
        np.testing.assert_allclose(np.asarray(terms["interval"]), 0.25, atol=1e-6)
        assert bool(jnp.all(terms["r"] > terms["t"]))
    # chaining the 4 jumps from pure noise lands on the data
    x = noise
    for k in range(4):
        x = x + (times[k + 1] - times[k]) * oracle(x, times[k], labels)
    assert float(jnp.max(jnp.abs(x - images))) < 1e-5


def test_endpoint_matches_one_sampler_euler_step():
    images, noise, labels = _toy()
    times = fixed_grid_times(0.0, 1.0, 4)
    fn = lambda x, s, y: jnp.tanh(x) * (1.0 + s.reshape(-1, 1, 1, 1))
    idx = sample_step_indices(jax.random.PRNGKey(3), images.shape[0], 4)
    terms = fixed_grid_endpoint_terms(fn, images, labels, noise, idx, times)
    s = times[idx]; s_next = times[idx + 1]
    euler = terms["x_t"] + (s_next - s).reshape(-1, 1, 1, 1) * fn(terms["x_t"], s, labels)
    np.testing.assert_allclose(np.asarray(terms["x_r_fake"]), np.asarray(euler), atol=1e-6)


def test_loss_signs():
    hi, lo = jnp.full((4,), 2.0), jnp.full((4,), -2.0)
    x = jnp.zeros((4, 2, 2, 1)); interval = jnp.full((4,), 0.25)
    d_good, _ = discriminator_loss(hi, lo, 0.0, 0.0, lambda_gp=0.0, lambda_cp=0.0)
    d_bad, _ = discriminator_loss(lo, hi, 0.0, 0.0, lambda_gp=0.0, lambda_cp=0.0)
    assert float(d_good) < float(d_bad), (d_good, d_bad)        # D wants real > fake
    g_fooled, _ = generator_loss(lo, hi, x, x, interval, lambda_adv=1.0, lambda_ot=0.0)
    g_caught, _ = generator_loss(hi, lo, x, x, interval, lambda_adv=1.0, lambda_ot=0.0)
    assert float(g_fooled) < float(g_caught), (g_fooled, g_caught)  # G wants fake > real


def test_label_dropout_and_no_cfg_in_training():
    labels = jnp.arange(10000) % 7
    assert bool(jnp.all(drop_labels(jax.random.PRNGKey(0), labels, 7, 0.0) == labels))
    dropped = drop_labels(jax.random.PRNGKey(0), labels, 7, 0.1)
    frac = float(jnp.mean(dropped == 7))
    assert 0.08 < frac < 0.12, frac
    assert bool(jnp.all((dropped == labels) | (dropped == 7)))
    # Training endpoint: ONE conditional pass per sample (batch B, given labels), no null/CFG pass.
    config = _cfg()
    model, _, _ = T._create_models(config)
    images, _, labels = _toy(batch=4)
    params = model.init({"params": jax.random.PRNGKey(1)}, images, jnp.full((4,), 0.5), labels)["params"]
    calls = []
    orig_apply = model.apply
    def counting_apply(variables, x, s, y, **kw):
        calls.append((x.shape[0], np.asarray(y).tolist()))
        return orig_apply(variables, x, s, y, **kw)
    object.__setattr__(model, "apply", counting_apply)
    terms = T._make_endpoint_terms(model, params, images, labels, jax.random.PRNGKey(2), config)
    assert calls == [(4, np.asarray(labels).tolist())], calls
    assert terms["x_r_fake"].shape == images.shape
    assert "guidance" not in " ".join(config.afm.keys())
    assert float(config.sampling.omega) == 1.5 and str(config.sampling.method) == "euler"


def test_discriminator_initialised_from_generator():
    config = _cfg()
    model, gen_net, dis = T._create_models(config)
    images, _, labels = _toy(batch=2)
    g_params = model.init({"params": jax.random.PRNGKey(1)}, images, jnp.full((2,), 0.5), labels)["params"]
    d_params = dis.init({"params": jax.random.PRNGKey(2)}, images, jnp.full((2,), 0.5), labels)["params"]
    d_params, copied = _copy_matching_generator_params(d_params, g_params["net"])
    n_d = len(jax.tree_util.tree_leaves(d_params))
    n_new = len(jax.tree_util.tree_leaves({k: v for k, v in d_params.items() if k.startswith("dis_")}))
    assert copied == n_d - n_new, (copied, n_d, n_new)
    out = dis.apply({"params": d_params}, images, jnp.full((2,), 0.5), labels)
    assert out.shape == (2,) and float(jnp.max(jnp.abs(out))) == 0.0   # zero-init head


def test_cfg_only_at_sampling():
    """The eval sampler applies CFG at omega=1.5 (uncond + w (cond - uncond)); omega=1 is plain conditional."""
    from utils.sample_util import has_controllable_sampling_guidance
    from utils.sit_sample_util import _guided_velocity
    from train_sit import _build_plain_sit
    config = _cfg()
    assert has_controllable_sampling_guidance(config.model)   # train_afm then passes sampling.omega
    model = _build_plain_sit(config, eval_mode=True)
    images, _, labels = _toy(batch=4)
    s = jnp.full((4,), 0.3)
    params = model.init({"params": jax.random.PRNGKey(1)}, images, s, labels)["params"]
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(5), len(leaves))
    params = jax.tree_util.tree_unflatten(tree, [l + 0.05 * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)])
    var = {"params": params}
    cond = model.apply(var, images, s, labels)
    uncond = model.apply(var, images, s, jnp.full((4,), 7, jnp.int32))
    assert float(jnp.max(jnp.abs(cond - uncond))) > 1e-4
    np.testing.assert_allclose(np.asarray(_guided_velocity(model, var, images, s, labels, jnp.asarray(1.0))),
                               np.asarray(cond), atol=1e-5)
    np.testing.assert_allclose(np.asarray(_guided_velocity(model, var, images, s, labels, jnp.asarray(1.5))),
                               np.asarray(uncond + 1.5 * (cond - uncond)), atol=1e-4)


if __name__ == "__main__":
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
