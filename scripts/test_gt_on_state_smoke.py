"""CPU smoke test for the gt_on_state branches of forward_gt_on_policy.

Runs on a two-layer backbone so it costs no GPU and no checkpoint: the point is
to trace and execute every branch, not to measure anything. Catches the class of
bug that otherwise only shows up thirty seconds into a real launch -- a missing
diagnostic key, a shape that only works for one construction, a mask with the
wrong dtype.

    JAX_PLATFORMS=cpu .venv/bin/python scripts/test_gt_on_state_smoke.py
"""

import functools
import sys

import jax
import jax.numpy as jnp

import models.imfDiT as imfDiT
from sit import PlainSiT

# A backbone small enough to run on a laptop CPU. The prefix must stay
# "flaxSiT" -- sit.py:95 gates on it before the getattr.
imfDiT.flaxSiT_TINY_2 = functools.partial(
    imfDiT.FlaxSiT,
    depth=2,
    hidden_size=64,
    patch_size=2,
    num_heads=2,
    learn_sigma=True,
)

BATCH = 4
# The backbone's positional embedding is built for 32x32 latents at patch 2, so
# the spatial size is not a free parameter here even though the width is.
IMG = 32
CHANNELS = 4


def build(**overrides):
    kwargs = dict(
        model_str="flaxSiT_TINY_2",
        dtype=jnp.float32,
        num_classes=11,
        class_dropout_prob=0.1,
        target_use_null_class=True,
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        objective="sit",
        path_power_k=1.0,
        output_prediction_space="velocity",
        velocity_map_mode="transport",
        input_alignment_mode="none",
        wrapper_eps=1e-3,
        wrapped_loss_weight="none",
        model_time_scale=999.0,
        model_time_flip=True,
        gt_on_lambda=0.0,
        gt_on_mix="lambda",
        gt_on_aux_weight=0.0,
        gt_on_t_delta=0.2,
        gt_on_target="data",
        gt_on_rollout_k=0,
        gt_on_rollout_dt=0.1,
        gt_on_rollout_solver="euler",
        gt_on_rollout_omega=1.0,
        gt_on_rollout_index_random=False,
        gt_on_t_min=0.0,
        gt_on_t_max=0.0,
        gt_on_state="perturb",
        # 4 steps rather than the real 16: the arithmetic and the masking are
        # what is under test, and 4 keeps the unrolled loop cheap on CPU.
        gt_on_traj_steps=4,
        gt_on_traj_solver="heun",
        gt_on_traj_omega=1.5,
        gt_on_traj_index_min=0,
    )
    kwargs.update(overrides)
    return PlainSiT(**kwargs)


def run(label, **overrides):
    model = build(**overrides)
    key = jax.random.PRNGKey(0)
    variables = model.init(
        {"params": key},
        jnp.ones((1, IMG, IMG, CHANNELS), jnp.float32),
        jnp.ones((1,), jnp.float32),
        jnp.ones((1,), jnp.int32),
    )
    # SiT zero-initialises its final layer, so a freshly initialised model has
    # v_theta identically 0 -- every solver then takes the same (zero) step and
    # the rollout tests cannot tell Euler from Heun+CFG. Jitter the parameters so
    # the velocity is a genuine function of x, t and y. Deterministic, and the
    # same jitter for every configuration, so the arms stay comparable.
    leaves, treedef = jax.tree_util.tree_flatten(variables["params"])
    jitter_keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    params = jax.tree_util.tree_unflatten(
        treedef,
        [
            leaf + 0.05 * jax.random.normal(k, leaf.shape, leaf.dtype)
            for leaf, k in zip(leaves, jitter_keys)
        ],
    )
    images = jax.random.normal(
        jax.random.PRNGKey(1), (BATCH, IMG, IMG, CHANNELS), jnp.float32
    )
    labels = jnp.arange(BATCH, dtype=jnp.int32)

    def loss_fn(params):
        # Same entry point the trainer uses: sit_trainstate_util wires
        # apply_fn as partial(model.apply, method=model.forward).
        return model.apply(
            {"params": params},
            images=images,
            labels=labels,
            rngs={"gen": jax.random.PRNGKey(2)},
            method=model.forward,
        )

    # Differentiate it: the rollout sits inside the differentiated function even
    # though every velocity in it is detached, so a non-differentiable construct
    # in there fails here and nowhere earlier.
    (loss, diags), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
    )
    assert jnp.isfinite(loss), f"{label}: loss is not finite ({loss})"
    assert jnp.isfinite(grad_norm), f"{label}: gradient is not finite"
    assert grad_norm > 0, f"{label}: zero gradient, nothing is being trained"
    print(
        f"{label:26s} loss={float(loss):.5f} grad_norm={float(grad_norm):.4f} "
        f"drift_rms={float(diags['gt_on_drift_rms']):.5f} "
        f"target_rms={float(diags['gt_on_target_rms']):.5f}"
    )
    return {k: float(v) for k, v in diags.items()}


def main():
    perturb = run("perturb/data")
    traj_fm = run("trajectory/fm_velocity", gt_on_state="trajectory",
                  gt_on_target="fm_velocity")
    traj_chord = run("trajectory/data", gt_on_state="trajectory",
                     gt_on_target="data")
    sched = run("schedule/fm_velocity", gt_on_state="schedule",
                gt_on_target="fm_velocity")
    late = run("trajectory/late", gt_on_state="trajectory",
               gt_on_target="fm_velocity", gt_on_traj_index_min=2)
    # The GT-anchored rollout arms: state = K steps of the model's own dynamics
    # away from the *true* interpolant, so the state still carries x1.
    interp = run("interp/fm_velocity", gt_on_state="interp",
                 gt_on_target="fm_velocity")
    roll_euler = run("rollout1/euler/fm_velocity", gt_on_rollout_k=1,
                     gt_on_rollout_dt=0.0625, gt_on_target="fm_velocity")
    roll_heun = run("rollout1/heun-cfg/fm_velocity", gt_on_rollout_k=1,
                    gt_on_rollout_dt=0.0625, gt_on_rollout_solver="heun",
                    gt_on_rollout_omega=1.5, gt_on_target="fm_velocity")
    roll_chord = run("rollout1/heun-cfg/data", gt_on_rollout_k=1,
                     gt_on_rollout_dt=0.0625, gt_on_rollout_solver="heun",
                     gt_on_rollout_omega=1.5, gt_on_target="data")
    # Phase 3: the t' band. Three disjoint slices of the same truncated range.
    ROLL4 = dict(gt_on_rollout_k=4, gt_on_rollout_dt=0.0625,
                 gt_on_rollout_solver="heun", gt_on_rollout_omega=1.5,
                 gt_on_target="fm_velocity")
    roll4 = run("rollout4/full-range", **ROLL4)
    band_lo = run("rollout4/band-noise", gt_on_t_min=0.0, gt_on_t_max=0.27, **ROLL4)
    band_mid = run("rollout4/band-middle", gt_on_t_min=0.27, gt_on_t_max=0.54, **ROLL4)
    band_hi = run("rollout4/band-data", gt_on_t_min=0.54, gt_on_t_max=0.8, **ROLL4)
    # Experiment C: one randomly drawn state per example out of the K visited.
    roll4_rand = run("rollout4/random-index", gt_on_rollout_index_random=True, **ROLL4)

    failures = []

    # The two new knobs must be inert at their defaults, or every number already
    # recorded stops being reproducible from this code. Re-running the K=1 arm
    # with the defaults spelled out explicitly has to reproduce it bit for bit.
    roll_heun_again = run("rollout1/defaults-explicit", gt_on_rollout_k=1,
                          gt_on_rollout_dt=0.0625, gt_on_rollout_solver="heun",
                          gt_on_rollout_omega=1.5, gt_on_target="fm_velocity",
                          gt_on_t_min=0.0, gt_on_t_max=0.0,
                          gt_on_rollout_index_random=False)
    for key in ("gt_on_drift_rms", "gt_on_target_rms", "t_prime_mean"):
        if roll_heun_again[key] != roll_heun[key]:
            failures.append(
                f"the new knobs are not inert at their defaults: {key} moved from "
                f"{roll_heun[key]} to {roll_heun_again[key]}"
            )

    # Each band must actually move the mean supervision time, in order.
    if not (band_lo["t_prime_mean"] < band_mid["t_prime_mean"] < band_hi["t_prime_mean"]):
        failures.append(
            "the t' bands do not order by mean supervision time: "
            f"{band_lo['t_prime_mean']:.4f} / {band_mid['t_prime_mean']:.4f} / "
            f"{band_hi['t_prime_mean']:.4f}"
        )
    # ... and stay inside their own bounds.
    for label, d, lo, hi in (
        ("noise", band_lo, 0.0, 0.27),
        ("middle", band_mid, 0.27, 0.54),
        ("data", band_hi, 0.54, 0.8),
    ):
        if not lo - 1e-6 <= d["t_prime_mean"] <= hi + 1e-6:
            failures.append(
                f"the {label} band's mean t' {d['t_prime_mean']} sits outside "
                f"[{lo}, {hi}]"
            )
    # The band restricts t' and nothing else, so the full-range arm's mean must
    # sit between the extreme bands rather than outside them.
    if not band_lo["t_prime_mean"] < roll4["t_prime_mean"] < band_hi["t_prime_mean"]:
        failures.append("the full-range mean t' is not bracketed by the bands")
    # Clamping t_start at t0 means the data-end band gets the longest rollout, so
    # its drift must exceed the noise-end band's. This is the quantity Phase 3 is
    # actually about, and if it does not move the ablation measures nothing.
    if not band_hi["gt_on_drift_rms"] > band_lo["gt_on_drift_rms"]:
        failures.append(
            "the t' band does not change the achieved drift "
            f"({band_hi['gt_on_drift_rms']} at the data end vs "
            f"{band_lo['gt_on_drift_rms']} at the noise end), so Phase 3 would "
            "vary the time axis without varying the rollout it is meant to probe"
        )

    # Experiment C must supervise earlier states on average than the fixed-K arm,
    # which always takes index K.
    if not roll4_rand["gt_on_rollout_index_mean"] < 4.0:
        failures.append(
            "gt_on_rollout_index_random still supervises index K on every example "
            f"(mean {roll4_rand['gt_on_rollout_index_mean']})"
        )
    if roll4["gt_on_rollout_index_mean"] != 4.0:
        failures.append(
            "the fixed-K arm no longer reports index K, so the random-index change "
            "altered the default path"
        )
    # A mixture over shorter rollouts must drift less than always taking all K.
    if not roll4_rand["gt_on_drift_rms"] < roll4["gt_on_drift_rms"]:
        failures.append(
            "the random-index mixture drifts as far as the full-K rollout "
            f"({roll4_rand['gt_on_drift_rms']} vs {roll4['gt_on_drift_rms']})"
        )
    # The supervision time follows the chosen state, so it must fall too.
    if not roll4_rand["t_prime_mean"] < roll4["t_prime_mean"]:
        failures.append(
            "the random-index arm's supervision time did not follow the chosen "
            "state: mean t' is unchanged, so the state and the time disagree"
        )

    # The no-rollout control is the interpolant, so its drift is exactly zero --
    # if it is not, it is not a control.
    if interp["gt_on_drift_rms"] > 1e-6:
        failures.append(
            f"interp drift is {interp['gt_on_drift_rms']}, expected exactly 0"
        )
    # ... and its target must be the plain FM target, i.e. the same rms the
    # trajectory arm's fm_velocity target has (both are x1 - eps).
    if abs(interp["gt_on_target_rms"] - traj_fm["gt_on_target_rms"]) > 1e-4:
        failures.append("interp/fm_velocity target is not the plain FM target")
    # A rollout step must actually move the state off the interpolant.
    if not roll_euler["gt_on_drift_rms"] > 1e-4:
        failures.append("euler rollout does not move the state")
    # Heun + CFG is a different step from one conditional Euler step, so the
    # induced state must differ; if the drifts match, the new knobs are dead.
    if abs(roll_heun["gt_on_drift_rms"] - roll_euler["gt_on_drift_rms"]) < 1e-6:
        failures.append(
            "gt_on_rollout_solver/omega had no effect on the induced state"
        )
    # Same states, two targets: the chord carries 1/(1 - t'), so it is larger.
    if not roll_chord["gt_on_target_rms"] > roll_heun["gt_on_target_rms"]:
        failures.append("rollout chord target is not larger than the FM target")
    if abs(roll_chord["gt_on_drift_rms"] - roll_heun["gt_on_drift_rms"]) > 1e-6:
        failures.append(
            "the two rollout targets do not share the same induced state"
        )
    # The rollout branch keeps its own diagnostics and must not lose them.
    for key in ("gt_on_rollout_dt_mean", "gt_on_rollout_t_start_mean"):
        if key not in roll_heun:
            failures.append(f"rollout lost {key}")
    # The GT-anchored state must stay far closer to the interpolant than the
    # from-noise trajectory state, which is the entire point of the correction.
    if not roll_heun["gt_on_drift_rms"] < traj_fm["gt_on_drift_rms"]:
        failures.append(
            "one GT-anchored rollout step drifts as far as a from-noise "
            "trajectory: the anchoring is not doing anything"
        )

    # The trajectory state must actually leave the interpolant, or the whole
    # construction is a no-op dressed up as a method.
    if not traj_fm["gt_on_drift_rms"] > 1e-4:
        failures.append("trajectory drift is ~0: the rollout is not moving")
    # The schedule control is the interpolant by construction, so its drift is
    # exactly zero. If it is not, the control is not a control.
    if sched["gt_on_drift_rms"] > 1e-6:
        failures.append(
            f"schedule drift is {sched['gt_on_drift_rms']}, expected exactly 0"
        )
    # 4 schedule steps with delta=0.2 caps the index at 3.
    if traj_fm["gt_on_traj_k_max"] != 3.0:
        failures.append(f"k_max is {traj_fm['gt_on_traj_k_max']}, expected 3")
    # index_min=2 must raise the mean index above the unrestricted draw's.
    if not late["gt_on_traj_k_mean"] > traj_fm["gt_on_traj_k_mean"]:
        failures.append("index_min did not restrict the draw to later indices")
    # The two candidate targets disagree at the visited state; that gap is the
    # whole question, so a zero gap would mean the branches are identical.
    if not traj_fm["gt_on_traj_target_gap_rms"] > 1e-4:
        failures.append("the two trajectory targets are numerically identical")
    # The perturb branch keeps its legacy diagnostic; the others must not have it.
    if "gt_on_corr_vs_fm_ratio" not in perturb:
        failures.append("perturb lost gt_on_corr_vs_fm_ratio")
    for name, d in (("trajectory", traj_fm), ("schedule", sched)):
        if "gt_on_corr_vs_fm_ratio" in d:
            failures.append(f"{name} logged the perturb-only ratio")
    # The chord target carries 1/(1 - t'); the plain FM target does not. On the
    # same states the chord must therefore be the larger of the two.
    if not traj_chord["gt_on_target_rms"] > traj_fm["gt_on_target_rms"]:
        failures.append("chord target is not larger than the plain FM target")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall gt_on_state branches run, differentiate, and disagree as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
