# GT-anchored on-policy FM — draft v3 vs. what is implemented

Draft: `docs/gt_anchored_on_policy_fm.tex` (received 2026-09-14). It supersedes the
earlier rollout-based "Self-Forcing Flow Matching" write-up, which is not to be
revisited.

## What changed in v3

v3 keeps the v2 construction (predict an endpoint, re-noise it, supervise with the
**real** endpoint) but reframes it as the **cheap** end of a compute–fidelity
spectrum, and adds two things that were not in v2:

1. **Explicit `K`-step Euler rollouts**, `K ∈ {1, 2, 4}`, as the *primary* method —
   the induced state comes from actually running the model's dynamics rather than
   from endpoint-prediction + re-noising. Cost is `K+1` forwards per example.
2. **A new supervision ablation (draft Q4):** at the same induced state, compare the
   endpoint-directed corrective target against the *original FM velocity*
   `ε − x_0`. This is a third target option, distinct from both of ours.

Also newly first-class: Q6, a generic-perturbation control whose displacement
magnitude is matched to `‖x_s^θ − x_t^FM‖`, to show the benefit is specific to
*model-induced* states rather than generic state-space augmentation.

## Current experiment = the cheap variant

We are deliberately running only the cheap variant for now. It is implemented in
`sit.py::PlainSiT.forward_gt_on_policy`, gated on `model.sit_gt_on_lambda`.

**Time-convention warning.** The draft uses `t=0` data / `t=1` noise. This repo is
the reverse: `utils/sit_transport_jax.py` sets `alpha_t = t`, `sigma_t = 1-t`, so
**`t=0` is noise and `t=1` is data**. Every formula below is in *repo* coordinates,
where the draft's `(x_0 − x̂_{t'})/(0 − t')` becomes `(x1 − x̂_{t'})/(1 − t')`.

| draft concept | repo implementation |
|---|---|
| `x_t^FM` | the ordinary FM example, reused from `transport.training_losses` |
| `x̂_0^θ` (endpoint prediction) | `_velocity_to_data(pred, xt, t)`, then `stop_gradient` |
| `Corrupt(x̂_0^θ, ε', t')` | `transport.path_sampler.plan(t_prime, eps, x1_hat)` |
| `u_cheap` | `(x1 − x̂_{t'}) / (1 − t')` |
| `λ L_FM + (1−λ) L_on` | `sit_gt_on_lambda` (0.5 for the first run) |
| endpoint = GT `x_0` | `sit_gt_on_target: data` |
| endpoint = self `x̂_0^θ` (Q5) | `sit_gt_on_target: self` |

The cheap variant costs **2 forwards/step, not 3**: the endpoint prediction is
recovered from the FM branch's own forward pass, because
`utils/sit_transport_jax.py` now returns `xt`/`x0` alongside `pred`/`target`.

### The `1/(1−t')` divergence

The target decomposes as

```
u = (x1 − ε')  +  [t'/(1−t')] · (x1 − x̂1^θ)
    └ standard FM ┘  └ correction, coefficient → ∞ at the data end ┘
```

Handled by **truncating the support**, not clamping the divisor:
`t' = (1−δ)·U(t0,t1)` with `δ = sit_gt_on_t_delta = 0.2`, so `1−t' ≥ 0.2` and the
coefficient is capped at 4. Truncation keeps the target exact everywhere it is
used; clamping would make targets in `(1−δ, 1]` silently wrong. No loss
reweighting (SiT has none for velocity prediction) and no gradient clipping.

## Arms currently configured

All from `configs/caltech_plain_sit_gton_config.yml`, plain SiT-XL/2 on
Caltech-101, FID-5k during training.

| arm | flag | draft question |
|---|---|---|
| A | `--config.eval_only=True` | base checkpoint reference |
| B | `--config.model.sit_gt_on_lambda=1.0` | Q1 control (matched-budget ordinary FM) |
| C | as configured, λ=0.5 | Q1 — the cheap method |
| D | `--config.model.sit_gt_on_target=self` | Q5 — is the GT anchor necessary |

Arm B uses λ=1.0 rather than disabling the branch so B and C see an identical data
order and bit-identical FM examples; at λ=1 the corrective term contributes exactly
no gradient. Verified: λ=1 reproduces the plain SiT loss to `|diff| = 0`.

## Not yet implemented (v3 additions, in rough priority order)

1. **`K`-step Euler rollout** state construction (`K ∈ {1,2,4}`) — the draft's
   primary method. Needs a new state-generation branch; the target and loss code
   already generalize since they only depend on `(x_induced, s)`.
2. **`gt_on_target: fm_velocity`** — draft Q4's `ε − x_0` target at the induced
   state. A third enum value; the FM target is already available as
   `terms["target"]`.
3. **Q6 generic-perturbation control** — needs `‖δ_self‖` measured first, which the
   existing `gt_on_delta_rms` diagnostic already logs.
4. `Δt` ablation, gradient-through-rollout, λ sweep.

## Diagnostics already logged

`loss_transport`, `loss_gt_on`, `t_mean`, `t_prime_mean`, `gt_on_delta_rms`
(`= ‖x1 − x̂1^θ‖`), `gt_on_fm_target_rms` (`= ‖x1 − ε'‖`),
`gt_on_corr_coeff_mean`, `gt_on_corr_vs_fm_ratio`. The ratio is the δ sanity check:
if it is ≫2 at step 0, δ=0.2 was too permissive for this checkpoint.
