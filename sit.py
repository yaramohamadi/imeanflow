"""Plain SiT wrapper for official transport-style training."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import imfDiT
from utils.dit_diffusion import create_diffusion
from utils.sit_transport_jax import create_transport, mean_flat


class PlainSiT(nn.Module):
    """Dedicated plain SiT training wrapper around the exact Flax SiT backbone."""

    model_str: str
    dtype: jnp.dtype = jnp.float32
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    path_type: str = "Linear"
    prediction: str = "velocity"
    loss_weight: str = None
    train_eps: float = None
    sample_eps: float = None
    objective: str = "sit"
    path_power_k: float = 1.0
    P_mean: float = -0.4
    P_std: float = 1.0
    data_proportion: float = 0.5
    output_prediction_space: str = "velocity"
    velocity_map_mode: str = "transport"
    input_alignment_mode: str = "none"
    native_velocity_derivative_mode: str = "finite_difference"
    native_diffusion_steps: int = 1000
    native_noise_schedule: str = "linear"
    wrapper_eps: float = 1e-6
    wrapped_loss_weight: str = "none"
    model_time_scale: float = 1.0
    model_time_flip: bool = False
    # Ground-truth-anchored on-policy post-training. Inactive while both
    # `gt_on_lambda` and `gt_on_aux_weight` are None, which leaves the plain SiT
    # loss untouched.
    gt_on_lambda: float = None
    # How the two terms are combined. 'lambda' is the convex mix above.
    # 'additive' is loss = loss_fm + gt_on_aux_weight * loss_corr, which is the
    # form a *small* auxiliary weight is naturally expressed in (0.01 leaves the
    # FM term at weight 1 instead of rescaling the whole loss); gt_on_lambda is
    # then unused. A string rather than "aux_weight is not None" because
    # ml_collections cannot override a None-valued config field from the command
    # line, so the choice has to be expressible as a value of a typed field.
    gt_on_mix: str = "lambda"
    gt_on_aux_weight: float = 0.0
    gt_on_t_delta: float = 0.2
    # 'data'          - the chord to the real endpoint (the method).
    # 'self'          - the chord to the model's own endpoint estimate (cheap
    #                   variant only; the self-endpoint contrast).
    # 'self_velocity' - the detached velocity at the state one Euler step back,
    #                   i.e. local velocity consistency (rollout variant only).
    #                   Same induced states as 'data', different target, which is
    #                   what isolates state exposure from ground-truth anchoring.
    gt_on_target: str = "data"
    # Rollout depth for the induced state. 0 = the cheap endpoint-reconstruction
    # variant; K >= 1 = K explicit Euler steps of nominal size `gt_on_rollout_dt`
    # along the model's own dynamics (the draft's primary construction).
    gt_on_rollout_k: int = 0
    gt_on_rollout_dt: float = 0.1
    # Where the induced state comes from.
    # 'perturb'    - the two constructions above. Both start from the *true*
    #                interpolant built from the real x1 and then perturb it, so
    #                both have a step size whose zero limit is exactly plain
    #                flow matching.
    # 'trajectory' - Denoising Resampling Forcing: initialise at pure noise and
    #                integrate the *inference* schedule with the *inference*
    #                solver and guidance, then supervise at one schedule point.
    #                The state never touches the real x1, so there is no knob
    #                that collapses the construction back to plain FM -- which
    #                is the point of the construction.
    gt_on_state: str = "perturb"
    # The inference schedule to reproduce. Must match sampling.num_steps /
    # sampling.method / sampling.omega for the states to be the ones evaluation
    # actually visits.
    gt_on_traj_steps: int = 16
    gt_on_traj_solver: str = "heun"
    gt_on_traj_omega: float = 1.5
    # Ablation knob: the smallest schedule index eligible for supervision. 0
    # makes the whole trajectory eligible; raising it restricts supervision to
    # the later, more data-like part of the trajectory.
    gt_on_traj_index_min: int = 0
    eval: bool = False

    def setup(self):
        self._validate_output_prediction_space()
        self._validate_wrapper_configuration()
        if not (
            self.model_str.startswith("flaxSiT")
            or self.model_str.startswith("flaxDiT")
        ):
            raise ValueError(
                "PlainSiT expects a flaxSiT_* or flaxDiT_* backbone, got "
                f"{self.model_str!r}."
            )

        net_fn = getattr(imfDiT, self.model_str)
        self.net = net_fn(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            use_r_conditioning=(self.objective == "power_meanflow"),
            eval=self.eval,
        )
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
        )
        self.transport = self._maybe_adjust_transport_eps(self.transport)
        if self.objective not in {"sit", "power_meanflow"}:
            raise ValueError(
                "PlainSiT objective must be one of ['sit', 'power_meanflow'], got "
                f"{self.objective!r}."
            )
        if self._needs_native_diffusion_schedule():
            native_diffusion = create_diffusion(
                "",
                noise_schedule=self.native_noise_schedule,
                learn_sigma=True,
                predict_xstart=False,
                rescale_learned_sigmas=False,
                diffusion_steps=self.native_diffusion_steps,
            )
            self.native_alpha = jnp.asarray(
                native_diffusion.sqrt_alphas_cumprod, dtype=jnp.float32
            )
            self.native_sigma = jnp.asarray(
                native_diffusion.sqrt_one_minus_alphas_cumprod, dtype=jnp.float32
            )
            self.native_time_scale = jnp.asarray(
                max(int(self.native_diffusion_steps) - 1, 1), dtype=jnp.float32
            )
            self.native_tau = (
                jnp.arange(int(self.native_diffusion_steps), dtype=jnp.float32)
                / self.native_time_scale
            )
            diff2flow_t_fm = self.native_alpha / jnp.maximum(
                self.native_alpha + self.native_sigma, self.wrapper_eps
            )
            self.diff2flow_t_fm_asc = diff2flow_t_fm[::-1]
            self.native_indices_asc = jnp.arange(
                int(self.native_diffusion_steps), dtype=jnp.float32
            )[::-1]
            self.native_alpha_asc = self.native_alpha[::-1]
            self.native_sigma_asc = self.native_sigma[::-1]

    def _validate_output_prediction_space(self):
        if self.output_prediction_space not in {"velocity", "data", "noise"}:
            raise ValueError(
                "PlainSiT output_prediction_space must be one of "
                "['velocity', 'data', 'noise'], got "
                f"{self.output_prediction_space!r}."
            )
        if self.output_prediction_space != "velocity" and self.objective != "sit":
            raise ValueError(
                "PlainSiT non-velocity output wrappers are only supported for "
                f"objective='sit', got objective={self.objective!r}."
            )

    def _validate_wrapper_configuration(self):
        if self.velocity_map_mode not in {"transport", "dit_native"}:
            raise ValueError(
                "PlainSiT velocity_map_mode must be one of "
                "['transport', 'dit_native'], got "
                f"{self.velocity_map_mode!r}."
            )
        if self.input_alignment_mode not in {"none", "diff2flow"}:
            raise ValueError(
                "PlainSiT input_alignment_mode must be one of "
                "['none', 'diff2flow'], got "
                f"{self.input_alignment_mode!r}."
            )
        if self.native_velocity_derivative_mode not in {
            "finite_difference",
            "analytic",
        }:
            raise ValueError(
                "PlainSiT native_velocity_derivative_mode must be one of "
                "['finite_difference', 'analytic'], got "
                f"{self.native_velocity_derivative_mode!r}."
            )
        if self.output_prediction_space == "velocity":
            if self.velocity_map_mode != "transport":
                raise ValueError(
                    "PlainSiT wrapped velocity_map_mode only applies to native "
                    f"noise/data outputs, got velocity_map_mode={self.velocity_map_mode!r}."
                )
            if self.input_alignment_mode != "none":
                raise ValueError(
                    "PlainSiT input_alignment_mode requires a native noise wrapper, "
                    f"got output_prediction_space={self.output_prediction_space!r}."
                )
        if self.output_prediction_space == "data" and self.velocity_map_mode != "transport":
            raise ValueError(
                "PlainSiT data outputs only support velocity_map_mode='transport', got "
                f"{self.velocity_map_mode!r}."
            )
        if self.input_alignment_mode == "diff2flow":
            if self.path_type != "Linear":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires path_type='Linear', "
                    f"got {self.path_type!r}."
                )
            if self.output_prediction_space != "noise":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires "
                    f"output_prediction_space='noise', got {self.output_prediction_space!r}."
                )
            if self.velocity_map_mode != "transport":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires "
                    f"velocity_map_mode='transport', got {self.velocity_map_mode!r}."
                )

    def _needs_nonzero_transport_eps(self):
        return self.output_prediction_space != "velocity"

    def _needs_native_diffusion_schedule(self):
        return self.output_prediction_space == "noise" and (
            self.velocity_map_mode == "dit_native"
            or self.input_alignment_mode == "diff2flow"
        )

    def _maybe_adjust_transport_eps(self, transport):
        if not self._needs_nonzero_transport_eps():
            return transport

        train_eps = float(transport.train_eps)
        sample_eps = float(transport.sample_eps)
        min_eps = max(float(self.wrapper_eps), 1e-3)
        if train_eps >= min_eps and sample_eps >= min_eps:
            return transport

        return create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=max(train_eps, min_eps),
            sample_eps=max(sample_eps, min_eps),
        )

    def _broadcast_scalar(self, value, ref):
        value = jnp.asarray(value, dtype=jnp.float32)
        return value.reshape((value.shape[0],) + (1,) * (ref.ndim - 1))

    def _map_model_time(self, t):
        t_model = t.astype(self.dtype)
        if self.model_time_flip:
            t_model = 1.0 - t_model
        return t_model * jnp.asarray(self.model_time_scale, dtype=self.dtype)

    def _compute_transport_schedule(self, t, xt):
        t_expanded = t.reshape((t.shape[0],) + (1,) * (xt.ndim - 1))
        alpha_t, d_alpha_t = self.transport.path_sampler.compute_alpha_t(t_expanded)
        sigma_t, d_sigma_t = self.transport.path_sampler.compute_sigma_t(t_expanded)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def _diff2flow_schedule_values(self, t):
        t_query = jnp.clip(
            t.astype(jnp.float32),
            self.diff2flow_t_fm_asc[0],
            self.diff2flow_t_fm_asc[-1],
        )
        tau_model = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_indices_asc
        )
        alpha_tau = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_alpha_asc
        )
        sigma_tau = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_sigma_asc
        )
        return t_query, tau_model, alpha_tau, sigma_tau

    def _prepare_prediction_context(self, xt, t):
        xt = xt.astype(self.dtype)
        t = t.astype(self.dtype)
        context = {
            "model_x": xt,
            "model_t": self._map_model_time(t),
            "velocity_x": xt,
            "velocity_alpha": None,
            "velocity_sigma": None,
        }
        if self.input_alignment_mode != "diff2flow":
            return context

        _, tau_model, alpha_tau, sigma_tau = self._diff2flow_schedule_values(t)
        velocity_x = self._broadcast_scalar(alpha_tau + sigma_tau, xt) * xt.astype(
            jnp.float32
        )
        context["model_x"] = velocity_x.astype(self.dtype)
        context["model_t"] = tau_model.astype(self.dtype)
        context["velocity_x"] = velocity_x.astype(self.dtype)
        context["velocity_alpha"] = alpha_tau.astype(jnp.float32)
        context["velocity_sigma"] = sigma_tau.astype(jnp.float32)
        return context

    def _run_backbone(self, x, t, y, r=None):
        return self.net(
            x.astype(self.dtype),
            t.astype(self.dtype),
            y,
            r=r,
        )

    def _predict_backbone_output(self, x, t, y, r=None):
        r_model = None
        if r is not None:
            r_model = self._map_model_time(r)
        return self._run_backbone(x, self._map_model_time(t), y, r=r_model)

    def _noise_to_data(self, raw_output, xt, alpha_t, sigma_t):
        alpha_b = self._broadcast_scalar(alpha_t, xt)
        sigma_b = self._broadcast_scalar(sigma_t, xt)
        return (xt - sigma_b * raw_output) / jnp.maximum(alpha_b, self.wrapper_eps)

    def _velocity_to_data(self, velocity, xt, t):
        """Recover the data endpoint x1 implied by a transport velocity at xt.

        Inverts the interpolant pair
            xt = alpha_t * x1 + sigma_t * x0
            v  = d_alpha_t * x1 + d_sigma_t * x0
        for x1. On the linear path this is exactly ``xt + (1 - t) * v``.
        """
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._compute_transport_schedule(t, xt)
        denom = d_sigma_t * alpha_t - sigma_t * d_alpha_t
        denom = jnp.where(
            jnp.abs(denom) > self.wrapper_eps,
            denom,
            jnp.where(denom >= 0.0, self.wrapper_eps, -self.wrapper_eps),
        )
        return (d_sigma_t * xt - sigma_t * velocity) / denom

    def _data_to_transport_velocity(self, raw_output, xt, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._compute_transport_schedule(t, xt)
        sigma_safe = jnp.where(
            jnp.abs(sigma_t) > self.wrapper_eps,
            sigma_t,
            jnp.where(sigma_t >= 0.0, self.wrapper_eps, -self.wrapper_eps),
        )
        x1_hat = raw_output
        x0_hat = (xt - alpha_t * x1_hat) / sigma_safe
        return d_alpha_t * x1_hat + d_sigma_t * x0_hat

    def _noise_to_transport_velocity(
        self,
        raw_output,
        xt,
        t,
        *,
        alpha_override=None,
        sigma_override=None,
    ):
        if alpha_override is not None and sigma_override is not None:
            x1_hat = self._noise_to_data(raw_output, xt, alpha_override, sigma_override)
            return x1_hat - raw_output

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._compute_transport_schedule(t, xt)
        x1_hat = self._noise_to_data(raw_output, xt, alpha_t.reshape((alpha_t.shape[0],)), sigma_t.reshape((sigma_t.shape[0],)))
        return d_alpha_t * x1_hat + d_sigma_t * raw_output

    def _noise_to_dit_native_velocity(self, raw_output, xt, t):
        diffusion_steps = max(int(self.native_diffusion_steps), 1)
        time_scale = self.native_time_scale
        model_time_scale = jnp.asarray(self.model_time_scale, dtype=jnp.float32)
        tau = jnp.asarray(t, dtype=jnp.float32) * (model_time_scale / time_scale)
        dtau_dt = model_time_scale / time_scale
        if self.model_time_flip:
            tau = 1.0 - tau
            dtau_dt = -dtau_dt
        tau = jnp.clip(tau, 0.0, 1.0)

        schedule_pos = jnp.clip(
            jnp.rint(tau * time_scale).astype(jnp.int32),
            0,
            diffusion_steps - 1,
        )
        alpha_scalar = self.native_alpha[schedule_pos]
        sigma_scalar = self.native_sigma[schedule_pos]

        if self.native_velocity_derivative_mode == "analytic":
            if self.native_noise_schedule != "linear":
                raise ValueError(
                    "PlainSiT analytic native velocity derivatives currently require "
                    f"native_noise_schedule='linear', got {self.native_noise_schedule!r}."
                )
            beta_start = jnp.asarray(1e-4, dtype=jnp.float32)
            beta_end = jnp.asarray(2e-2, dtype=jnp.float32)
            beta_tau = time_scale * (beta_start + (beta_end - beta_start) * tau)
            sigma_safe = jnp.maximum(sigma_scalar, self.wrapper_eps)
            alpha_dot_tau = -0.5 * beta_tau * alpha_scalar
            sigma_dot_tau = 0.5 * beta_tau * (alpha_scalar ** 2) / sigma_safe
        else:
            next_pos = jnp.where(schedule_pos > 0, schedule_pos - 1, schedule_pos + 1)
            alpha_next = self.native_alpha[next_pos]
            sigma_next = self.native_sigma[next_pos]
            tau_next = self.native_tau[next_pos]
            tau_cur = self.native_tau[schedule_pos]
            dt = tau_next - tau_cur
            dt = jnp.where(
                jnp.abs(dt) > self.wrapper_eps,
                dt,
                jnp.where(dt >= 0.0, self.wrapper_eps, -self.wrapper_eps),
            )
            alpha_dot_tau = (alpha_next - alpha_scalar) / dt
            sigma_dot_tau = (sigma_next - sigma_scalar) / dt

        alpha_b = self._broadcast_scalar(alpha_scalar, xt)
        sigma_b = self._broadcast_scalar(sigma_scalar, xt)
        alpha_dot_b = self._broadcast_scalar(alpha_dot_tau * dtau_dt, xt)
        sigma_dot_b = self._broadcast_scalar(sigma_dot_tau * dtau_dt, xt)
        x0_hat = (xt - sigma_b * raw_output) / jnp.maximum(alpha_b, self.wrapper_eps)
        return alpha_dot_b * x0_hat + sigma_dot_b * raw_output

    def _compute_wrapped_velocity(self, raw_output, xt, t, context=None):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "velocity":
            return raw_output

        if self.output_prediction_space == "data":
            return self._data_to_transport_velocity(raw_output, xt, t)
        if self.velocity_map_mode == "dit_native":
            return self._noise_to_dit_native_velocity(raw_output, xt, t)
        if context is not None and context["velocity_alpha"] is not None:
            return self._noise_to_transport_velocity(
                raw_output,
                context["velocity_x"].astype(self.dtype),
                t,
                alpha_override=context["velocity_alpha"],
                sigma_override=context["velocity_sigma"],
            )
        return self._noise_to_transport_velocity(raw_output, xt, t)

    def _compute_data_prediction(self, raw_output, xt, t, context=None):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "data":
            return raw_output
        if self.output_prediction_space == "noise":
            if context is not None and context["velocity_alpha"] is not None:
                return self._noise_to_data(
                    raw_output,
                    context["velocity_x"].astype(self.dtype),
                    context["velocity_alpha"],
                    context["velocity_sigma"],
                )
            alpha_t, sigma_t, _, _ = self._compute_transport_schedule(t, xt)
            return self._noise_to_data(
                raw_output,
                xt,
                alpha_t.reshape((alpha_t.shape[0],)),
                sigma_t.reshape((sigma_t.shape[0],)),
            )
        raise ValueError(
            "PlainSiT data reconstruction is only defined for output_prediction_space "
            f"'data' or 'noise', got {self.output_prediction_space!r}."
        )

    def _wrapped_velocity_loss_weight(self, t):
        if self.wrapped_loss_weight in {"", "none", None}:
            return 1.0
        if self.wrapped_loss_weight != "denom_squared":
            raise ValueError(
                "PlainSiT wrapped_loss_weight must be one of "
                "['none', 'denom_squared'], got "
                f"{self.wrapped_loss_weight!r}."
            )
        if self.output_prediction_space == "velocity":
            return 1.0

        if self.output_prediction_space == "noise" and self.input_alignment_mode == "diff2flow":
            _, _, alpha_t, sigma_t = self._diff2flow_schedule_values(t.astype(self.dtype))
        else:
            t_expanded = t.reshape((t.shape[0],) + (1,) * 0)
            alpha_t, _ = self.transport.path_sampler.compute_alpha_t(t_expanded)
            sigma_t, _ = self.transport.path_sampler.compute_sigma_t(t_expanded)
        denom = sigma_t if self.output_prediction_space == "data" else alpha_t
        denom = jnp.maximum(jnp.abs(denom), self.wrapper_eps)
        return jnp.square(denom)

    def _predict_transport_output(self, x, t, y, r=None):
        if r is not None:
            raw_output = self._predict_backbone_output(x, t, y, r=r)
            return self._compute_wrapped_velocity(
                raw_output, x.astype(self.dtype), t.astype(self.dtype)
            )

        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        raw_output = self._run_backbone(context["model_x"], context["model_t"], y)
        return self._compute_wrapped_velocity(
            raw_output, x.astype(self.dtype), t.astype(self.dtype), context=context
        )

    def predict_native_output(self, x, t, y, r=None):
        """Return the backbone output in its configured native prediction space."""
        if r is not None:
            return self._predict_backbone_output(x, t, y, r=r)
        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._run_backbone(context["model_x"], context["model_t"], y)

    def predict_data(self, x, t, y, r=None):
        """Reconstruct x1/data from a native non-velocity prediction."""
        raw_output = self.predict_native_output(x, t, y, r=r)
        context = None if r is not None else self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._compute_data_prediction(
            raw_output, x.astype(self.dtype), t.astype(self.dtype), context=context
        )

    def convert_native_output_to_data(self, raw_output, x, t):
        """Reconstruct x1/data from a provided native backbone prediction."""
        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._compute_data_prediction(
            raw_output,
            x.astype(self.dtype),
            t.astype(self.dtype),
            context=context,
        )

    def debug_noise_reconstruction(self, images, labels):
        """Return training-path noise prediction diagnostics for one batch."""
        if self.objective != "sit" or self.output_prediction_space != "noise":
            raise ValueError(
                "debug_noise_reconstruction is only defined for objective='sit' "
                "with output_prediction_space='noise'."
            )

        x1 = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)
        t, x0, x1 = self.transport.sample(x1, self.make_rng("gen"))
        t, xt, _ = self.transport.path_sampler.plan(t, x0, x1)
        context = self._prepare_prediction_context(xt, t)
        x0_hat = self._run_backbone(context["model_x"], context["model_t"], labels)
        x1_hat = self._compute_data_prediction(x0_hat, xt, t, context=context)

        mse_x0 = jnp.mean(jnp.square(x0_hat - x0), axis=tuple(range(1, x0.ndim)))
        mse_x1 = jnp.mean(jnp.square(x1_hat - x1), axis=tuple(range(1, x1.ndim)))
        return {
            "x0": x0,
            "x1": x1,
            "xt": xt,
            "x0_hat": x0_hat,
            "x1_hat": x1_hat,
            "t": t,
            "mse_x0": jnp.mean(mse_x0),
            "mse_x1": jnp.mean(mse_x1),
        }

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)
        return t, r, fm_mask

    def _drop_labels(self, labels, rng):
        if (
            (not self.target_use_null_class)
            or self.class_dropout_prob <= 0.0
            or self.eval
        ):
            return labels

        drop_mask = jax.random.uniform(rng, labels.shape, dtype=jnp.float32)
        drop_mask = drop_mask < self.class_dropout_prob
        null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
        return jnp.where(drop_mask, null_labels, labels)

    def _guided_velocity_sg(self, x, t, labels, omega):
        """The detached, classifier-free-guided velocity the sampler would use.

        Mirrors ``utils/sit_sample_util._guided_velocity`` so that a training
        rollout visits the same states evaluation does. The two branches are
        written out rather than selected with ``lax.cond`` because ``omega`` is a
        static config value here, so the unguided case costs one forward instead
        of two and the guided case never traces the dead branch.
        """
        if float(omega) == 1.0:
            out = self._predict_transport_output(x, t, labels)
        else:
            null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
            x_cat = jnp.concatenate([x, x], axis=0)
            t_cat = jnp.concatenate([t, t], axis=0)
            y_cat = jnp.concatenate([labels, null_labels], axis=0)
            out_cat = self._predict_transport_output(x_cat, t_cat, y_cat)
            cond, uncond = jnp.split(out_cat, 2, axis=0)
            out = uncond + jnp.asarray(omega, dtype=x.dtype) * (cond - uncond)
        return jax.lax.stop_gradient(out)

    def _rollout_inference_trajectory(self, eps, labels, k_index, times, k_max):
        """Integrate the inference ODE from pure noise and stop at ``k_index``.

        ``eps`` is the state at schedule index 0 (repo time runs noise -> data,
        so index 0 is pure noise, exactly as ``sit_sample_util.generate``
        initialises it). Each sample advances until it has taken its own
        ``k_index`` steps and is then held fixed, which gives every sample in the
        batch its own supervision time -- matching plain flow matching's per-
        sample time draw -- at the cost of always paying ``k_max`` steps rather
        than the mean. The loop is a Python loop over a static bound so it
        unrolls: a traced bound would need ``lax.while_loop``, which is not
        reverse-mode differentiable, and this whole rollout sits inside a
        function that is differentiated even though every velocity here is
        detached.
        """
        solver = str(self.gt_on_traj_solver).lower()
        omega = float(self.gt_on_traj_omega)
        x = eps
        for j in range(k_max):
            tau_cur = times[j]
            tau_next = times[j + 1]
            dt = tau_next - tau_cur
            t_cur = jnp.full((x.shape[0],), tau_cur, dtype=x.dtype)
            drift = self._guided_velocity_sg(x, t_cur, labels, omega)
            if solver == "heun":
                x_pred = x + dt * drift
                t_next = jnp.full((x.shape[0],), tau_next, dtype=x.dtype)
                drift_next = self._guided_velocity_sg(x_pred, t_next, labels, omega)
                x_step = x + 0.5 * dt * (drift + drift_next)
            else:
                x_step = x + dt * drift
            # Samples that have already reached their own schedule index stop
            # advancing, so after the loop every sample sits at times[k_index].
            # Reshaped by hand rather than with _broadcast_scalar, which casts to
            # float32 and would hand jnp.where a non-boolean condition.
            still_going = (k_index > j).reshape((-1,) + (1,) * (x.ndim - 1))
            x = jnp.where(still_going, x_step, x)
        return jax.lax.stop_gradient(x)

    def forward(self, images, labels):
        """Compute the official SiT transport loss."""
        if self.objective == "power_meanflow":
            return self.forward_power_meanflow(images, labels)
        if self.gt_on_lambda is not None or self.gt_on_mix == "additive":
            return self.forward_gt_on_policy(images, labels)

        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_loss = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)

        def model_fn(xt, t, y):
            return self._predict_transport_output(xt, t, y)

        terms = self.transport.training_losses(
            model_fn,
            x,
            rng=rng_loss,
            model_kwargs={"y": labels},
        )
        loss_weight = self._wrapped_velocity_loss_weight(terms["t"])
        weighted_loss = terms["loss"] * loss_weight
        loss = jnp.mean(weighted_loss)
        dict_losses = {
            "loss": loss,
            "loss_transport": loss,
            "loss_transport_unweighted": jnp.mean(terms["loss"]),
            "wrapped_loss_weight_mean": jnp.mean(loss_weight),
            "t_mean": jnp.mean(terms["t"]),
        }
        return loss, dict_losses

    def forward_gt_on_policy(self, images, labels):
        """Ground-truth-anchored on-policy loss.

        The input state is model-induced but the regression target is built from
        the *real* data endpoint, so the objective is corrective rather than
        self-consistent.

        Stage 2 reuses the ordinary flow-matching example that ``loss_fm``
        already computes: its velocity prediction inverts to a data estimate
        ``x1_hat``, which is detached. (Reusing it is exactly the draft's
        "perform ordinary FM training examples", and it costs no extra forward
        pass.) Stage 3 re-noises ``x1_hat`` to a fresh time ``t_prime`` and
        regresses the velocity onto the chord that reaches the real ``x1``.

        Note the repo's time convention is the reverse of the draft's: here
        ``t=0`` is noise and ``t=1`` is data, so the draft's
        ``(x_0 - x_hat_t)/(0 - t)`` becomes ``(x1 - x_hat_t)/(1 - t')``.

        ``gt_on_rollout_k`` selects where on the draft's compute--fidelity
        spectrum this sits. ``0`` is the cheap variant above (2 forwards/step).
        ``K >= 1`` replaces stages 2-3 with K detached Euler steps along the
        model's own dynamics, ending at the supervision time -- the draft's
        primary construction, at K+2 forwards/step as implemented (K rollout +
        1 supervised + the anchor branch's; K+1 would be reachable by dropping
        the anchor branch at lambda=0, at the cost of the paired data order).

        ``gt_on_target`` selects what the induced state is regressed onto, and
        with the state construction held fixed this is what separates *state
        exposure* from *ground-truth anchoring*: ``'data'`` uses the chord to the
        real image, ``'self_velocity'`` uses the detached velocity one Euler step
        back and carries no ground truth at all. Running the two at the same
        weight over the same states is the controlled comparison.
        """
        if self.path_type != "Linear":
            raise ValueError(
                "forward_gt_on_policy assumes the linear interpolant so that the "
                "target chord (x1 - xt) / (1 - t) is the constant velocity "
                f"reaching x1 at t=1; got path_type={self.path_type!r}."
            )
        if not 0.0 < self.gt_on_t_delta < 1.0:
            raise ValueError(
                "gt_on_t_delta must lie in (0, 1) so the target divisor stays "
                f"bounded away from zero; got {self.gt_on_t_delta!r}."
            )
        if self.gt_on_target not in {"data", "self", "self_velocity", "fm_velocity"}:
            raise ValueError(
                "gt_on_target must be 'data' (the method), 'self' (the "
                "self-endpoint contrast), 'self_velocity' (local velocity "
                "consistency) or 'fm_velocity' (the plain FM target x1 - eps at "
                f"the induced state); got {self.gt_on_target!r}."
            )
        if self.gt_on_state not in {"perturb", "trajectory", "schedule"}:
            raise ValueError(
                "gt_on_state must be 'perturb' (a perturbation of the true "
                "interpolant), 'trajectory' (the model's own inference "
                "trajectory from pure noise) or 'schedule' (the true "
                "interpolant at the trajectory's schedule times, the "
                f"time-matched control); got {self.gt_on_state!r}."
            )
        if self.gt_on_state in {"trajectory", "schedule"}:
            if int(self.gt_on_traj_steps) < 1:
                raise ValueError(
                    "gt_on_traj_steps is the inference schedule length and must "
                    f"be >= 1; got {self.gt_on_traj_steps!r}."
                )
            if str(self.gt_on_traj_solver).lower() not in {"euler", "heun"}:
                raise ValueError(
                    "gt_on_traj_solver must be 'euler' or 'heun' to match the "
                    f"evaluation sampler; got {self.gt_on_traj_solver!r}."
                )
            if not self.gt_on_traj_omega > 0.0:
                raise ValueError(
                    "gt_on_traj_omega is the rollout's guidance scale and must "
                    f"be positive; got {self.gt_on_traj_omega!r}."
                )
            if int(self.gt_on_traj_index_min) < 0:
                raise ValueError(
                    "gt_on_traj_index_min is a schedule index and cannot be "
                    f"negative; got {self.gt_on_traj_index_min!r}."
                )
        if self.gt_on_mix not in {"lambda", "additive"}:
            raise ValueError(
                "gt_on_mix must be 'lambda' (convex mix, weight gt_on_lambda) "
                "or 'additive' (loss_fm + gt_on_aux_weight * loss_corr); got "
                f"{self.gt_on_mix!r}."
            )
        if self.gt_on_mix == "lambda" and self.gt_on_lambda is None:
            raise ValueError(
                "gt_on_mix='lambda' needs gt_on_lambda set; got None."
            )
        if self.gt_on_mix == "additive" and not self.gt_on_aux_weight > 0.0:
            raise ValueError(
                "gt_on_mix='additive' needs a positive gt_on_aux_weight -- a "
                "weight of zero is plain flow matching and should be run as "
                f"such; got {self.gt_on_aux_weight!r}."
            )
        if int(self.gt_on_rollout_k) < 0:
            raise ValueError(
                "gt_on_rollout_k is the number of Euler steps and cannot be "
                f"negative; got {self.gt_on_rollout_k!r}."
            )
        if int(self.gt_on_rollout_k) > 0 and not self.gt_on_rollout_dt > 0.0:
            raise ValueError(
                "gt_on_rollout_dt must be positive when rolling out; got "
                f"{self.gt_on_rollout_dt!r}."
            )

        x1 = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        # Derive rng_drop/rng_loss exactly as `forward` does, and fold the two
        # extra keys off the base instead of widening the split. That keeps the
        # anchor term's data, noise and time draws bit-identical to the plain
        # loss, so the lambda=1 arm is an exact paired control of the method.
        rng_base = self.make_rng("gen")
        rng_drop, rng_loss = jax.random.split(rng_base)
        rng_t = jax.random.fold_in(rng_base, 1)
        rng_eps = jax.random.fold_in(rng_base, 2)
        labels = self._drop_labels(labels, rng_drop)

        def model_fn(xt, t, y):
            return self._predict_transport_output(xt, t, y)

        # --- anchor term: the untouched flow-matching loss -------------------
        terms = self.transport.training_losses(
            model_fn,
            x1,
            rng=rng_loss,
            model_kwargs={"y": labels},
        )
        loss_weight = self._wrapped_velocity_loss_weight(terms["t"])
        loss_fm = jnp.mean(terms["loss"] * loss_weight)

        # --- stages 2-3: build the model-induced state ------------------------
        # The supervision time is squeezed into [t0, (1 - delta) * t1] rather
        # than drawn full range and clipped: truncating leaves the top of the
        # range unsupervised but keeps the target exact everywhere it is used,
        # whereas clamping the divisor would silently understate the target near
        # the data end. This holds for both constructions below.
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps
        )
        rollout_k = int(self.gt_on_rollout_k)
        t_prime = jax.random.uniform(
            rng_t, (x1.shape[0],), minval=t0, maxval=t1, dtype=x1.dtype
        )
        t_prime = t_prime * (1.0 - self.gt_on_t_delta)
        eps = jax.random.normal(rng_eps, x1.shape, dtype=x1.dtype)

        if self.gt_on_state in {"trajectory", "schedule"} and self.gt_on_target in {
            "self",
            "self_velocity",
        }:
            raise NotImplementedError(
                "gt_on_target={!r} is defined for the 'perturb' constructions "
                "only: 'self' needs the cheap variant's endpoint estimate and "
                "'self_velocity' needs the velocity one Euler step back, neither "
                "of which the trajectory rollout produces. The targets defined "
                "on trajectory states are 'fm_velocity' (the proposal) and "
                "'data' (the state-consistent chord).".format(self.gt_on_target)
            )

        if self.gt_on_state in {"trajectory", "schedule"}:
            # Denoising Resampling Forcing: reproduce the inference trajectory
            # from pure noise and supervise at one of its schedule points. The
            # supervision time is a schedule point rather than a continuous draw,
            # so the run differs from plain FM in the time distribution as well
            # as the state distribution -- the paired control for that is plain
            # FM restricted to the same discrete times, not the continuous one.
            num_traj_steps = int(self.gt_on_traj_steps)
            times = jnp.linspace(t0, t1, num_traj_steps + 1, dtype=x1.dtype)
            # Static schedule arithmetic in Python so the loop bound is static.
            step = (t1 - t0) / num_traj_steps
            t_cap = t1 * (1.0 - self.gt_on_t_delta)
            k_max = int((t_cap - t0) // step)
            index_min = int(self.gt_on_traj_index_min)
            if k_max < 1:
                raise ValueError(
                    "The truncation gt_on_t_delta leaves no usable schedule "
                    f"point: gt_on_traj_steps={num_traj_steps} with "
                    f"delta={self.gt_on_t_delta} caps the index at {k_max}. "
                    "Lower delta or raise the schedule length."
                )
            if index_min > k_max:
                raise ValueError(
                    f"gt_on_traj_index_min={index_min} exceeds the largest index "
                    f"the truncation allows ({k_max}), so no schedule point is "
                    "eligible for supervision."
                )
            k_index = jax.random.randint(
                rng_t, (x1.shape[0],), index_min, k_max + 1, dtype=jnp.int32
            )
            t_prime = times[k_index]
            induced_ref = self.transport.path_sampler.plan(t_prime, eps, x1)[1]
            if self.gt_on_state == "trajectory":
                xt_hat = self._rollout_inference_trajectory(
                    eps, labels, k_index, times, k_max
                )
            else:
                # 'schedule': the analytic interpolant at the *same* schedule
                # time. With gt_on_target='fm_velocity' this is plain flow
                # matching restricted to the inference schedule's timesteps and
                # nothing else -- the paired control that removes the time
                # distribution as a confound when the trajectory arm is compared
                # against continuous-time flow matching. It costs no rollout
                # forwards, and drift is identically zero by construction.
                xt_hat = induced_ref
            # The two candidate targets on this state, and the gap between them.
            # The gap is the whole disagreement between the proposal and its
            # state-consistent repair: 'fm_velocity' points along the (x1, eps)
            # line the state has left, 'data' points from where the state
            # actually is to x1. Logging it costs no forward pass and says
            # directly how wrong the proposed target is at the visited state.
            t_b_traj = self._broadcast_scalar(t_prime, xt_hat)
            u_chord = (x1 - xt_hat) / (1.0 - t_b_traj)
            u_fm = x1 - eps
            extra_diagnostics = {
                "gt_on_traj_k_mean": jnp.mean(k_index.astype(x1.dtype)),
                "gt_on_traj_k_max": jnp.asarray(k_max, dtype=x1.dtype),
                "gt_on_traj_chord_rms": jnp.sqrt(jnp.mean(jnp.square(u_chord))),
                "gt_on_traj_fm_target_rms": jnp.sqrt(jnp.mean(jnp.square(u_fm))),
                "gt_on_traj_target_gap_rms": jnp.sqrt(
                    jnp.mean(jnp.square(u_chord - u_fm))
                ),
            }
        elif rollout_k == 0:
            # Cheap variant: the endpoint prediction the FM forward already
            # produced, re-noised at a fresh time. No extra forward pass.
            x1_hat = jax.lax.stop_gradient(
                self._velocity_to_data(terms["pred"], terms["xt"], terms["t"])
            )
            _, xt_hat, _ = self.transport.path_sampler.plan(t_prime, eps, x1_hat)
            induced_ref = self.transport.path_sampler.plan(t_prime, eps, x1)[1]
            extra_diagnostics = {
                "gt_on_delta_rms": jnp.sqrt(jnp.mean(jnp.square(x1 - x1_hat))),
            }
        else:
            # Rollout variant: K explicit Euler steps along the model's own
            # dynamics, detached. The draft samples the FM time t and lands at
            # s = t - K*dt (its t runs data->noise); in repo coordinates t runs
            # noise->data, so the same construction is reparameterised as "draw
            # the supervision time s, start K nominal steps behind it" -- which
            # additionally guarantees s <= 1 - delta without clipping the
            # target. dt shrinks for samples too close to t0 to fit K steps.
            if self.gt_on_target == "self":
                raise NotImplementedError(
                    "gt_on_target='self' is defined only for the cheap variant "
                    "(rollout_k=0); the draft does not specify a self-endpoint "
                    "target for the rollout construction, so it is not guessed "
                    "here. For a self-referential target on rollout states use "
                    "gt_on_target='self_velocity'."
                )
            dt_nominal = jnp.asarray(self.gt_on_rollout_dt, dtype=x1.dtype)
            t_start = jnp.maximum(t_prime - rollout_k * dt_nominal, t0)
            dt = (t_prime - t_start) / rollout_k
            _, x_roll, _ = self.transport.path_sampler.plan(t_start, eps, x1)
            induced_ref = self.transport.path_sampler.plan(t_prime, eps, x1)[1]
            dt_b = self._broadcast_scalar(dt, x_roll)
            t_cur = t_start
            for _ in range(rollout_k):
                v_step = jax.lax.stop_gradient(
                    self._predict_transport_output(x_roll, t_cur, labels)
                )
                x_roll = x_roll + dt_b * v_step
                t_cur = t_cur + dt
            xt_hat = jax.lax.stop_gradient(x_roll)
            extra_diagnostics = {
                "gt_on_rollout_dt_mean": jnp.mean(dt),
                "gt_on_rollout_t_start_mean": jnp.mean(t_start),
            }

        # --- corrective target: the chord to the REAL endpoint ---------------
        t_b = self._broadcast_scalar(t_prime, xt_hat)
        if self.gt_on_target == "fm_velocity":
            # Denoising Resampling Forcing as proposed: the *plain* FM target for
            # the pair (x1, eps) that started the trajectory, applied unchanged at
            # the induced state. In repo coordinates (t: noise -> data) this is
            # x1 - eps, the draft's eps - x0 under its reversed time.
            #
            # Note what this target is NOT: it is not the velocity that carries
            # xt_hat to x1. The state has drifted off the (x1, eps) line, and this
            # target still points along that line, so it is only consistent with
            # the state in the limit where the rollout does not drift. The
            # consistent alternative on the same states is gt_on_target='data',
            # whose chord (x1 - xt_hat) / (1 - t') is exactly the target obtained
            # by solving xt_hat = t'*x1 + (1 - t')*eps' for the implied noise.
            u_gt_on = x1 - eps
        elif self.gt_on_target == "self_velocity":
            # Local velocity consistency: regress the velocity at the induced
            # state onto the (detached) velocity that produced the step into it.
            # `v_step` is the last loop iterate, i.e. the velocity at the state
            # one Euler step behind the supervision time. Deliberately carries no
            # ground truth: this is the arm that isolates whether exposure to the
            # model's own nearby states helps on its own. Note the target does
            # NOT carry the 1/(1 - t') amplification the chord targets do.
            u_gt_on = v_step
        else:
            endpoint = x1 if self.gt_on_target == "data" else x1_hat
            # 1 - t_prime >= gt_on_t_delta by construction, so no clamp needed.
            u_gt_on = (endpoint - xt_hat) / (1.0 - t_b)

        pred_corr = self._predict_transport_output(xt_hat, t_prime, labels)
        corr_weight = self._wrapped_velocity_loss_weight(t_prime)
        per_sample_corr = mean_flat((pred_corr - u_gt_on) ** 2) * corr_weight
        loss_corr = jnp.mean(per_sample_corr)

        if self.gt_on_mix == "additive":
            aux_w = jnp.asarray(self.gt_on_aux_weight, dtype=jnp.float32)
            loss = loss_fm + aux_w * loss_corr
        else:
            lam = jnp.asarray(self.gt_on_lambda, dtype=jnp.float32)
            loss = lam * loss_fm + (1.0 - lam) * loss_corr

        # Diagnostics for the step-0 sanity checks. `drift_rms` is how far the
        # induced state sits from the interpolant state at the same time with
        # the same noise: if it is ~0 the construction is a no-op. The
        # correction the target carries over plain FM is exactly
        # drift / (1 - t'), so `corr_over_fm_rms` says whether the correction
        # dominates the standard FM part of the target. Both are defined
        # identically for the two constructions, so the series are comparable.
        drift = xt_hat - induced_ref
        fm_target_rms = jnp.sqrt(jnp.mean(jnp.square(x1 - eps)))
        corr_coeff = jnp.mean(t_b / (1.0 - t_b))
        dict_losses = {
            "loss": loss,
            "loss_transport": loss_fm,
            "loss_transport_unweighted": jnp.mean(terms["loss"]),
            "loss_gt_on": loss_corr,
            "t_mean": jnp.mean(terms["t"]),
            "t_prime_mean": jnp.mean(t_prime),
            "gt_on_fm_target_rms": fm_target_rms,
            "gt_on_corr_coeff_mean": corr_coeff,
            "gt_on_drift_rms": jnp.sqrt(jnp.mean(jnp.square(drift))),
            "gt_on_corr_over_fm_rms": (
                jnp.sqrt(jnp.mean(jnp.square(drift / (1.0 - t_b)))) / fm_target_rms
            ),
            # The target's own scale, so the chord targets and the
            # self_velocity target can be compared on the record: they differ by
            # the 1/(1 - t') amplification, not only in what they point at.
            "gt_on_target_rms": jnp.sqrt(jnp.mean(jnp.square(u_gt_on))),
        }
        dict_losses.update(extra_diagnostics)
        if self.gt_on_state == "perturb" and rollout_k == 0:
            # Kept for continuity with the arms already recorded, which logged
            # this product-of-means form rather than the rms above. Only the
            # cheap perturb branch defines gt_on_delta_rms -- the trajectory and
            # schedule branches leave rollout_k at its default 0 and would hit a
            # KeyError here without the state check.
            dict_losses["gt_on_corr_vs_fm_ratio"] = (
                corr_coeff * dict_losses["gt_on_delta_rms"] / fm_target_rms
            )

        # --- the same quantities, binned by t' -------------------------------
        # The batch-mean loss above is NOT comparable across steps: the target
        # carries 1/(1 - t'), so the mean mostly reports which t' were drawn.
        # These bins separate "the model got worse" from "t' happened to be
        # higher", and they are what says whether the gradient is dominated by
        # the top of the truncated range (where the coefficient reaches
        # 1/delta - 1) rather than spread over it. An empty bin logs NaN, which
        # the metric importers skip rather than record as zero.
        # The FM branch is binned on the SAME edges, which is what makes the two
        # branches comparable at all: the batch means are confounded because the
        # branches draw their times from different distributions (question #2).
        n_bins = 4
        per_sample_ratio = jnp.sqrt(mean_flat(jnp.square(drift / (1.0 - t_b))))
        per_sample_fm = terms["loss"] * loss_weight
        edges = jnp.linspace(t0, (1.0 - self.gt_on_t_delta) * t1, n_bins + 1)

        def _binned(times, values, b):
            in_bin = times >= edges[b]
            in_bin = in_bin & (
                times <= edges[b + 1] if b == n_bins - 1 else times < edges[b + 1]
            )
            count = jnp.sum(in_bin)
            total = jnp.sum(jnp.where(in_bin, values, 0.0))
            mean = jnp.where(count > 0, total / jnp.maximum(count, 1), jnp.nan)
            return mean, count.astype(jnp.float32)

        for b in range(n_bins):
            corr_mean, corr_count = _binned(t_prime, per_sample_corr, b)
            ratio_mean, _ = _binned(t_prime, per_sample_ratio / fm_target_rms, b)
            fm_mean, fm_count = _binned(terms["t"], per_sample_fm, b)
            dict_losses[f"gt_on_loss_t{b}"] = corr_mean
            dict_losses[f"gt_on_corr_over_fm_t{b}"] = ratio_mean
            dict_losses[f"gt_on_count_t{b}"] = corr_count
            dict_losses[f"fm_loss_t{b}"] = fm_mean
            dict_losses[f"fm_count_t{b}"] = fm_count
        return loss, dict_losses

    def forward_power_meanflow(self, images, labels):
        """Compute the experimental power-geometry mean-flow loss."""
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)
        bz = x.shape[0]

        rng_drop, rng_eps = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)
        eps = jax.random.normal(rng_eps, x.shape, dtype=self.dtype)

        t, r, fm_mask = self.sample_tr(bz)
        t_scalar = t.reshape((bz,))
        r_scalar = r.reshape((bz,))
        one_minus_t = jnp.clip(1.0 - t, 1e-6, 1.0)
        one_minus_r = jnp.clip(1.0 - r, 1e-6, 1.0)
        t_clamped = jnp.clip(t, 1e-6, 1.0)
        r_clamped = jnp.clip(r, 1e-6, 1.0)
        k = jnp.asarray(self.path_power_k, dtype=self.dtype)

        a_t = one_minus_t**k
        b_t = t_clamped**k
        z_t = a_t * x + b_t * eps

        inst_target = (
            -k * (one_minus_t ** (k - 1.0)) * x
            + k * (t_clamped ** (k - 1.0)) * eps
        )

        a_r = one_minus_r**k
        b_r = r_clamped**k
        z_r = a_r * x + b_r * eps
        denom = jnp.maximum(jnp.abs(t - r), 1e-6)
        mf_target = (z_t - z_r) / denom
        target = jnp.where(fm_mask, inst_target, mf_target)

        pred = self.net(
            z_t.astype(self.dtype),
            t_scalar.astype(self.dtype),
            labels,
            r=r_scalar.astype(self.dtype),
        )
        sq_error = (pred - target.astype(pred.dtype)) ** 2
        per_example_loss = jnp.mean(sq_error.reshape((bz, -1)), axis=1)
        inst_mask = fm_mask.reshape((bz,)).astype(self.dtype)
        mf_mask = 1.0 - inst_mask
        inst_denom = jnp.maximum(jnp.sum(inst_mask), 1.0)
        mf_denom = jnp.maximum(jnp.sum(mf_mask), 1.0)
        inst_loss = jnp.sum(per_example_loss * inst_mask) / inst_denom
        mf_loss = jnp.sum(per_example_loss * mf_mask) / mf_denom
        loss = jnp.mean(per_example_loss)

        dict_losses = {
            "loss": loss,
            "loss_power_meanflow": loss,
            "loss_instantaneous": inst_loss,
            "loss_meanflow": mf_loss,
            "diag_fraction": jnp.mean(inst_mask),
            "t_mean": jnp.mean(t_scalar),
            "r_mean": jnp.mean(r_scalar),
            "interval_mean": jnp.mean(jnp.abs(t_scalar - r_scalar)),
        }
        return loss, dict_losses

    def __call__(self, x, t, y, r=None):
        """Initialization-only forward that mirrors the exact SiT backbone."""
        return self._predict_transport_output(x, t, y, r=r)
