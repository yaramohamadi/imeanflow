import flax.linen as nn
import jax
import jax.numpy as jnp

from models import imfDiT


def generate(variable, model, rng, n_sample, config, 
             num_steps, omega, t_min, t_max, sample_idx=None):
    """
    Generate samples from the model
    
    Args:
        variable: Model parameters.
        model: iMeanFlow model.
        rng: JAX random key.
        n_sample: Number of samples to generate.
        config: Configuration object.
        num_steps: Number of sampling steps.
        omega: CFG scale.
        t_min, t_max: Guidance interval.
        sample_idx: Optional index for class-conditional sampling.

    Returns:
        images: Generated images.
    """
    num_classes = config.dataset.num_classes
    img_size, img_channels = config.dataset.image_size, config.dataset.image_channels

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng, rng_xt, rng_sample = jax.random.split(rng, 3)

    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    meanflow_reverse_time = bool(config.sampling.get("meanflow_reverse_time", False))
    if model._uses_auxiliary_v_head() or meanflow_reverse_time:
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
    else:
        t_steps = jnp.linspace(0.0, 1.0, num_steps + 1)

    def step_fn(i, x_i):
        return model.apply(variable, x_i, y, i, t_steps,
            omega, t_min, t_max, method=model.sample_one_step)

    images = jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    return images


class iMeanFlow(nn.Module):
    """improved MeanFlow"""

    # Model and dataset
    model_str: str
    dtype = jnp.float32
    num_classes: int = 1000

    # Noise distribution
    P_mean: float = -0.4
    P_std: float = 1.0

    # Loss
    data_proportion: float = 0.5
    cfg_beta: float = 1.0
    class_dropout_prob: float = 0.1
    use_dogfit: bool = False
    target_use_null_class: bool = True
    source_prediction_space: str = "v"
    source_num_classes: int = 1000
    use_auxiliary_v_head: bool = True
    use_context_guidance_conditioning: bool = False
    use_training_guidance: bool = True
    training_guidance_interval_strategy: str = "sampled"
    training_guidance_t_min: float = 0.0
    training_guidance_t_max: float = 1.0
    training_guidance_start_step: int = 0
    guidance_scale_strategy: str = "sampled"
    fixed_guidance_scale: float = 7.5
    use_positive_sit_dmf_mf_target: bool = False

    # Training dynamics
    norm_p: float = 1.0
    norm_eps: float = 0.01

    # Evaluation mode
    eval: bool = False

    def setup(self):
        """
        Setup improved MeanFlow model.
        """
        net_fn = getattr(imfDiT, self.model_str)
        net_kwargs = dict(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            eval=self.eval,
        )
        if (not self.use_auxiliary_v_head) and ("SiT_DMF" in self.model_str):
            net_kwargs["use_context_guidance_conditioning"] = self.use_context_guidance_conditioning
        self.net: imfDiT.imfDiT = net_fn(**net_kwargs)
        if self.use_dogfit:
            source_net_kwargs = dict(
                name="source_net",
                num_classes=self.source_num_classes,
                use_null_class=True,
                eval=False,
            )
            if (not self.use_auxiliary_v_head) and ("SiT_DMF" in self.model_str):
                source_net_kwargs["use_context_guidance_conditioning"] = (
                    self.use_context_guidance_conditioning
                )
            self.source_net: imfDiT.imfDiT = net_fn(**source_net_kwargs)

    def _uses_auxiliary_v_head(self):
        return self.use_auxiliary_v_head

    def _uses_sit_dmf_time_convention(self):
        return (not self._uses_auxiliary_v_head()) and ("SiT" in self.model_str)

    def _uses_sit_cfg_channel_rule(self):
        return "SiT" in self.model_str

    def _uses_sit_guidance_context_conditioning(self):
        return (
            (not self._uses_auxiliary_v_head())
            and ("SiT_DMF" in self.model_str)
            and self.use_context_guidance_conditioning
        )

    def _mf_target_interval_coeff(self, t, r):
        if self._uses_sit_dmf_time_convention() and self.use_positive_sit_dmf_mf_target:
            return r - t
        return t - r

    def _sample_guidance_scale(self, bz):
        if self.guidance_scale_strategy == "fixed":
            return jnp.full((bz, 1, 1, 1), self.fixed_guidance_scale, dtype=jnp.float32)
        if self.guidance_scale_strategy != "sampled":
            raise ValueError(
                f"Unsupported guidance_scale_strategy: {self.guidance_scale_strategy}"
            )
        return self.sample_cfg_scale(bz)

    def guided_u_fn(self, x, t, r, omega, t_min, t_max, y):
        """
        Compute a classifier-free guided average velocity for single-head DMF sampling.

        Guidance is applied on the first three channels to mirror the SiT-compatible
        preview path; the remaining channel is taken from the conditioned branch.
        """
        bz = x.shape[0]
        y_null = jnp.full((bz,), self.num_classes, dtype=jnp.int32)
        x_cat = jnp.concatenate([x, x], axis=0)
        y_cat = jnp.concatenate([y, y_null], axis=0)
        t_cat = jnp.concatenate([t, t], axis=0)
        h = t - r
        h_cat = jnp.concatenate([h, h], axis=0)
        omega_cat = jnp.concatenate([omega, omega], axis=0)
        t_min_cat = jnp.concatenate([t_min, t_min], axis=0)
        t_max_cat = jnp.concatenate([t_max, t_max], axis=0)
        u_cat, _ = self.u_fn(
            x_cat,
            t_cat,
            h_cat,
            omega_cat,
            t_min_cat,
            t_max_cat,
            y_cat,
        )
        u_c, u_u = jnp.split(u_cat, 2, axis=0)
        omega_scale = omega.reshape((bz, 1, 1, 1))
        guided_first_three = u_u[..., :3] + omega_scale * (u_c[..., :3] - u_u[..., :3])
        return jnp.concatenate([guided_first_three, u_c[..., 3:]], axis=-1)

    #######################################################
    #                       Solver                        #
    #######################################################

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
        """
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        omega = jnp.broadcast_to(omega, (bsz,))
        t_min = jnp.broadcast_to(t_min, (bsz,))
        t_max = jnp.broadcast_to(t_max, (bsz,))

        if self._uses_auxiliary_v_head():
            u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]
        else:
            u = self.guided_u_fn(z_t, t, r, omega, t_min, t_max, labels)

        return z_t + jnp.einsum("n,n...->n...", r - t, u)

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        """
        Sample t and r from logit-normal distribution.
        """
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        if self._uses_sit_dmf_time_convention():
            t, r = jnp.minimum(t, r), jnp.maximum(t, r)
        else:
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, s_max=7.0):
        """
        Sample CFG scale omega from power distribution.
        """
        ukey = self.make_rng("gen")
        u = jax.random.uniform(
            ukey, (bz, 1, 1, 1), minval=0.0, maxval=1.0, dtype=jnp.float32
        )

        if self.cfg_beta == 1.0:
            s = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            smax = jnp.asarray(s_max, jnp.float32)
            b = jnp.asarray(self.cfg_beta, jnp.float32)

            log_base = (1.0 - b) * jnp.log1p(smax)
            log_inner = jnp.log1p(u * jnp.expm1(log_base))

            s = jnp.exp(log_inner / (1.0 - b))

        return jnp.asarray(s, jnp.float32)

    def sample_cfg_interval(self, bz, fm_mask=None):
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        """
        if self.training_guidance_interval_strategy == "fixed":
            t_min = jnp.full((bz, 1, 1, 1), self.training_guidance_t_min, dtype=self.dtype)
            t_max = jnp.full((bz, 1, 1, 1), self.training_guidance_t_max, dtype=self.dtype)
            return t_min, t_max
        if self.training_guidance_interval_strategy != "sampled":
            raise ValueError(
                "Unsupported training_guidance_interval_strategy: "
                f"{self.training_guidance_interval_strategy}"
            )

        rng_start, rng_end = jax.random.split(self.make_rng("gen"))

        t_min = jax.random.uniform(
            rng_start, (bz, 1, 1, 1), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (bz, 1, 1, 1), minval=0.5, maxval=1.0, dtype=self.dtype
        )

        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        In dual-head mode this returns (u, v_head). In single-head mode it
        returns (u, u_boundary), where u_boundary is u(x_t, t, t, y).

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v_boundary)
            u: Predicted u (average velocity field).
            v_boundary: Auxiliary v prediction in dual-head mode, or the
                single-head boundary estimate in single-head mode.
        """
        bz = x.shape[0]
        if self._uses_auxiliary_v_head():
            return self.net(
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )

        r = t - h
        if self._uses_sit_guidance_context_conditioning():
            u = self.net(
                x,
                t.reshape(bz),
                r.reshape(bz),
                y,
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
            )
            v_boundary = self.net(
                x,
                t.reshape(bz),
                t.reshape(bz),
                y,
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
            )
        else:
            del omega, t_min, t_max
            u = self.net(
                x,
                t.reshape(bz),
                r.reshape(bz),
                y,
            )
            v_boundary = self.net(
                x,
                t.reshape(bz),
                t.reshape(bz),
                y,
            )
        return u, v_boundary

    def v_cond_fn(self, x, t, omega, y):
        """
        Compute the predicted v component conditioned on class labels.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.
        
        Returns:
            v: Predicted v component.
        """

        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)
        return self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned predicted v components.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v_c: Predicted v component conditioned on class labels.
            v_u: Predicted v component without class labels.
        """
        bz = x.shape[0]

        # Create duplicated batch for conditioned and unconditioned predictions
        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.array([self.num_classes] * bz)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        w = jnp.concatenate([omega, jnp.ones_like(omega)], axis=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def source_v_cond_fn(self, source_params, x, t, omega, y):
        """
        Compute a source-model velocity prediction from a frozen source model.

        The interface is intentionally velocity-based so other source families can
        later be adapted behind the same abstraction.
        """
        if source_params is None:
            raise ValueError("source_params must be provided when use_dogfit=True.")
        if self.source_prediction_space != "v":
            raise NotImplementedError(
                f"Unsupported source_prediction_space: {self.source_prediction_space}"
            )

        bz = x.shape[0]
        if self._uses_auxiliary_v_head():
            h = jnp.zeros_like(t)
            t_min = jnp.zeros_like(t)
            t_max = jnp.ones_like(t)
            _, v = self.source_net.apply(
                {"params": source_params["net"]},
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )
        else:
            if self._uses_sit_guidance_context_conditioning():
                t_min = jnp.zeros_like(t)
                t_max = jnp.ones_like(t)
                v = self.source_net.apply(
                    {"params": source_params["net"]},
                    x,
                    t.reshape(bz),
                    t.reshape(bz),
                    y,
                    omega.reshape(bz),
                    t_min.reshape(bz),
                    t_max.reshape(bz),
                )
            else:
                del omega
                v = self.source_net.apply(
                    {"params": source_params["net"]},
                    x,
                    t.reshape(bz),
                    t.reshape(bz),
                    y,
                )
        return v

    def source_v_uncond_fn(self, source_params, x, t):
        bz = x.shape[0]
        y_null = jnp.full((bz,), self.source_num_classes, dtype=jnp.int32)
        omega = jnp.ones_like(t)
        return self.source_v_cond_fn(source_params, x, t, omega, y_null)

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        if (not self.target_use_null_class) or self.class_dropout_prob <= 0:
            return labels, v_g

        bz = v_t.shape[0]

        rand_mask = (
            jax.random.uniform(self.make_rng("gen"), shape=(bz,))
            < self.class_dropout_prob
        )
        num_drop = jnp.sum(rand_mask).astype(jnp.int32)
        drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

        labels = jnp.where(
            drop_mask.reshape(bz),
            self.num_classes,
            labels,
        )
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(
        self,
        v_t,
        z_t,
        t,
        r,
        y,
        fm_mask,
        w,
        t_min,
        t_max,
        source_params=None,
        current_step=None,
    ):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        del r, fm_mask  # This method variant uses one interval-adjusted v_c everywhere.

        if not self.use_training_guidance:
            v_c = self.v_cond_fn(z_t, t, jnp.ones_like(w), y=y)
            return v_t, v_c

        w_eff = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)
        if current_step is not None:
            guidance_enabled = current_step >= jnp.asarray(
                self.training_guidance_start_step, dtype=current_step.dtype
            )
            w_eff = jnp.where(guidance_enabled, w_eff, jnp.ones_like(w_eff))

        if self.use_dogfit:
            v_c = self.v_cond_fn(z_t, t, w_eff, y=y)
            v_u = self.source_v_uncond_fn(source_params, z_t, t)
        else:
            v_c, v_u = self.v_fn(z_t, t, w_eff, y=y)

        if self._uses_sit_cfg_channel_rule():
            guided_first_three = (
                v_t[..., :3]
                + (1 - 1 / w_eff) * (v_c[..., :3] - v_u[..., :3])
            )
            v_g = jnp.concatenate([guided_first_three, v_t[..., 3:]], axis=-1)
        else:
            v_g = v_t + (1 - 1 / w_eff) * (v_c - v_u)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, labels, source_params=None, current_step=None):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).
        
        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            current_step=current_step,
        )

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # Warped u-function for jvp computation
        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)

        # Our compound function V = u + (t - r) * du/dt
        V = u + self._mf_target_interval_coeff(t, r) * jax.lax.stop_gradient(du_dt)

        v_g = jax.lax.stop_gradient(v_g)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # improved MeanFlow objective is conceptually v-loss
        loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss, or single-head boundary loss
        loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        loss = loss_u + loss_v
        loss = loss.mean()  # mean over batch

        dict_losses = {
            "loss": loss,
            "loss_u": jnp.mean((V - v_g) ** 2),
            "loss_v": jnp.mean((v - v_g) ** 2),
        }

        return loss, dict_losses

    def debug_forward(self, images, labels, source_params=None, current_step=None):
        """
        Forward process with intermediate tensors exposed for debugging.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)

        base_w = omega
        if self.use_dogfit:
            v_u = self.source_v_uncond_fn(source_params, z_t, t)
        else:
            _, v_u = self.v_fn(z_t, t, base_w, y=labels)

        v_g, v_c = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            current_step=current_step,
        )

        labels_after_drop, v_g = self.cond_drop(v_t, v_g, labels)

        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels_after_drop)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
        V = u + self._mf_target_interval_coeff(t, r) * jax.lax.stop_gradient(du_dt)

        return {
            "x": x,
            "z_t": z_t,
            "v_t": v_t,
            "v_u": v_u,
            "v_c": v_c,
            "v_g": v_g,
            "v_pred": v,
            "V": V,
            "omega": omega,
            "w_eff_mean": jnp.mean(
                jnp.where(
                    current_step >= jnp.asarray(self.training_guidance_start_step, dtype=current_step.dtype)
                    if current_step is not None
                    else jnp.asarray(True),
                    jnp.where((t >= t_min) & (t <= t_max), omega, 1.0),
                    jnp.ones_like(omega),
                )
            ),
            "t": t,
            "r": r,
            "t_min": t_min,
            "t_max": t_max,
            "fm_mask": fm_mask.astype(self.dtype),
        }

    def __call__(self, x, t, y):
        if self._uses_auxiliary_v_head():
            return self.net(x, t, t, t, t, t, y)  # initialization only
        if self._uses_sit_guidance_context_conditioning():
            ones = jnp.ones_like(t)
            zeros = jnp.zeros_like(t)
            return self.net(x, t, t, y, ones, zeros, ones)  # initialization only
        return self.net(x, t, t, y)  # initialization only
