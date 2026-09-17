"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()

    dataset.name = "imgnet_latent"
    dataset.root = "DATA_ROOT"

    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = False

    dataset.image_size = 32
    dataset.image_channels = 4
    dataset.num_classes = 1000
    dataset.num_classes_from_data = False
    dataset.vae = "mse"

    # ------------------------------------------------------------
    # Training
    config.training = training = ml_collections.ConfigDict()

    training.learning_rate = 0.0001
    training.batch_size = 256
    training.use_ema = True

    training.num_epochs = 1000

    training.log_per_step = 100
    training.sample_per_step = 1000
    training.checkpoint_per_epoch = 10
    training.fid_per_step = 1000
    training.fid_schedule = []
    training.force_fid_steps = ""
    training.force_fid_per_step = 0
    training.force_metric_num_steps = ""
    training.metric_num_steps = ()
    training.preview_guidance_scales = ()
    training.preview_num_steps = ()
    training.debug_log_during_train = False
    training.debug_log_images = True
    training.debug_num_images = 4
    training.debug_velocity_decode_scale = 0.1
    training.grad_accum_steps = 1
    training.save_best_fid_only = False
    training.save_best_fid_eval_state_only = False
    training.best_fid_checkpoint_dir = "best_fid"
    training.save_eval_checkpoint_per_fid = False
    training.eval_checkpoint_dir = "latest_eval"
    training.fid_use_online_only = False
    training.capture_source_from_load = False
    training.half_precision = False
    training.half_precision_dtype = "float16"
    training.print_model_params = False

    training.seed = 42

    training.adam_b2 = 0.95
    training.ema_val = 0.9999

    training.lr_schedule = "warmup_const"
    training.warmup_epochs = 0

    # ------------------------------------------------------------
    # MeanFlow
    config.model = model = ml_collections.ConfigDict()
    model.num_classes = dataset.num_classes

    # Noise Distribution
    model.P_mean = -0.4
    model.P_std = 1.0

    # Loss
    model.data_proportion = 0.5
    model.cfg_beta = 1.0
    model.class_dropout_prob = 0.1
    model.training_mode = "imf_jvp"
    model.use_dogfit = False
    model.target_use_null_class = True
    model.source_prediction_space = "v"
    model.source_model_str = ""
    model.source_num_classes = dataset.num_classes
    model.source_path_type = "Linear"
    model.source_velocity_map_mode = "transport"
    model.source_native_velocity_derivative_mode = "finite_difference"
    model.source_wrapper_eps = 1e-6
    model.source_model_time_scale = 1.0
    model.source_model_time_flip = False
    model.source_native_diffusion_steps = 1000
    model.source_native_beta_schedule = "linear"
    model.target_output_prediction_space = "velocity"
    model.target_velocity_map_mode = "transport"
    model.target_input_alignment_mode = "none"
    model.target_native_velocity_derivative_mode = "finite_difference"
    model.target_native_diffusion_steps = 1000
    model.target_native_beta_schedule = "linear"
    model.target_wrapper_eps = 1e-6
    model.target_model_time_scale = 1.0
    model.target_model_time_flip = False
    model.use_auxiliary_v_head = True
    model.use_context_guidance_conditioning = False
    model.use_adaln_guidance_scale_conditioning = False
    model.adaln_guidance_scale_init = "timestep"
    model.use_adaln_condition_mixing = False
    model.decoder_only_guidance_conditioning = False
    model.time_conditioning_mode = "split"
    model.use_ema_vc = False
    model.use_v_only_teacher_source_copies = False
    model.use_training_guidance = True
    model.training_guidance_interval_strategy = "sampled"
    model.training_guidance_t_min = 0.0
    model.training_guidance_t_max = 1.0
    model.training_guidance_start_step = 0
    model.guidance_scale_strategy = "sampled"
    model.max_sampled_guidance_scale = 8.0
    model.fixed_guidance_scale = 7.5
    model.baked_guidance_blend = 0.5
    model.use_positive_sit_dmf_mf_target = False
    model.split_consistency_midpoint_strategy = "uniform"
    model.split_consistency_midpoint_eps = 1e-3
    model.split_consistency_source_first_prob = 0.0
    model.split_consistency_source_second_prob = 0.0
    model.split_consistency_boundary_mode = "exact"
    model.split_consistency_boundary_epsilon_distribution = "half_normal"
    model.split_consistency_boundary_epsilon = 1e-3
    model.split_consistency_boundary_epsilon_min = 1e-6
    model.output_prediction_space = "epsilon"
    model.sit_output_prediction_space = "velocity"
    model.sit_velocity_map_mode = "transport"
    model.sit_input_alignment_mode = "none"
    model.sit_native_velocity_derivative_mode = "finite_difference"
    model.sit_native_diffusion_steps = 1000
    model.sit_native_noise_schedule = "linear"
    model.sit_wrapper_eps = 1e-6
    model.sit_wrapped_loss_weight = "none"
    model.sit_model_time_scale = 1.0
    model.sit_model_time_flip = False

    # Ground-truth-anchored on-policy post-training. `sit_gt_on_lambda` is the
    # weight on the plain FM anchor: leave it None to keep the plain SiT loss.
    model.sit_gt_on_lambda = None
    # "lambda" = the convex mix above. "additive" = loss_fm + w * loss_corr, for
    # the case where the corrective term is a small auxiliary rather than half
    # the objective; `sit_gt_on_lambda` is then unused. Kept as a string because
    # a None-valued config field cannot be overridden from the command line.
    model.sit_gt_on_mix = "lambda"
    model.sit_gt_on_aux_weight = 0.0
    # t' is drawn from the ordinary FM time distribution squeezed into
    # [t0, (1 - delta) * t1], keeping the target divisor 1 - t' >= delta.
    model.sit_gt_on_t_delta = 0.2
    # "data" = the method (real endpoint); "self" = self-endpoint contrast
    # (cheap variant only); "self_velocity" = local velocity consistency, the
    # detached velocity one Euler step back (rollout variant only). Holding the
    # state construction fixed and switching between "data" and "self_velocity"
    # is what separates state exposure from ground-truth anchoring.
    model.sit_gt_on_target = "data"
    # Rollout depth: 0 = cheap endpoint-reconstruction variant (2 forwards/step),
    # K >= 1 = K detached Euler steps of nominal size `sit_gt_on_rollout_dt`
    # along the model's own dynamics. The draft studies K in {1, 2, 4}.
    model.sit_gt_on_rollout_k = 0
    model.sit_gt_on_rollout_dt = 0.1
    # Which step the rollout takes. "euler" + omega 1.0 is the original
    # construction and the default, so the arms already recorded keep their
    # meaning. "heun" + omega 1.5 makes each rollout step the *inference* step,
    # so the state is one true sampler step off the interpolant. Set these to
    # sampling.method / sampling.omega, and sit_gt_on_rollout_dt to
    # 1 / sampling.num_steps, for a rollout step that matches evaluation.
    model.sit_gt_on_rollout_solver = "euler"
    model.sit_gt_on_rollout_omega = 1.0
    # "perturb" = the constructions above, which perturb the true interpolant.
    # "interp" = no perturbation: the true interpolant at t' itself. With
    # sit_gt_on_target="fm_velocity" the corrective term is exactly plain flow
    # matching on the corrective branch's own t' draw, which is the paired
    # control for the rollout arms (same times, same target, rollout removed).
    # "trajectory" = Denoising Resampling Forcing: integrate the inference
    # schedule from pure noise and supervise at one of its steps. The traj_*
    # keys must mirror sampling.num_steps / method / omega for the training
    # states to be the ones evaluation actually visits.
    # "schedule" = the true interpolant at the same schedule times, i.e. the
    # trajectory arm with the rollout removed. With sit_gt_on_target=
    # "fm_velocity" it is exactly plain flow matching restricted to the
    # inference timesteps, which is the control for the time distribution.
    model.sit_gt_on_state = "perturb"
    model.sit_gt_on_traj_steps = 16
    model.sit_gt_on_traj_solver = "heun"
    model.sit_gt_on_traj_omega = 1.5
    model.sit_gt_on_traj_index_min = 0

    # Training Dynamics
    model.norm_p = 1.0
    model.norm_eps = 0.01

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_steps = 1
    sampling.num_classes = dataset.num_classes
    sampling.meanflow_reverse_time = False
    sampling.method = "euler"
    sampling.flip_time = False
    sampling.eval_modes = ()
    sampling.half_precision = False
    sampling.half_precision_dtype = "float16"
    sampling.native_velocity_cfg_space = "epsilon"
    sampling.native_velocity_derivative_mode = "finite_difference"
    sampling.native_velocity_sigma_clamp = 1e-6
    sampling.transport_velocity_cfg_space = "velocity"
    sampling.transport_velocity_time_map = "noise_ratio"
    sampling.transport_velocity_eps = 1e-3
    sampling.transport_velocity_scale_input = True

    # ------------------------------------------------------------
    # Plain SiT transport
    config.transport = transport = ml_collections.ConfigDict()
    transport.path_type = "Linear"
    transport.prediction = "velocity"
    transport.loss_weight = None
    transport.train_eps = None
    transport.sample_eps = None
    transport.objective = "sit"
    transport.path_power_k = 1.0

    # ------------------------------------------------------------
    # Original DiT diffusion
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.diffusion_steps = 1000
    diffusion.noise_schedule = "linear"
    diffusion.learn_sigma = True
    diffusion.predict_xstart = False
    diffusion.rescale_learned_sigmas = False

    # ------------------------------------------------------------
    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.device_batch_size = 128
    fid.sample_device_batch_size = -1
    fid.sample_log_every = 1
    fid.sample_first_device_only = False
    fid.sample_num_local_devices = 0
    fid.cache_ref = "FID_CACHE_REF"
    fid.num_images_to_log = 100

    config.fd_dino = fd_dino = ml_collections.ConfigDict()
    fd_dino.arch = "vitb14"
    fd_dino.model_name = ""
    fd_dino.cache_ref = ""

    # ------------------------------------------------------------
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.use_wandb = False
    logging.wandb_name = ""
    logging.wandb_project = ""
    logging.wandb_entity = ""
    logging.wandb_notes = ""
    logging.wandb_group = ""
    logging.wandb_tags = []
    logging.wandb_max_retries = 3
    logging.wandb_retry_cooldown_seconds = 300
    logging.wandb_eval_replay_buffer_size = 100

    # others
    config.load_from = ""
    config.partial_load = False
    # Set False when the source run had `use_ema: false`, in which case its
    # `ema_params` are the pre-training initialisation rather than a trained
    # average and restoring them silently discards the training.
    config.prefer_ema = True
    config.eval_only = False

    return config
