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
    training.preview_guidance_scales = ()
    training.debug_log_during_train = False
    training.debug_log_images = True
    training.debug_num_images = 4
    training.debug_velocity_decode_scale = 0.1
    training.grad_accum_steps = 1
    training.save_best_fid_only = False
    training.best_fid_checkpoint_dir = "best_fid"
    training.capture_source_from_load = False
    training.half_precision = False

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
    model.use_dogfit = False
    model.target_use_null_class = True
    model.source_prediction_space = "v"
    model.source_num_classes = dataset.num_classes
    model.use_auxiliary_v_head = True
    model.use_context_guidance_conditioning = False
    model.use_adaln_guidance_scale_conditioning = False
    model.adaln_guidance_scale_init = "timestep"
    model.use_adaln_condition_mixing = False
    model.decoder_only_guidance_conditioning = False
    model.use_training_guidance = True
    model.training_guidance_interval_strategy = "sampled"
    model.training_guidance_t_min = 0.0
    model.training_guidance_t_max = 1.0
    model.training_guidance_start_step = 0
    model.guidance_scale_strategy = "sampled"
    model.max_sampled_guidance_scale = 8.0
    model.fixed_guidance_scale = 7.5
    model.use_positive_sit_dmf_mf_target = False

    # Training Dynamics
    model.norm_p = 1.0
    model.norm_eps = 0.01

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_steps = 1
    sampling.num_classes = dataset.num_classes
    sampling.meanflow_reverse_time = False

    # ------------------------------------------------------------
    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.device_batch_size = 128
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
    logging.wandb_tags = []

    # others
    config.load_from = ""
    config.partial_load = False
    config.eval_only = False

    return config
