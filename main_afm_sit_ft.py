"""Entry point: AFM post-training of a target-finetuned plain SiT (FT) checkpoint (Tab. 2 "FT + AFM").

  python main_afm_sit_ft.py --config=configs/load_config.py:sit_ft_afm_tab2 --workdir=<wd> \
      --config.load_from=<FT best_fid/checkpoint_*>                     # train
  python main_afm_sit_ft.py ... --final_eval_ckpt=<wd>/best_fid/checkpoint_*   # eval only
"""

import os

import jax
from absl import app, flags
from ml_collections import config_flags

import train_afm_sit_ft
from utils import logging_util
from utils.logging_util import log_for_0


logging_util.supress_checkpt_info()
FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory for SiT-FT AFM outputs.")
flags.DEFINE_string("final_eval_ckpt", "", "If set: only evaluate this checkpoint (CFG and no-CFG).")
config_flags.DEFINE_config_file("config", None, "SiT-FT AFM configuration.", lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coordinator:
        jax.distributed.initialize(coordinator_address=coordinator)
    log_for_0("JAX local devices: %r", jax.local_devices())
    try:
        if FLAGS.final_eval_ckpt:
            train_afm_sit_ft.final_evaluate(FLAGS.config, FLAGS.workdir, FLAGS.final_eval_ckpt)
        else:
            train_afm_sit_ft.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    finally:
        logging_util.close_wandb()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
