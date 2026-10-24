"""Load a full config dumped by a previous run's log (FLAGS.config YAML) as a ConfigDict.

Usage: --config=configs/load_dump.py:/path/to/eval_config.yml
Lets an eval be replayed with exactly the config it ran with; pass overrides as flags.
"""
import ml_collections
import yaml


def get_config(path):
    with open(path) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return ml_collections.ConfigDict(config_dict)
