#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

module load python/3.10.13 cuda/12.2
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/$USER-matplotlib

python - <<'PY'
import jax
import torch
import diffusers
import transformers
import timm
import matplotlib
import wandb

print("jax", jax.__version__)
print("torch", torch.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("timm", timm.__version__)
print("matplotlib", matplotlib.__version__)
print("wandb", wandb.__version__)
print("jax_devices", jax.devices())
print("torch_cuda_available", torch.cuda.is_available())
print("torch_cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch_cuda_name", torch.cuda.get_device_name(0))

import train
import main
import prepare_dataset

print("repo_imports_ok")
PY
