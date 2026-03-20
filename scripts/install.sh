#!/bin/bash

set -euo pipefail

# This project's dependency set fits Python 3.9 better than the system's 3.13.
# Keep JAX pinned close to the original repo version for compatibility.
JAX_VERSION="${JAX_VERSION:-0.4.27}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"

# Default to the NVIDIA GPU build of JAX. Override with:
#   JAX_PLATFORM=cpu bash scripts/install.sh
#   JAX_PLATFORM=tpu bash scripts/install.sh
JAX_PLATFORM="${JAX_PLATFORM:-gpu}"

case "$JAX_PLATFORM" in
    gpu)
        # Use the CUDA wheel path that matches the repo's original JAX pin.
        pip install --upgrade "jax[cuda12_pip]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ;;
    cpu)
        pip install --upgrade "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
        ;;
    tpu)
        pip install --upgrade "jax[tpu]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ;;
    *)
        echo "Unsupported JAX_PLATFORM: $JAX_PLATFORM"
        echo "Expected one of: gpu, cpu, tpu"
        exit 1
        ;;
esac

pip install --upgrade "flax>=0.8" "jaxlib==${JAX_VERSION}"

# PyTorch GPU wheels can pull in a newer cuDNN runtime that conflicts with the
# pinned JAX GPU wheel above. Install CPU-only PyTorch so JAX keeps control of
# the CUDA/cuDNN stack for the main training/eval path.
pip uninstall -y \
    torch \
    torchvision \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-cuda-nvcc-cu12 \
    triton || true

pip install --upgrade pillow clu tensorflow==2.15.0 "keras<3" tensorflow_datasets matplotlib==3.9.2
pip install --upgrade "torch==${TORCH_VERSION}" "torchvision" --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade orbax-checkpoint==0.6.4 ml-dtypes==0.5.0 tensorstore==0.1.67
pip install --upgrade diffusers dm-tree cached_property
pip install --upgrade wandb
