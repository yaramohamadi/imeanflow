#!/bin/bash

set -euo pipefail

# This repo was originally developed in a TPU/JAX environment. On Compute
# Canada we prefer the local wheelhouse when it exists, because it provides
# cluster-compatible builds without requiring internet access.
CC_WHEELHOUSE_ROOT="${CC_WHEELHOUSE_ROOT:-/cvmfs/soft.computecanada.ca/custom/python/wheelhouse}"
IS_COMPUTE_CANADA=0
if [ -d "${CC_WHEELHOUSE_ROOT}" ]; then
    IS_COMPUTE_CANADA=1
fi

VENV_DIR="${VENV_DIR:-.venv}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d "${VENV_DIR}" ]; then
        if [ "${IS_COMPUTE_CANADA}" -eq 1 ] && command -v virtualenv >/dev/null 2>&1; then
            virtualenv --no-download "${VENV_DIR}"
        else
            python -m venv "${VENV_DIR}"
        fi
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
fi

# Keep the JAX pin close to the original repo version when possible, but use
# the nearest Compute Canada build when the exact upstream pin is unavailable.
DEFAULT_JAX_VERSION="${JAX_VERSION:-0.4.27}"
DEFAULT_TORCH_VERSION="${TORCH_VERSION:-2.4.0}"

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    GPU_JAX_VERSION="${GPU_JAX_VERSION:-0.4.28}"
    CPU_JAX_VERSION="${CPU_JAX_VERSION:-0.4.34}"
    FLAX_VERSION="${FLAX_VERSION:-0.8.5}"
    OPTAX_VERSION="${OPTAX_VERSION:-0.2.2}"
    CHEX_VERSION="${CHEX_VERSION:-0.1.86}"
    DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.32.2}"
    TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
    export PIP_NO_INDEX="${PIP_NO_INDEX:-1}"
    export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${CC_WHEELHOUSE_ROOT}/gentoo2023/x86-64-v4 ${CC_WHEELHOUSE_ROOT}/gentoo2023/x86-64-v3 ${CC_WHEELHOUSE_ROOT}/gentoo2023/generic ${CC_WHEELHOUSE_ROOT}/generic}"
else
    JAX_VERSION="${JAX_VERSION:-${DEFAULT_JAX_VERSION}}"
    TORCH_VERSION="${TORCH_VERSION:-${DEFAULT_TORCH_VERSION}}"
fi

# Default to the NVIDIA GPU build of JAX. Override with:
#   JAX_PLATFORM=cpu bash scripts/install.sh
#   JAX_PLATFORM=tpu bash scripts/install.sh
JAX_PLATFORM="${JAX_PLATFORM:-gpu}"

case "$JAX_PLATFORM" in
    gpu)
        if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
            python -m pip install --upgrade \
                "jax==${GPU_JAX_VERSION}+computecanada" \
                "jaxlib==${GPU_JAX_VERSION}+cuda12.cudnn89.computecanada"
        else
            python -m pip install --upgrade \
                "jax[cuda12_pip]==${JAX_VERSION}" \
                -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        fi
        ;;
    cpu)
        if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
            python -m pip install --upgrade \
                "jax==${CPU_JAX_VERSION}+computecanada" \
                "jaxlib==${CPU_JAX_VERSION}+computecanada"
        else
            python -m pip install --upgrade "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
        fi
        ;;
    tpu)
        python -m pip install --upgrade \
            "jax[tpu]==${JAX_VERSION}" \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ;;
    *)
        echo "Unsupported JAX_PLATFORM: $JAX_PLATFORM"
        echo "Expected one of: gpu, cpu, tpu"
        exit 1
        ;;
esac

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    python -m pip install --upgrade \
        "flax==${FLAX_VERSION}+computecanada" \
        "absl-py" \
        "chex==${CHEX_VERSION}+computecanada" \
        "diffusers==${DIFFUSERS_VERSION}+computecanada" \
        "matplotlib==3.9.2" \
        "ml-collections" \
        "ml-dtypes==0.5.0" \
        "optax==${OPTAX_VERSION}+computecanada" \
        "pillow" \
        "PyYAML" \
        "requests" \
        "timm" \
        "tqdm" \
        "transformers" \
        "wandb"
else
    python -m pip install --upgrade \
        "flax>=0.8" \
        "jaxlib==${JAX_VERSION}" \
        "absl-py" \
        "cached_property" \
        "clu" \
        "diffusers" \
        "dm-tree" \
        "keras<3" \
        "matplotlib==3.9.2" \
        "ml-collections" \
        "ml-dtypes==0.5.0" \
        "optax" \
        "orbax-checkpoint==0.6.4" \
        "pillow" \
        "PyYAML" \
        "requests" \
        "tensorflow==2.15.0" \
        "tensorflow_datasets" \
        "tensorstore==0.1.67" \
        "timm" \
        "tqdm" \
        "transformers" \
        "wandb"
fi

# PyTorch GPU wheels can pull in a newer cuDNN runtime that conflicts with the
# pinned JAX GPU wheel above. Install CPU-only PyTorch so JAX keeps control of
# the CUDA/cuDNN stack for the main training/eval path.
python -m pip uninstall -y \
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

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    python -m pip install --upgrade \
        "torch==${TORCH_VERSION}+computecanada" \
        "torchvision==${TORCHVISION_VERSION}+computecanada"
else
    python -m pip install --upgrade \
        "torch==${TORCH_VERSION}" \
        "torchvision" \
        --index-url https://download.pytorch.org/whl/cpu
fi
