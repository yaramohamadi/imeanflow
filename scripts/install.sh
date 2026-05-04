#!/bin/bash

set -euo pipefail

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

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
DEFAULT_FLAX_VERSION="${FLAX_VERSION:-0.8.5}"
DEFAULT_OPTAX_VERSION="${OPTAX_VERSION:-0.2.2}"
DEFAULT_CHEX_VERSION="${CHEX_VERSION:-0.1.86}"
DEFAULT_ORBAX_CHECKPOINT_VERSION="${ORBAX_CHECKPOINT_VERSION:-0.6.4}"
DEFAULT_CLU_VERSION="${CLU_VERSION:-0.0.11}"

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    GPU_JAX_VERSION="${GPU_JAX_VERSION:-0.4.28}"
    CPU_JAX_VERSION="${CPU_JAX_VERSION:-0.4.34}"
    FLAX_VERSION="${FLAX_VERSION:-0.8.5}"
    OPTAX_VERSION="${OPTAX_VERSION:-0.2.2}"
    CHEX_VERSION="${CHEX_VERSION:-0.1.86}"
    ORBAX_CHECKPOINT_VERSION="${ORBAX_CHECKPOINT_VERSION:-0.6.4}"
    DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.32.2}"
    TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.49.0}"
    TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
    export PIP_NO_INDEX="${PIP_NO_INDEX:-1}"
    export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${CC_WHEELHOUSE_ROOT}/gentoo2023/x86-64-v4 ${CC_WHEELHOUSE_ROOT}/gentoo2023/x86-64-v3 ${CC_WHEELHOUSE_ROOT}/gentoo2023/generic ${CC_WHEELHOUSE_ROOT}/generic}"
else
    JAX_VERSION="${JAX_VERSION:-${DEFAULT_JAX_VERSION}}"
    TORCH_VERSION="${TORCH_VERSION:-${DEFAULT_TORCH_VERSION}}"
    FLAX_VERSION="${FLAX_VERSION:-${DEFAULT_FLAX_VERSION}}"
    OPTAX_VERSION="${OPTAX_VERSION:-${DEFAULT_OPTAX_VERSION}}"
    CHEX_VERSION="${CHEX_VERSION:-${DEFAULT_CHEX_VERSION}}"
    ORBAX_CHECKPOINT_VERSION="${ORBAX_CHECKPOINT_VERSION:-${DEFAULT_ORBAX_CHECKPOINT_VERSION}}"
    CLU_VERSION="${CLU_VERSION:-${DEFAULT_CLU_VERSION}}"
fi

# Python 3.12 cannot use the original TensorFlow 2.15 pin from this repo.
# These defaults keep TensorFlow in the same environment while remaining
# compatible with the newer ml-dtypes stack used by JAX.
TENSORFLOW_VERSION="${TENSORFLOW_VERSION:-2.19.1}"
ML_DTYPES_VERSION="${ML_DTYPES_VERSION:-0.5.1}"

# Default to the NVIDIA GPU build of JAX. Override with:
#   JAX_PLATFORM=cpu bash scripts/install.sh
#   JAX_PLATFORM=tpu bash scripts/install.sh
JAX_PLATFORM="${JAX_PLATFORM:-gpu}"

install_jax() {
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
}

install_jax

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    python -m pip install --upgrade \
        "flax==${FLAX_VERSION}+computecanada" \
        "absl-py" \
        "chex==${CHEX_VERSION}+computecanada" \
        "diffusers==${DIFFUSERS_VERSION}+computecanada" \
        "matplotlib==3.9.2" \
        "ml-collections" \
        "ml-dtypes==${ML_DTYPES_VERSION}" \
        "optax==${OPTAX_VERSION}+computecanada" \
        "orbax-checkpoint==${ORBAX_CHECKPOINT_VERSION}+computecanada" \
        "pillow" \
        "PyYAML" \
        "requests" \
        "timm" \
        "tqdm" \
        "transformers==${TRANSFORMERS_VERSION}+computecanada" \
        "wandb"

    # Some Compute Canada wheels have broad JAX requirements and may pull in a
    # newer CPU-only jaxlib during the dependency install above. Re-apply the
    # requested JAX platform pin so GPU jobs do not silently fall back to CPU.
    install_jax
else
    python -m pip install --upgrade \
        "flax==${FLAX_VERSION}" \
        "absl-py" \
        "cached_property" \
        "chex==${CHEX_VERSION}" \
        "clu==${CLU_VERSION}" \
        "diffusers" \
        "dm-tree" \
        "matplotlib==3.9.2" \
        "ml-collections" \
        "ml-dtypes==${ML_DTYPES_VERSION}" \
        "optax==${OPTAX_VERSION}" \
        "orbax-checkpoint==${ORBAX_CHECKPOINT_VERSION}" \
        "pillow" \
        "PyYAML" \
        "requests" \
        "tensorflow==${TENSORFLOW_VERSION}" \
        "tensorflow_datasets" \
        "tensorstore==0.1.67" \
        "timm" \
        "tqdm" \
        "transformers" \
        "wandb"

    # Some upstream wheels have broad JAX requirements and may pull in a newer
    # CPU-only jaxlib during the dependency install above. Re-apply the
    # requested JAX platform pin so GPU jobs do not silently fall back to CPU.
    install_jax
fi

# PyTorch GPU wheels can pull in a newer cuDNN runtime that conflicts with the
# pinned JAX GPU wheel above. Install CPU-only PyTorch so JAX keeps control of
# the CUDA/cuDNN stack for the main training/eval path.
#
# For the upstream `jax[cuda12_pip]` path, JAX itself depends on the pip
# `nvidia-*` packages that provide CUDA/cuDNN userspace libraries, so we must
# not remove them here. The Compute Canada path uses cluster-provided JAX wheels
# instead, so there it is safe to clear the NVIDIA pip packages.
python -m pip uninstall -y torch torchvision triton || true

if [ "${IS_COMPUTE_CANADA}" -eq 1 ]; then
    python -m pip uninstall -y \
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
        nvidia-cuda-nvcc-cu12 || true
fi

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
