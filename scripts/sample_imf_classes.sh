#!/usr/bin/env bash
# Usage: bash scripts/sample_imf_classes.sh <cub200|stanfordcars> <checkpoint_dir> <classes csv> <out.npz> [gpu] [nfe] [n_per_class] [seed]
set -euo pipefail
ds=$1 ck=$2 cls=$3 out=$4 gpu=${5:-0} nfe=${6:-4} n=${7:-4} seed=${8:-0}
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; PY=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python
case $ds in cub200) LAT=cub-200-2011_processed_latents NC=200;; stanfordcars) LAT=stanford-cars_processed_latents NC=196;; esac
cd $ADV
CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
$PY scripts/sample_imf_classes.py --config=configs/load_config.py:caltech_imf_caimf_posttrain \
  --config.load_from=$ck --config.dataset.name=${ds}_latent --config.dataset.root=$ADV/../datasets/$LAT \
  --config.dataset.class_mapping_root= --config.dataset.num_classes=$NC --config.model.num_classes=$NC \
  --config.sampling.num_classes=$NC --classes=$cls --n_per_class=$n --num_steps=$nfe --seed=$seed --out=$out
