#!/usr/bin/env bash
# Class-conditional samples for qualitative figures, same configs as the final evals.
# Usage: bash scripts/sample_imf_classes.sh <imf|sit|dit|jit> <cub200|stanfordcars> <checkpoint_dir> <classes csv> <out.npz> [gpu] [nfe] [n_per_class] [seed]
set -euo pipefail
bb=$1 ds=$2 ck=$3 cls=$4 out=$5 gpu=${6:-0} nfe=${7:-4} n=${8:-4} seed=${9:-0}
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; MAIN=/opt/dlami/nvme/meanflow/imeanflow; PY=$MAIN/.venv/bin/python
DATA=/opt/dlami/nvme/meanflow/datasets
case $ds in cub200) LAT=cub-200-2011_processed_latents IMG=cub-200-2011_images NC=200;;
  stanfordcars) LAT=stanford-cars_processed_latents IMG=stanford-cars_images NC=196;; esac
REPO=$ADV EXTRA=()
case $bb in
  imf) MODE=caltech_imf_caimf_posttrain;;
  sit) MODE=caltech_sit_meft_caimf_posttrain;;
  dit) MODE=caltech_dit_meft_caimf_posttrain;;
  # JiT CAMF trained with EMA and Tab. 2 reports its EMA FID; caltech_jit_dmf_meft defaults to online.
  jit) MODE=caltech_jit_dmf_meft REPO=$MAIN EXTRA=(--pixel --config.dataset.num_classes_from_data=False --config.training.use_ema=True);;
esac
if [ $bb = jit ]; then DS=(--config.dataset.root=$DATA/$IMG)
else DS=(--config.dataset.name=${ds}_latent --config.dataset.root=$DATA/$LAT --config.dataset.class_mapping_root=
         --config.sampling.num_classes=$NC); fi
cd $REPO
REPO=$REPO CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
$PY $ADV/scripts/sample_imf_classes.py --config=$REPO/configs/load_config.py:$MODE \
  --config.load_from=$ck "${DS[@]}" --config.dataset.num_classes=$NC --config.model.num_classes=$NC \
  "${EXTRA[@]}" --classes=$cls --n_per_class=$n --num_steps=$nfe --seed=$seed --out=$out
