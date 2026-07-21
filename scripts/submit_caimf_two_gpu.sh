#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   LAMBDA_OT=0.001 MAX_POSTTRAIN_BATCHES=155000 DISCRIMINATOR_WARMUP_BATCHES=5000 \
#     bash scripts/submit_caimf_two_gpu.sh 1 0,1 4
#   LAMBDA_OT=0.001 MAX_POSTTRAIN_BATCHES=155000 DISCRIMINATOR_WARMUP_BATCHES=5000 \
#     bash scripts/submit_caimf_two_gpu.sh 2 2,3 4
#
# The third argument is discriminator steps per cycle. It defaults to 4 for
# the post-16D:1G ablation. The discriminator-only warm-up remains 10,000
# batches so this run changes only one experimental variable.

EXPERIMENT="${1:?Provide experiment number: 1, 2, 3, or 5/adv_only}"
GPUS="${2:-1,2}"
DISCRIMINATOR_STEPS_PER_CYCLE="${3:-4}"
DISCRIMINATOR_WARMUP_BATCHES="${DISCRIMINATOR_WARMUP_BATCHES:-10000}"
MAX_POSTTRAIN_BATCHES="${MAX_POSTTRAIN_BATCHES:-40000}"
LAMBDA_OT="${LAMBDA_OT:-0.001}"
LAMBDA_CP="${LAMBDA_CP:-0.001}"

if ! [[ "$DISCRIMINATOR_STEPS_PER_CYCLE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: discriminator steps per cycle must be a positive integer" >&2
    exit 2
fi
if ! [[ "$DISCRIMINATOR_WARMUP_BATCHES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: DISCRIMINATOR_WARMUP_BATCHES must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "$MAX_POSTTRAIN_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_POSTTRAIN_BATCHES must be a positive integer" >&2
    exit 2
fi

case "$EXPERIMENT" in
    1)
        LAMBDA_IMF=0.01
        LAMBDA_ADV=1.0
        ;;
    2)
        LAMBDA_IMF=1.0
        LAMBDA_ADV=1.0
        ;;
    3)
        LAMBDA_IMF=1.0
        LAMBDA_ADV=0.01
        ;;
    5|adv_only)
        EXPERIMENT=5
        LAMBDA_IMF=0.0
        LAMBDA_ADV=1.0
        ;;
    *)
        echo "ERROR: experiment must be 1, 2, 3, or 5/adv_only" >&2
        exit 2
        ;;
esac

REPO=/home/ens/Zdehghani/imeanflow
CHECKPOINT="$REPO/files/weights/iMF-XL-2-full"
DATASET=/home/ens/Zdehghani/datasets/caltech-101_processed_latents

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
    echo "ERROR: Python executable not found: $REPO/.venv/bin/python" >&2
    exit 2
fi
if [[ ! -e "$CHECKPOINT" ]]; then
    echo "ERROR: ImageNet iMF checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -d "$DATASET/train" ]]; then
    echo "ERROR: Caltech latent dataset not found: $DATASET/train" >&2
    exit 2
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
OT_TAG="${LAMBDA_OT//./p}"
CP_TAG="${LAMBDA_CP//./p}"
WORKDIR="$REPO/files/logs/caimf/caltech_imagenet_exp${EXPERIMENT}_ot${OT_TAG}_cp${CP_TAG}_w${DISCRIMINATOR_WARMUP_BATCHES}_d${DISCRIMINATOR_STEPS_PER_CYCLE}_b${MAX_POSTTRAIN_BATCHES}_${STAMP}"
SCREEN_NAME="caimf_imagenet_exp${EXPERIMENT}_ot${OT_TAG}_cp${CP_TAG}_d${DISCRIMINATOR_STEPS_PER_CYCLE}_b${MAX_POSTTRAIN_BATCHES}"

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "ERROR: screen session already exists: $SCREEN_NAME" >&2
    exit 2
fi

mkdir -p "$WORKDIR"

echo "Experiment: $EXPERIMENT"
echo "lambda_iMF: $LAMBDA_IMF"
echo "lambda_adv: $LAMBDA_ADV"
echo "lambda_ot: $LAMBDA_OT"
echo "lambda_cp: $LAMBDA_CP"
echo "Physical GPUs: $GPUS"
echo "Discriminator-only warm-up: $DISCRIMINATOR_WARMUP_BATCHES batches"
echo "Post-warm-up update ratio: ${DISCRIMINATOR_STEPS_PER_CYCLE}D:1G"
echo "Maximum post-training batches: $MAX_POSTTRAIN_BATCHES"
echo "Initial checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"

screen -dmS "$SCREEN_NAME" bash -lc "
cd '$REPO'
CUDA_VISIBLE_DEVICES='$GPUS' \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHON='$REPO/.venv/bin/python' \
IMF_CHECKPOINT='$CHECKPOINT' \
EXPERIMENT='$EXPERIMENT' \
LAMBDA_OT='$LAMBDA_OT' \
LAMBDA_CP='$LAMBDA_CP' \
ENTRY_MODE=imagenet \
WORKDIR='$WORKDIR' \
DATASET_ROOT='$DATASET' \
bash scripts/train_caltech_caimf.sh \
  --config.training.batch_size=2 \
  --config.caimf.discriminator_warmup_batches='$DISCRIMINATOR_WARMUP_BATCHES' \
  --config.caimf.discriminator_steps_per_cycle='$DISCRIMINATOR_STEPS_PER_CYCLE' \
  --config.caimf.max_posttrain_batches='$MAX_POSTTRAIN_BATCHES' \
  --config.logging.use_wandb=False \
  > '$WORKDIR/launcher.log' 2>&1
"

echo "Submitted screen: $SCREEN_NAME"
echo "Monitor with: tail -f $WORKDIR/launcher.log"
