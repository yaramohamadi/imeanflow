#!/bin/bash

# Usage:
#   CONFIG_MODE=train bash scripts/train.sh JOB_NAME
#   CONFIG_MODE=caltech_finetune bash scripts/train.sh JOB_NAME
export DATA_ROOT="YOUR_OUTPUT_DIR_FROM_DATA_PREPARATION"
export LOG_DIR="${LOG_DIR:-files/logs}"
export CONFIG_MODE="${CONFIG_MODE:-train}"
export DATASET_NAME="${DATASET_NAME:-${CONFIG_MODE%_finetune}}"
export RUN_LABEL="${1:-run}"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${DATASET_NAME}_${RUN_LABEL}_${now}_${salt}
export LOG_DIR=$LOG_DIR/finetuning/$JOBNAME

mkdir -p "${LOG_DIR}"

python3 main.py \
    --workdir="${LOG_DIR}" \
    --config=configs/load_config.py:${CONFIG_MODE} \
    2>&1 | stdbuf -oL grep -a -v -E '(\+ptx[0-9]+|recognized feature for this target|ignoring feature)' | tee -a "${LOG_DIR}/output.log"
