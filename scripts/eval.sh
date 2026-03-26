#!/bin/bash

# Usage:
#   CONFIG_MODE=eval bash scripts/eval.sh JOB_NAME
#   CONFIG_MODE=caltech_eval bash scripts/eval.sh JOB_NAME
export LOG_DIR="files/logs"
export CONFIG_MODE="${CONFIG_MODE:-eval}"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:${CONFIG_MODE} \
    2>&1 | tee -a $LOG_DIR/output.log
