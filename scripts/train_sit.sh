#!/bin/bash

# Usage:
#   CONFIG_MODE=sit_train bash scripts/train_sit.sh JOB_NAME
#   CONFIG_MODE=sit_train bash scripts/train_sit.sh JOB_NAME --config.load_from=/path/to/SiT-XL-2-256.pt --config.partial_load=True
export LOG_DIR="${LOG_DIR:-files/logs}"
export CONFIG_MODE="${CONFIG_MODE:-sit_train}"
export DATASET_NAME="${DATASET_NAME:-${CONFIG_MODE%_train}}"

if [[ $# -lt 1 ]]; then
  echo "Usage: CONFIG_MODE=sit_train bash scripts/train_sit.sh JOB_NAME [extra main_sit.py args...]"
  exit 1
fi

export RUN_LABEL="${1}"
shift
EXTRA_ARGS=("$@")

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${DATASET_NAME}_${RUN_LABEL}_${now}_${salt}
export LOG_DIR=$LOG_DIR/finetuning/$JOBNAME

mkdir -p "${LOG_DIR}"

python3 main_sit.py \
    --workdir="${LOG_DIR}" \
    --config=configs/load_config.py:${CONFIG_MODE} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | stdbuf -oL grep -a -v -E '(\+ptx[0-9]+|recognized feature for this target|ignoring feature)' | tee -a "${LOG_DIR}/output.log"
