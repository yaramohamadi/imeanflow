#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_MODE="${CONFIG_MODE:-caltech_dit_dmf_ddpmv}"
ENABLE_DOGFIT="${ENABLE_DOGFIT:-False}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 16 250}"

export CONFIG_MODE
export ENABLE_DOGFIT
export FINAL_EVAL_STEPS

bash "${SCRIPT_DIR}/run_caltech_dit_dmf_meanflow_taylor.sh" "$@"
