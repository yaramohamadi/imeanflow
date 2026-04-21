#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export NUM_STEPS="${NUM_STEPS:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-files/logs/jit_imagenet_h16_${NUM_STEPS}step_sample_$(date +%Y%m%d_%H%M%S)}"

bash scripts/sample_jit_imagenet_h16_50step_taylor.sh "$@"
