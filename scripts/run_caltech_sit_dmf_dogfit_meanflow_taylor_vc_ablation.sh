#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor_vc_ablation.sh <run_label_prefix> [extra main.py args...]

Example:
  bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor_vc_ablation.sh caltech_dogfit_vc_ablate

This script launches two Taylor-local DogFit meanflow runs:
  1) VC_TARGET_SOURCE=ema
  2) VC_TARGET_SOURCE=online

Both runs keep stop-gradient on v_c and v_u in the DogFit path.
EOF
  exit 1
fi

RUN_LABEL_PREFIX="$1"
shift
EXTRA_ARGS=("$@")

VC_TARGET_SOURCE=ema \
  bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh \
    "${RUN_LABEL_PREFIX}_ema" \
    "${EXTRA_ARGS[@]}"

VC_TARGET_SOURCE=online \
  bash scripts/run_caltech_sit_dmf_dogfit_meanflow_taylor.sh \
    "${RUN_LABEL_PREFIX}_online" \
    "${EXTRA_ARGS[@]}"
