#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export MODES="${MODES:-transport_velocity_noalign}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-fid5k_dit_transport_velocity_noalign_steps}"
export WORKDIR_ROOT="${WORKDIR_ROOT:-$REPO_ROOT/files/debug/fid5k_dit_transport_velocity_noalign_steps}"

STEPS=("${@}")
if [[ ${#STEPS[@]} -eq 0 ]]; then
  STEPS=(64 250)
fi

exec bash "$REPO_ROOT/scripts/sweep_fid5k_plain_dit_modes_imagenet.sh" "${STEPS[@]}"
