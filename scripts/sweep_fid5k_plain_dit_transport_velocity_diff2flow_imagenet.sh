#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODES="transport_velocity_diff2flow" \
WORKDIR_ROOT="${WORKDIR_ROOT:-$REPO_ROOT/files/debug/fid5k_dit_transport_velocity_diff2flow_steps}" \
scripts/sweep_fid5k_plain_dit_modes_imagenet.sh "$@"
