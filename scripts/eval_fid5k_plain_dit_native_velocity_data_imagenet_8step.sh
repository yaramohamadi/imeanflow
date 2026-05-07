#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKDIR_ROOT="${WORKDIR_ROOT:-$REPO_ROOT/files/debug/fid5k_plain_dit_native_velocity_data_imagenet}" \
scripts/sweep_fid5k_plain_dit_native_velocity_data_imagenet.sh 8
