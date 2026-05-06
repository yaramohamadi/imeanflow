#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODES="p_sample" \
WORKDIR_ROOT="${WORKDIR_ROOT:-$REPO_ROOT/files/debug/fid5k_plain_dit_psample_imagenet}" \
scripts/sweep_fid5k_plain_dit_modes_imagenet.sh "$@"
