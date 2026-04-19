#!/usr/bin/env bash
set -euo pipefail

exec bash scripts/train_plain_imf_finetune.sh "$@"
