#!/usr/bin/env bash
#SBATCH --job-name=imf_smoke10
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=files/logs/smoke/imf_smoke10_%j.out
#SBATCH --error=files/logs/smoke/imf_smoke10_%j.err

set -euo pipefail

cd /project/6086105/ymbahram/imeanflow
module load python/3.10.13 cuda/12.2
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

bash scripts/smoke_plain_imf_local.sh "${RUN_LABEL:-slurm_smoke10}"
