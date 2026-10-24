#!/usr/bin/env bash
# Replay a checkpoint's original eval (its dumped FLAGS.config) at NFE steps with 10k samples,
# dumping raw Inception / DINOv2 features for precision-recall.
# The checkpoint is hard-linked into a fresh dir, so the original run dir is never written.
# Usage: bash scripts/pr_eval_one.sh <repo_dir> <run_dir> <eval_output_log> <nfe> <out_name>
set -euo pipefail
REPO="$1"; RUN_DIR="$2"; EVAL_LOG="$3"; NFE="$4"; NAME="$5"
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
OUT=$MAIN/files/logs/pr_eval/$NAME
CKPT_SRC=$(find "$RUN_DIR/best_fid" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint_*')
[[ $(echo "$CKPT_SRC" | wc -l) -eq 1 ]] || { echo "need exactly one best_fid checkpoint in $RUN_DIR" >&2; exit 2; }
mkdir -p "$OUT/ckpt" "$OUT/work"
[[ -d "$OUT/ckpt/$(basename "$CKPT_SRC")" ]] || cp -al "$CKPT_SRC" "$OUT/ckpt/"
$PY "$MAIN/scripts/pr_extract_config.py" "$EVAL_LOG" "$OUT/eval_config.yml"
echo "repo=$REPO run=$RUN_DIR ckpt=$CKPT_SRC nfe=$NFE" > "$OUT/provenance.txt"
cd "$REPO"
FID_DUMP_FEATURES_PREFIX="$OUT/feats" TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $PY main.py \
    --workdir="$OUT/work" \
    --config="$REPO/configs/load_dump.py:$OUT/eval_config.yml" \
    --config.eval_only=True \
    --config.partial_load=False \
    --config.load_from="$OUT/ckpt/$(basename "$CKPT_SRC")" \
    --config.sampling.num_steps="$NFE" \
    --config.training.force_metric_num_steps="$NFE" \
    --config.fid.num_samples=10000 \
    --config.logging.use_wandb=False \
    2>&1 | tee "$OUT/output.log"
ls -la "$OUT"/feats_*.npy
