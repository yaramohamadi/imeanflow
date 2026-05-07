#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-files/logs/finetuning}"
MODE="dry-run"

if [[ "${1:-}" == "--delete" ]]; then
  MODE="delete"
  shift
fi

if [[ "${1:-}" == "--root" ]]; then
  ROOT="$2"
  shift 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: finetuning root not found: $ROOT" >&2
  exit 1
fi

python3 - "$ROOT" "$MODE" <<'PY'
import glob
import os
import shutil
import subprocess
import sys

root = os.path.abspath(sys.argv[1])
cwd = os.getcwd()
mode = sys.argv[2]

known_error_markers = (
    "Traceback",
    "FATAL Flags parsing error",
    "Too many command-line arguments.",
    "No such file or directory",
    "ScopeParamShapeError",
    "ValueError:",
    "KeyError:",
)

def active_paths():
    try:
        out = subprocess.check_output(
            'ps -fu "$USER"',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    active = set()
    for line in out.splitlines():
        if root not in line:
            continue
        for part in line.split():
            candidate = part.strip()
            if root in candidate:
                active.add(os.path.realpath(candidate))
                continue
            if "files/logs/finetuning" in candidate:
                active.add(os.path.realpath(os.path.join(cwd, candidate)))
    return active

active = active_paths()
delete = []
keepers = []
review = []

for path in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(path):
        continue

    log_path = os.path.join(path, "output.log")
    best_ckpt = bool(glob.glob(os.path.join(path, "best_fid", "checkpoint_*")))
    train_eval = os.path.exists(os.path.join(path, "eval_metrics.csv"))
    final_eval = bool(glob.glob(os.path.join(path, "eval_best_fid_*steps", "eval_metrics.csv")))
    dir_active = any(
        ap == os.path.realpath(path) or ap.startswith(os.path.realpath(path) + os.sep)
        for ap in active
    )
    final_eval_done = False

    reason = None
    if dir_active:
        keepers.append((path, "active"))
        continue

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        if not best_ckpt and not train_eval and not final_eval:
            reason = "empty_or_missing_log"
    else:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
        has_error = any(marker in text for marker in known_error_markers)
        final_eval_done = "=== FINAL BEST_FID EVAL DONE ===" in text
        if has_error and not best_ckpt and not train_eval and not final_eval:
            reason = "error_without_artifacts"

    if reason:
        delete.append((path, reason))
    else:
        if final_eval_done:
            keepers.append((path, "final_eval_done"))
        elif final_eval and best_ckpt and train_eval:
            keepers.append((path, "has_final_eval_artifacts"))
        elif best_ckpt or train_eval or final_eval:
            review.append((path, "partial_artifacts_review"))
        else:
            review.append((path, "nonempty_log_review"))

print(f"Mode: {mode}")
print(f"Root: {root}")
print()
print("Delete candidates:")
for path, reason in delete:
    print(f"  {os.path.basename(path)}\t{reason}")

print()
print("Keepers:")
for path, reason in keepers:
    print(f"  {os.path.basename(path)}\t{reason}")

print()
print("Manual review:")
for path, reason in review:
    print(f"  {os.path.basename(path)}\t{reason}")

if mode == "delete":
    for path, _ in delete:
        shutil.rmtree(path)
    print()
    print(f"Deleted {len(delete)} directories.")
else:
    print()
    print(
        f"Dry run only. Would delete {len(delete)} directories, "
        f"keep {len(keepers)}, and leave {len(review)} for manual review."
    )
PY
