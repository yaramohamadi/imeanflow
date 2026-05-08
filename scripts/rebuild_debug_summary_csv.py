#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SUMMARY_FIELDNAMES = [
    "mode",
    "num_steps",
    "fid",
    "is",
    "eval_metrics_csv",
    "workdir",
    "checkpoint_path",
    "fid_cache_ref",
]

WORKDIR_RE = re.compile(r"^(?P<mode>.+)_(?P<num_steps>\d+)steps$")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild debug summary.csv files from per-step eval_metrics.csv files."
        )
    )
    parser.add_argument(
        "summary_roots",
        nargs="+",
        type=Path,
        help="Directories that contain per-step subdirectories with eval_metrics.csv.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=repo_root / "files/weights/DiT-XL-2-256x256.pt",
        help="Checkpoint path to record in the rebuilt summary rows.",
    )
    parser.add_argument(
        "--fid-cache-ref",
        type=Path,
        default=repo_root / "files/fid_stats/imagenet_256_fid_stats.npz",
        help="FID cache reference path to record in the rebuilt summary rows.",
    )
    return parser.parse_args()


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_eval_row(eval_csv: Path, expected_num_steps: int) -> dict[str, str]:
    with eval_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {eval_csv}")

    matching_rows = []
    for row in rows:
        raw_num_steps = row.get("sampling_num_steps", "").strip()
        if not raw_num_steps:
            continue
        try:
            row_num_steps = int(float(raw_num_steps))
        except ValueError:
            continue
        if row_num_steps == expected_num_steps:
            matching_rows.append(row)

    return matching_rows[-1] if matching_rows else rows[-1]


def build_summary_row(
    *,
    repo_root: Path,
    eval_csv: Path,
    checkpoint_path: Path,
    fid_cache_ref: Path,
) -> dict[str, str]:
    match = WORKDIR_RE.match(eval_csv.parent.name)
    if match is None:
        raise ValueError(
            f"Expected workdir name like '<mode>_<num_steps>steps', got {eval_csv.parent.name}"
        )

    mode = match.group("mode")
    num_steps = int(match.group("num_steps"))
    eval_row = load_eval_row(eval_csv, num_steps)

    fid = eval_row.get("fid", "").strip()
    is_score = eval_row.get("inception_score", "").strip()
    if not fid or not is_score:
        raise ValueError(f"Missing fid or inception_score in {eval_csv}")

    return {
        "mode": mode,
        "num_steps": str(num_steps),
        "fid": fid,
        "is": is_score,
        "eval_metrics_csv": to_repo_relative(eval_csv, repo_root),
        "workdir": to_repo_relative(eval_csv.parent, repo_root),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "fid_cache_ref": str(fid_cache_ref.resolve()),
    }


def rebuild_summary(
    *,
    repo_root: Path,
    summary_root: Path,
    checkpoint_path: Path,
    fid_cache_ref: Path,
) -> tuple[Path, int]:
    summary_root = summary_root.resolve()
    eval_csvs = sorted(summary_root.glob("*/eval_metrics.csv"))
    if not eval_csvs:
        raise ValueError(f"No eval_metrics.csv files found under {summary_root}")

    rows = [
        build_summary_row(
            repo_root=repo_root,
            eval_csv=eval_csv,
            checkpoint_path=checkpoint_path,
            fid_cache_ref=fid_cache_ref,
        )
        for eval_csv in eval_csvs
    ]
    rows.sort(key=lambda row: int(row["num_steps"]))

    summary_csv = summary_root / "summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return summary_csv, len(rows)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    for summary_root in args.summary_roots:
        summary_csv, row_count = rebuild_summary(
            repo_root=repo_root,
            summary_root=summary_root,
            checkpoint_path=args.checkpoint_path,
            fid_cache_ref=args.fid_cache_ref,
        )
        print(f"{summary_csv}: wrote {row_count} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
