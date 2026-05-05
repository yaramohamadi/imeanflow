#!/usr/bin/env python3
import csv
from pathlib import Path


DATASET_PATTERNS = {
    "caltech101": [
        "caltech101_SiT_DMF_DogFit_meanflow_taylor_ema_*",
        "caltech_SiT_DMF_DogFit_meanflow_taylor_ema_*",
    ],
    "artbench10": ["artbench10_SiT_DMF_DogFit_meanflow_taylor_ema_*"],
    "cub200": ["cub200_SiT_DMF_DogFit_meanflow_taylor_ema_*"],
    "food101": ["food101_SiT_DMF_DogFit_meanflow_taylor_ema_*"],
    "stanfordcars": ["stanfordcars_SiT_DMF_DogFit_meanflow_taylor_ema_*"],
}

STEP_DIRS = {
    1: "eval_best_fid_1steps",
    2: "eval_best_fid_2steps",
    250: "eval_best_fid_250steps",
}


def find_latest_complete_run(root: Path, patterns):
    candidates = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(pattern) if path.is_dir())
    candidates = sorted(candidates)
    for run_dir in reversed(candidates):
        if all((run_dir / step_dir / "eval_metrics.csv").exists() for step_dir in STEP_DIRS.values()):
            return run_dir
    return None


def read_target_step_row(csv_path: Path, desired_step: int):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if int(float(row["sampling_num_steps"])) == desired_step]
    if not rows:
        raise ValueError(f"No sampling_num_steps={desired_step} row found in {csv_path}")
    return rows[0]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    finetune_root = repo_root / "files" / "logs" / "finetuning"
    summary_root = repo_root / "files" / "logs" / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    run_rows = []

    for dataset, patterns in DATASET_PATTERNS.items():
        run_dir = find_latest_complete_run(finetune_root, patterns)
        if run_dir is None:
            continue
        run_rows.append(
            {
                "dataset": dataset,
                "run_dir": str(run_dir),
            }
        )
        for desired_step, step_dir in STEP_DIRS.items():
            csv_path = run_dir / step_dir / "eval_metrics.csv"
            row = read_target_step_row(csv_path, desired_step)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "run_dir": str(run_dir),
                    "eval_dir": str(run_dir / step_dir),
                    "training_step": row["training_step"],
                    "sampling_num_steps": row["sampling_num_steps"],
                    "omega": row["omega"],
                    "t_min": row["t_min"],
                    "t_max": row["t_max"],
                    "fid": row["fid"],
                    "inception_score": row["inception_score"],
                    "fd_dino": row["fd_dino"],
                    "checkpoint_path": row["checkpoint_path"],
                    "sample_mode": row["sample_mode"],
                }
            )

    summary_csv = summary_root / "sit_dmf_dogfit_ema_final_eval_latest_runs.csv"
    runs_csv = summary_root / "sit_dmf_dogfit_ema_final_eval_latest_run_dirs.csv"

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "run_dir",
                "eval_dir",
                "training_step",
                "sampling_num_steps",
                "omega",
                "t_min",
                "t_max",
                "fid",
                "inception_score",
                "fd_dino",
                "checkpoint_path",
                "sample_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "run_dir"])
        writer.writeheader()
        writer.writerows(run_rows)

    print(summary_csv)
    print(runs_csv)


if __name__ == "__main__":
    main()
