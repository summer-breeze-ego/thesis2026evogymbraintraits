#!/usr/bin/env python3
"""
Render a video of the best robot from each run of the cmp500 comparison.

Calls render_one.py via subprocess.run for each robot so that every
robot gets a fully isolated Python process (SDL + EvoGym init from scratch).

Outputs to tmp_out/thesis/videos/{condition}_run{N:02d}_best.mp4

Usage:
    /opt/miniconda3/envs/evogym/bin/python experiments/render_best_robots.py
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

PYTHON     = sys.executable
REPO_ROOT  = Path(__file__).resolve().parent.parent
WORKER     = Path(__file__).parent / "render_one.py"

STUDY_NAME = "thesis"
OUT_PATH   = REPO_ROOT / "tmp_out"
VIDEO_DIR  = OUT_PATH / STUDY_NAME / "videos"
N_RUNS     = 10

CONDITIONS = {
    "tf67":     {"experiment": "cmp500_tf67",     "use_brain": 1},
    "original": {"experiment": "cmp500_original", "use_brain": 0},
}


def load_best_robot(db_path: Path):
    """Return (robot_id, displacement) for the best-ever survivor in this run."""
    con = sqlite3.connect(db_path)
    row = con.execute("""
        SELECT r.robot_id, r.displacement
        FROM generation_survivors s
        JOIN all_robots r ON s.robot_id = r.robot_id
        WHERE r.displacement > -1e30
        ORDER BY r.displacement DESC
        LIMIT 1
    """).fetchone()
    con.close()
    return row  # (robot_id, displacement) or None


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    for cond_name, cond_cfg in CONDITIONS.items():
        experiment = cond_cfg["experiment"]
        use_brain  = cond_cfg["use_brain"]

        for run in range(1, N_RUNS + 1):
            db_path = OUT_PATH / STUDY_NAME / experiment / f"run_{run}" / f"run_{run}"
            if not db_path.exists():
                print(f"[SKIP] missing: {db_path}")
                continue

            result = load_best_robot(db_path)
            if result is None:
                print(f"[SKIP] no valid robot in {cond_name} run {run}")
                continue

            robot_id, stored_disp = result
            out_path = VIDEO_DIR / f"{cond_name}_run{run:02d}_best.mp4"
            print(f"\n[{cond_name}] run {run:2d}  robot_id={robot_id}  "
                  f"stored_displacement={stored_disp:.3f}")

            proc = subprocess.run(
                [
                    PYTHON, str(WORKER),
                    "--db_path",    str(db_path),
                    "--robot_id",   str(robot_id),
                    "--use_brain",  str(use_brain),
                    "--video_path", str(out_path),
                ],
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                print(f"  [FAIL] exit {proc.returncode}")
                if proc.stderr.strip():
                    print(f"  stderr: {proc.stderr.strip()}")
            else:
                stdout = proc.stdout.strip()
                print(f"  [OK]   {stdout}  -> {out_path.name}")
                if proc.stderr.strip():
                    # EvoGym prints version info to stderr — show it for confirmation
                    print(f"  info:  {proc.stderr.splitlines()[0]}")

    print(f"\nDone. Videos in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
