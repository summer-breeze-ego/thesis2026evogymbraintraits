#!/usr/bin/env python3
"""
Worker script: renders one robot to an mp4 via EvoViewer.
Called by render_best_robots.py via subprocess.run — never run directly.

SDL env vars are set before every other import so PyGame initialises
into offscreen mode before it touches the display.
"""
import faulthandler
faulthandler.enable()

import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["SDL_AUDIODRIVER"]  = "dummy"

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from algorithms.GRN_2D import GRN
from simulation.prepare_robot_files import prepare_robot_files
from simulation.simulation_resources import _simulate_one_robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path",         required=True)
    parser.add_argument("--robot_id",        type=int, required=True)
    parser.add_argument("--use_brain",       type=int, required=True)
    parser.add_argument("--video_path",      required=True)
    args = parser.parse_args()

    con = sqlite3.connect(args.db_path)
    row = con.execute(
        "SELECT genome FROM all_robots WHERE robot_id = ?", (args.robot_id,)
    ).fetchone()
    con.close()
    if row is None:
        print(f"ERROR: robot_id={args.robot_id} not found", file=sys.stderr)
        sys.exit(1)
    genome = json.loads(row[0])

    grn = GRN(
        promoter_threshold=0.95,
        max_voxels=27,
        cube_face_size=4,
        voxel_types="withbone",
        genotype=list(genome),
    )
    grn.develop()

    phenotype = np.zeros(grn.phenotype.shape, dtype=int)
    for idx, cell in np.ndenumerate(grn.phenotype):
        phenotype[idx] = cell.voxel_type if cell != 0 else 0

    ind = SimpleNamespace(
        id                = args.robot_id,
        phenotype         = phenotype,
        grn_phase_map     = grn.phase_map     if args.use_brain else None,
        grn_amplitude_map = grn.amplitude_map if args.use_brain else None,
    )
    prepare_robot_files(ind, SimpleNamespace(voxel_types="withbone"))

    ctrl = ind.evogym_controller
    task = {
        "id":                ind.id,
        "structure":         ind.evogym_structure,
        "connections":       ind.evogym_connections,
        "phase_offsets":     ind.evogym_phase_offsets,
        "amplitude_offsets": ind.evogym_amplitude_offsets,
        "action_bias":       ctrl["action_bias"],
        "action_amplitude":  ctrl["action_amplitude"],
        "period_steps":      ctrl["period_steps"],
        "sim_steps":         500,
        "init_x":            3,
        "init_y":            1,
        "headless":          0,
        "render_mode":       "rgb_array",
        "video_path":        args.video_path,
        "video_fps":         50,
    }

    rid, disp, err = _simulate_one_robot(task)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"displacement={disp:.6f}")


if __name__ == "__main__":
    main()
