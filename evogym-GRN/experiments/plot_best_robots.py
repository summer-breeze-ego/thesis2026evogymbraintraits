#!/usr/bin/env python3
"""
Render a grid figure of the best robot from each run.
Layout: 2 rows (original top, TF6/7 bottom) x 10 columns (runs).
Output: tmp_out/thesis/figures/best_robots.png
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from algorithms.GRN_2D import GRN

_C = {
    1: np.array([0.40, 0.40, 0.40]),   # bone:           dark grey
    2: np.array([0.95, 0.80, 0.10]),   # fat:            amber
    3: np.array([0.85, 0.18, 0.12]),   # phase_muscle:   red
    4: np.array([0.12, 0.30, 0.85]),   # offphase_muscle: blue
}

CONDITIONS = [
    ("cmp500_original", "Original",  0),
    ("cmp500_tf67",     "TF6/7",     1),
]
N_RUNS   = 10
OUT_PATH = REPO_ROOT / "tmp_out" / "thesis" / "figures" / "best_robots.png"


def load_best_robot(db_path):
    con = sqlite3.connect(db_path)
    row = con.execute("""
        SELECT r.robot_id, r.genome, r.displacement
        FROM generation_survivors s
        JOIN all_robots r ON s.robot_id = r.robot_id
        WHERE r.displacement > -1e30
        ORDER BY r.displacement DESC
        LIMIT 1
    """).fetchone()
    con.close()
    return row  # (robot_id, genome_json, displacement) or None


def develop(genome_json, use_brain):
    genome = json.loads(genome_json)
    grn = GRN(promoter_threshold=0.95, max_voxels=27, cube_face_size=4,
              voxel_types="withbone", genotype=list(genome))
    grn.develop()
    phenotype = np.zeros(grn.phenotype.shape, dtype=int)
    for idx, cell in np.ndenumerate(grn.phenotype):
        phenotype[idx] = cell.voxel_type if cell != 0 else 0
    phase_map = grn.phase_map if use_brain else None
    return phenotype, phase_map


def draw_robot(ax, phenotype, phase_map, title=""):
    H, W = phenotype.shape
    ax.set_facecolor("#f0f0f0")
    for gy in range(H):
        for gx in range(W):
            vt = int(phenotype[gy, gx])
            if vt == 0:
                continue
            sy = H - 1 - gy
            color = _C[vt].copy()
            rect = plt.Rectangle(
                (gx + 0.06, sy + 0.06), 0.88, 0.88,
                facecolor=color, edgecolor="#333333", linewidth=0.5,
            )
            ax.add_patch(rect)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=3)


COLS = 5   # robots per row; each condition gets 2 rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(CONDITIONS) * 2          # 4 rows total
    fig, axes = plt.subplots(
        n_rows, COLS,
        figsize=(COLS * 2.2, n_rows * 2.4),
    )

    for cond_i, (experiment, cond_label, use_brain) in enumerate(CONDITIONS):
        for run in range(1, N_RUNS + 1):
            # first 5 runs → top sub-row, next 5 → bottom sub-row
            sub_row = (run - 1) // COLS
            col     = (run - 1) %  COLS
            row_i   = cond_i * 2 + sub_row
            ax = axes[row_i, col]

            db_path = (REPO_ROOT / "tmp_out" / "thesis" / experiment
                       / f"run_{run}" / f"run_{run}")
            result = load_best_robot(db_path)
            if result is None:
                ax.axis("off")
                continue
            robot_id, genome_json, disp = result
            phenotype, phase_map = develop(genome_json, use_brain)
            draw_robot(ax, phenotype, phase_map,
                       title=f"run {run}  d={disp:.1f}")

        pass  # condition labels added below after tight_layout

    # Legend
    legend_elements = [
        mpatches.Patch(color=_C[1], label="Bone"),
        mpatches.Patch(color=_C[2], label="Fat"),
        mpatches.Patch(color=_C[3], label="Phase muscle"),
        mpatches.Patch(color=_C[4], label="Offphase muscle"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Best evolved robots per run", fontsize=16, y=1.01)
    fig.tight_layout(pad=0.5)

    # Add condition labels using figure coordinates (after tight_layout)
    for cond_i, (_, cond_label, _) in enumerate(CONDITIONS):
        ax_top = axes[cond_i * 2,     0]
        ax_bot = axes[cond_i * 2 + 1, 0]
        bbox_top = ax_top.get_position()
        bbox_bot = ax_bot.get_position()
        y_mid  = (bbox_top.y0 + bbox_top.y1 + bbox_bot.y0 + bbox_bot.y1) / 4
        x_left = bbox_top.x0 - 0.02
        fig.text(x_left, y_mid, cond_label,
                 fontsize=13, fontweight="bold",
                 ha="right", va="center", rotation=90)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
