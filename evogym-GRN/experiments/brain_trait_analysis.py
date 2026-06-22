#!/usr/bin/env python3
"""
Brain trait correlation analysis for the cmp500 comparison.

Computes Pearson correlations between parent-child brain trait distances
and displacement improvement (Δf), both pooled (original approach) and
per-run (n=10 per condition, fixing pseudo-replication).

Outputs:
  - Console table: pooled r, per-run mean ± std, Mann-Whitney test
  - Figure: box plot of per-run correlations saved to tmp_out/thesis/figures/
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from algorithms.GRN_2D import GRN

# ── config ──────────────────────────────────────────────────────────────────
STUDY_ROOT = REPO_ROOT / "tmp_out" / "thesis"
CONDITIONS = {
    "tf67":     {"experiment": "cmp500_tf67",     "use_brain": True},
    "original": {"experiment": "cmp500_original", "use_brain": False},
}
N_RUNS     = 10
BRAIN_COLS = ["phase_mean", "phase_std", "amp_mean", "amp_std"]
DIST_COLS  = ["brain_dist", "phase_dist", "amp_dist"]


# ── helpers ─────────────────────────────────────────────────────────────────

def brain_summary(genome_list):
    grn = GRN(promoter_threshold=0.95, max_voxels=27, cube_face_size=4,
              voxel_types="withbone", genotype=genome_list)
    grn.develop()
    muscle_types = {grn.structural_products.get("phase_muscle"),
                    grn.structural_products.get("offphase_muscle")}
    mask = np.zeros(grn.phenotype.shape, dtype=bool)
    for idx, cell in np.ndenumerate(grn.phenotype):
        if cell != 0 and cell.voxel_type in muscle_types:
            mask[idx] = True
    if not mask.any():
        return [np.nan] * 4
    phases = grn.phase_map[mask]
    amps   = grn.amplitude_map[mask]
    return [float(phases.mean()), float(phases.std()),
            float(amps.mean()),   float(amps.std())]


def load_condition(cond_name, experiment):
    rows = []
    for run in range(1, N_RUNS + 1):
        db = STUDY_ROOT / experiment / f"run_{run}" / f"run_{run}"
        con = sqlite3.connect(db)
        df = pd.read_sql("""
            SELECT robot_id, genome, parent1_id, parent2_id, displacement
            FROM all_robots
        """, con)
        con.close()
        df["condition"] = cond_name
        df["run"]       = run
        df["uid"]       = df.apply(
            lambda r: f"{cond_name}_{run}_{int(r.robot_id)}", axis=1)
        df["parent1_uid"] = df.apply(
            lambda r: f"{cond_name}_{run}_{int(r.parent1_id)}"
                      if pd.notna(r.parent1_id) else None, axis=1)
        df["parent2_uid"] = df.apply(
            lambda r: f"{cond_name}_{run}_{int(r.parent2_id)}"
                      if pd.notna(r.parent2_id) else None, axis=1)
        rows.append(df)
        print(f"  {cond_name} run {run}: {len(df)} robots", flush=True)
    return pd.concat(rows, ignore_index=True)


def compute_brain_traits(df):
    print("Computing brain traits (takes a few minutes)...", flush=True)
    summaries = df["genome"].apply(
        lambda g: brain_summary(json.loads(g) if isinstance(g, str) else g)
    )
    df = df.copy()
    df[BRAIN_COLS] = pd.DataFrame(summaries.tolist(), index=df.index)
    print("Done.", flush=True)
    return df


def zscore_within_condition(df):
    """Z-score brain cols per condition across all runs (same as paper)."""
    df = df.copy()
    for cond in df["condition"].unique():
        m = df["condition"] == cond
        for col in BRAIN_COLS:
            mu  = df.loc[m, col].mean()
            sig = df.loc[m, col].std()
            df.loc[m, col + "_z"] = (df.loc[m, col] - mu) / sig
    return df


def build_pairs(df, id_lookup):
    """Build parent-child pairs for rows in df using id_lookup for parent data."""
    z_cols = [c + "_z" for c in BRAIN_COLS]
    pairs = []
    for _, child in df.iterrows():
        for puid_col in ["parent1_uid", "parent2_uid"]:
            puid = child[puid_col]
            if pd.isna(puid) or puid not in id_lookup:
                continue
            p = id_lookup[puid]
            c_disp = float(child["displacement"])
            p_disp = float(p["displacement"])
            if not (np.isfinite(c_disp) and np.isfinite(p_disp)):
                continue
            c_vec = np.array([child[c] for c in z_cols], dtype=float)
            p_vec = np.array([p[c]     for c in z_cols], dtype=float)
            if np.any(np.isnan(c_vec)) or np.any(np.isnan(p_vec)):
                continue
            diff = c_vec - p_vec
            pairs.append({
                "brain_dist": np.linalg.norm(diff),
                "phase_dist": np.linalg.norm(diff[:2]),
                "amp_dist":   np.linalg.norm(diff[2:]),
                "delta_f":    c_disp - p_disp,
                "condition":  child["condition"],
                "run":        child["run"],
            })
    return pd.DataFrame(pairs)


# ── analysis ────────────────────────────────────────────────────────────────

def pooled_correlations(pairs_df):
    """Original approach: pool all pairs per condition."""
    rows = []
    for cond in pairs_df["condition"].unique():
        sub = pairs_df[pairs_df["condition"] == cond]
        row = {"condition": cond, "n_pairs": len(sub)}
        for dc in DIST_COLS:
            r, p = pearsonr(sub[dc], sub["delta_f"])
            row[f"r_{dc}"] = r
            row[f"p_{dc}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def perrun_correlations(pairs_df):
    """New approach: compute r per run → n=10 per condition."""
    rows = []
    for cond in pairs_df["condition"].unique():
        for run in range(1, N_RUNS + 1):
            sub = pairs_df[(pairs_df["condition"] == cond) &
                           (pairs_df["run"] == run)]
            if len(sub) < 10:
                print(f"  [WARN] {cond} run {run}: only {len(sub)} pairs, skipping")
                continue
            row = {"condition": cond, "run": run, "n_pairs": len(sub)}
            for dc in DIST_COLS:
                r, p = pearsonr(sub[dc], sub["delta_f"])
                row[f"r_{dc}"] = r
                row[f"p_{dc}"] = p
            rows.append(row)
    return pd.DataFrame(rows)


def condition_comparison(perrun_df):
    """Mann-Whitney U comparing tf67 vs original on n=10 r-values."""
    tf  = perrun_df[perrun_df["condition"] == "tf67"]
    ori = perrun_df[perrun_df["condition"] == "original"]
    rows = []
    for dc in DIST_COLS:
        col = f"r_{dc}"
        a, b = tf[col].values, ori[col].values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        cliff = (2 * u / (len(a) * len(b))) - 1
        rows.append({
            "distance":     dc,
            "tf67_mean":    a.mean(),
            "tf67_std":     a.std(),
            "original_mean": b.mean(),
            "original_std":  b.std(),
            "U":            u,
            "p":            p,
            "cliff_delta":  cliff,
        })
    return pd.DataFrame(rows)


# ── printing ────────────────────────────────────────────────────────────────

def print_pooled(pooled_df):
    print("\n" + "=" * 65)
    print("POOLED CORRELATIONS  (original approach, not independent)")
    print("=" * 65)
    print(f"{'condition':>10}  {'n_pairs':>8}  {'r_brain':>8}  {'r_phase':>8}  {'r_amp':>8}")
    for _, row in pooled_df.iterrows():
        print(f"{row.condition:>10}  {int(row.n_pairs):>8}  "
              f"{row.r_brain_dist:>8.3f}  {row.r_phase_dist:>8.3f}  {row.r_amp_dist:>8.3f}")


def print_perrun(perrun_df):
    print("\n" + "=" * 65)
    print("PER-RUN CORRELATIONS  (n=10 per condition, independent)")
    print("=" * 65)
    for cond in ["tf67", "original"]:
        sub = perrun_df[perrun_df["condition"] == cond]
        print(f"\n{cond}:")
        print(f"  {'run':>4}  {'n_pairs':>8}  {'r_brain':>8}  {'r_phase':>8}  {'r_amp':>8}")
        for _, row in sub.sort_values("run").iterrows():
            print(f"  {int(row.run):>4}  {int(row.n_pairs):>8}  "
                  f"{row.r_brain_dist:>8.3f}  {row.r_phase_dist:>8.3f}  {row.r_amp_dist:>8.3f}")
        print(f"  {'mean':>4}  {'':>8}  "
              f"{sub.r_brain_dist.mean():>8.3f}  "
              f"{sub.r_phase_dist.mean():>8.3f}  "
              f"{sub.r_amp_dist.mean():>8.3f}")
        print(f"  {'±std':>4}  {'':>8}  "
              f"{sub.r_brain_dist.std():>8.3f}  "
              f"{sub.r_phase_dist.std():>8.3f}  "
              f"{sub.r_amp_dist.std():>8.3f}")


def print_comparison(cmp_df):
    print("\n" + "=" * 65)
    print("CONDITION COMPARISON  (Mann-Whitney U, n=10 vs n=10)")
    print("=" * 65)
    print(f"{'distance':>12}  {'tf67':>7}  {'orig':>7}  {'U':>6}  {'p':>7}  {'Cliff Δ':>8}")
    for _, row in cmp_df.iterrows():
        sig = "*" if row.p < 0.05 else " "
        print(f"{row.distance:>12}  {row.tf67_mean:>7.3f}  {row.original_mean:>7.3f}  "
              f"{row.U:>6.0f}  {row.p:>7.4f}{sig}  {row.cliff_delta:>8.3f}")
    print("  (* p < 0.05)")


# ── figure ──────────────────────────────────────────────────────────────────

def plot_perrun(perrun_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    dist_labels = {
        "r_brain_dist": "Brain distance (4D)",
        "r_phase_dist": "Phase distance",
        "r_amp_dist":   "Amplitude distance",
    }
    COLORS = {"tf67": "#4488CC", "original": "#CC4444"}

    for ax, (col, label) in zip(axes, dist_labels.items()):
        tf  = perrun_df[perrun_df["condition"] == "tf67"][col].values
        ori = perrun_df[perrun_df["condition"] == "original"][col].values
        bp = ax.boxplot(
            [tf, ori], labels=["TF6/7", "Original"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, key in zip(bp["boxes"], ["tf67", "original"]):
            patch.set_facecolor(COLORS[key])
            patch.set_alpha(0.75)
        # overlay individual run points
        for x, vals, key in zip([1, 2], [tf, ori], ["tf67", "original"]):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(x + jitter, vals, color=COLORS[key],
                       alpha=0.8, s=25, zorder=5)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Pearson r  (with Δf)", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-run brain trait correlations with fitness improvement\n"
                 "(n = 10 runs per condition)", fontsize=10)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")
    plt.show()


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    # Load data
    dfs = []
    for cond_name, cfg in CONDITIONS.items():
        print(f"\nLoading {cond_name}...")
        dfs.append(load_condition(cond_name, cfg["experiment"]))
    df = pd.concat(dfs, ignore_index=True)

    # Compute brain traits
    df = compute_brain_traits(df)

    # Z-score within condition
    df = zscore_within_condition(df)

    # Build lookup for fast parent access
    z_cols = [c + "_z" for c in BRAIN_COLS]
    id_lookup = df.set_index("uid")[z_cols + ["displacement"]].to_dict("index")

    # Build all parent-child pairs
    print("\nBuilding parent-child pairs...", flush=True)
    pairs_df = build_pairs(df, id_lookup)
    print(f"Total pairs: {len(pairs_df)}  "
          f"(tf67={len(pairs_df[pairs_df.condition=='tf67'])}, "
          f"original={len(pairs_df[pairs_df.condition=='original'])})")

    # Analysis
    pooled_df  = pooled_correlations(pairs_df)
    perrun_df  = perrun_correlations(pairs_df)
    cmp_df     = condition_comparison(perrun_df)

    # Print results
    print_pooled(pooled_df)
    print_perrun(perrun_df)
    print_comparison(cmp_df)

    # Save figure
    out_fig = STUDY_ROOT / "figures" / "perrun_correlations.png"
    plot_perrun(perrun_df, out_fig)

    # Save per-run table to CSV for reference
    out_csv = STUDY_ROOT / "figures" / "perrun_correlations.csv"
    perrun_df.to_csv(out_csv, index=False)
    print(f"Table saved  -> {out_csv}")

    return perrun_df, cmp_df


if __name__ == "__main__":
    perrun_df, cmp_df = main()
