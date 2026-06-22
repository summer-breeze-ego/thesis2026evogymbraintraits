import sqlite3
import pandas as pd
from pathlib import Path

rows = []
for cond in ["cmp500_tf67", "cmp500_original"]:
    for run in range(1, 11):
        db = Path(f"tmp_out/thesis/{cond}/run_{run}/run_{run}")
        con = sqlite3.connect(db)
        row = con.execute("""
            SELECT r.num_voxels, r.phase_muscle_count, r.offphase_muscle_count
            FROM generation_survivors s JOIN all_robots r ON s.robot_id = r.robot_id
            WHERE s.generation = (SELECT MAX(generation) FROM generation_survivors)
            ORDER BY r.displacement DESC LIMIT 1
        """).fetchone()
        con.close()
        if row:
            rows.append({"cond": cond, "total": row[0],
                         "phase_m": row[1], "offphase_m": row[2],
                         "bone": row[0] - row[1] - row[2]})

df = pd.DataFrame(rows)
print(df.groupby("cond").mean().round(1))
