import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from algorithms.GRN_2D import GRN, initialization
from algorithms.voxel_types import VOXEL_TYPES, TF_WEIGHTS

rng = __import__('random').Random(42)
n_robots = 10

print(f"{'Robot':<8} {'Muscles':<10} {'Phase unique':<30} {'Amplitude unique'}")
print("-" * 80)

for i in range(n_robots):
    genome = initialization(rng, 300)
    grn = GRN(
        promoter_threshold=0.95,
        max_voxels=27,
        cube_face_size=3,
        voxel_types='withbone',
        genotype=genome,
    )
    grn.develop()

    muscle_mask = np.zeros(grn.phenotype.shape, dtype=bool)
    muscle_types = {VOXEL_TYPES.get('phase_muscle'), VOXEL_TYPES.get('offphase_muscle')}
    for idx, cell in np.ndenumerate(grn.phenotype):
        if cell != 0 and cell.voxel_type in muscle_types:
            muscle_mask[idx] = True

    muscle_count = np.sum(muscle_mask)
    phases = np.unique(grn.phase_map[muscle_mask].round(3)) if muscle_count > 0 else []
    amps = np.unique(grn.amplitude_map[muscle_mask].round(3)) if muscle_count > 0 else []

    print(f"{i+1:<8} {muscle_count:<10} {str(phases):<30} {str(amps)}")
