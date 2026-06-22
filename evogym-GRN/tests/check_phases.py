import sys
sys.path.append("/Users/augustincoman/Empire/Uni stuff/thesis-code/evogym-GRN")

import random
import numpy as np
from algorithms.GRN_2D import GRN, initialization
from simulation.prepare_robot_files import prepare_robot_files
from algorithms.EA_classes import Individual
from types import SimpleNamespace

rng = random.Random(42)
args = SimpleNamespace(voxel_types="withbone")

for i in range(4):
    genome = initialization(rng, ini_genome_size=300)
    grn = GRN(
        promoter_threshold=0.95,
        max_voxels=125,
        cube_face_size=5,
        voxel_types="withbone",
        genotype=genome,
        env_conditions="",
        plastic=0,
    )
    phenotype = grn.develop()

    import numpy as np
    phenotype_materials = np.zeros(phenotype.shape, dtype=int)
    for idx, val in np.ndenumerate(phenotype):
        phenotype_materials[idx] = val.voxel_type if val != 0 else 0

    ind = Individual(genome=genome, id_counter=i)
    ind.phenotype = phenotype_materials
    ind.grn_phase_map = grn.phase_map

    prepare_robot_files(ind, args)

    muscle_phases = ind.evogym_phase_offsets[ind.evogym_phase_offsets != 0]
    print(f"Robot {i}: {len(muscle_phases)} actuators | "
          f"phases min={muscle_phases.min():.3f} max={muscle_phases.max():.3f} "
          f"unique={len(np.unique(muscle_phases.round(3)))}"
          if len(muscle_phases) > 0 else f"Robot {i}: no actuators")
