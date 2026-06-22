import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import get_full_connectivity


def trim_phenotype_materials(phenotype):
    """
    Trim empty borders from a phenotype and return a 2D grid.
    """
    body = np.asarray(phenotype, dtype=int)

    if body.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got {body.shape}")

    x_mask = np.any(body != 0, axis=1)
    body = body[x_mask]
    y_mask = np.any(body != 0, axis=0)
    body = body[:, y_mask]
    return body


def _material_maps(voxel_types):
    """
    Map GRN material IDs -> EvoGym voxel IDs.
    We intentionally map both muscle classes to H_ACT so mechanics are equal;
    phase differences are carried by controller phase offsets.
    """
    EVOGYM = {
        "EMPTY": 0,
        "RIGID": 1,
        "SOFT": 2,
        "H_ACT": 3,
    }

    if voxel_types == "withbone":
        # bone, fat, phase_muscle, offphase_muscle
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["RIGID"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    elif voxel_types == "nobone":
        # fat, fat2, phase_muscle, offphase_muscle
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["SOFT"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    else:
        raise ValueError(f"Unsupported voxel_types: {voxel_types}")

    # Controller groups (phase in radians) based on original GRN labels.
    # Only actuator materials get non-zero phase offsets.
    phase_offsets = {
        3: 0.0,      # phase_muscle
        4: np.pi,    # offphase_muscle
    }

    return material_to_evogym, phase_offsets


def _build_evogym_robot_data(body_materials, voxel_types, grn_phase_map=None, grn_amplitude_map=None):
    material_to_evogym, phase_offsets_by_material = _material_maps(voxel_types)

    structure = np.vectorize(lambda m: material_to_evogym.get(int(m), 0), otypes=[int])(body_materials)
    structure = structure.astype(np.int32)
    connections = get_full_connectivity(structure).astype(np.int32)

    muscle_ids = list(phase_offsets_by_material.keys())  # [3, 4]
    muscle_mask = np.isin(body_materials, muscle_ids)

    if grn_phase_map is not None:
        # GRN-derived continuous phase per muscle voxel
        phase_offsets = np.where(muscle_mask, grn_phase_map, 0.0).astype(np.float32)
    else:
        # Fallback: binary 0 / pi pattern
        phase_offsets = np.zeros_like(structure, dtype=np.float32)
        for mat_id, phase in phase_offsets_by_material.items():
            phase_offsets[body_materials == mat_id] = phase

    if grn_amplitude_map is not None:
        # GRN-derived continuous amplitude per muscle voxel
        amplitude_offsets = np.where(muscle_mask, grn_amplitude_map, 0.0).astype(np.float32)
    else:
        # Fallback: fixed amplitude for all muscles
        amplitude_offsets = np.where(muscle_mask, 0.4, 0.0).astype(np.float32)

    controller = {
        "action_bias": 1.0,
        "action_amplitude": 0.4,
        "period_steps": 20,
    }

    return structure, connections, phase_offsets, amplitude_offsets, controller


def prepare_robot_files(individual, args):
    """
    Prepare EvoGym robot artifacts from an evolved phenotype.
    Keeps the old function name so the EA loop can call it unchanged.
    """
    raw_body = np.asarray(individual.phenotype, dtype=int)
    body = trim_phenotype_materials(individual.phenotype)

    x_mask = np.any(raw_body != 0, axis=1)
    y_mask = np.any(raw_body != 0, axis=0)

    grn_phase_map = getattr(individual, 'grn_phase_map', None)
    if grn_phase_map is not None:
        grn_phase_map = grn_phase_map[x_mask][:, y_mask]

    grn_amplitude_map = getattr(individual, 'grn_amplitude_map', None)
    if grn_amplitude_map is not None:
        grn_amplitude_map = grn_amplitude_map[x_mask][:, y_mask]

    structure, connections, phase_offsets, amplitude_offsets, controller = _build_evogym_robot_data(
        body, args.voxel_types, grn_phase_map, grn_amplitude_map
    )

    # Keep data in-memory for upcoming EvoGym simulation adapter.
    individual.evogym_structure = structure
    individual.evogym_connections = connections
    individual.evogym_phase_offsets = phase_offsets
    individual.evogym_amplitude_offsets = amplitude_offsets
    individual.evogym_controller = controller

