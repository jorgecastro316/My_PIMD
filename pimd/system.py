import numpy as np
from pimd.io_utils import *

def initialize_ring_polymer(geometry_file, velocity_file, temperature, n_beads):
    # === Load geometry and infer dimensionality ===
    geom = load_xyz_file_with_units(geometry_file, unit_type='position')
    symbols = geom['symbols']
    coords = np.array(geom['coords'], dtype=np.float64)
    n_atoms, dim = coords.shape

    # === Masses ===
    try:
        masses = np.array(get_atomic_masses_au(symbols), dtype=np.float64)  # shape (n_atoms,)
    except KeyError as e:
        raise ValueError(f"Unknown atomic symbol '{e.args[0]}' encountered. Please check the input file or update the mass table.")

    # === Positions ===
    positions = np.tile(coords[None, :, :], (n_beads, 1, 1)).astype(np.float64)

    # === Velocities ===
    if velocity_file:
        vel_data = load_xyz_file_with_units(velocity_file, unit_type='velocity')
        v_symbols = vel_data['symbols']
        if v_symbols != symbols:
            raise ValueError("Velocity file symbols do not match geometry file symbols.")
        v_coords = np.array(vel_data['coords'], dtype=np.float64)
        if v_coords.shape[1] != dim:
            raise ValueError("Velocity file dimensionality does not match geometry file.")
        velocities = np.tile(v_coords[None, :, :], (n_beads, 1, 1)).astype(np.float64)
    else:
        velocities = np.zeros((n_beads, n_atoms, dim), dtype=np.float64)
        for i in range(n_atoms):
            sigma = np.sqrt(kB_au * temperature / masses[i])
            velocities[:, i, :] = np.random.normal(
                loc=0.0,
                scale=sigma,
                size=(n_beads, dim)
            ).astype(np.float64)

    return positions, velocities, masses, dim

