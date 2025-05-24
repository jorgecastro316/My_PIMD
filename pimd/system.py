import numpy as np
from pimd.io_utils import get_atomic_masses_au, kB_au, load_xyz_file_with_units, parse_unit

def load_system(filename):
    """
    Load atomic symbols and positions from an XYZ file.
    The second line may include a tag {length_units: ...} to convert coordinates to Bohr.

    Parameters
    ----------
    filename : str
        Path to the .xyz file

    Returns
    -------
    system : dict
        Dictionary with keys 'symbols' and 'coords' (coordinates in Bohr)
    dim : int
        Spatial dimensionality inferred from coordinate columns
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    comment_line = lines[1]
    atom_lines = lines[2:2 + n_atoms]

    length_to_bohr = parse_unit(comment_line, unit_type='length')

    atoms = []
    coords = []
    for line in atom_lines:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append(list(map(float, parts[1:])))

    coords = np.array(coords) * length_to_bohr
    dim = coords.shape[1]

    return {'symbols': atoms, 'coords': coords}, dim

def initialize_ring_polymer(system, n_beads, temperature, velocity_file=None):
    symbols = system['symbols']
    coords = system['coords']
    n_atoms, dim = coords.shape

    # Masses in atomic units with validation
    try:
        masses = get_atomic_masses_au(symbols)  # shape (n_atoms,)
    except KeyError as e:
        raise ValueError(f"Unknown atomic symbol '{e.args[0]}' encountered. Please check the input file or update the mass table.")

    # Initial positions: replicated across beads
    positions = np.tile(coords, (n_beads, 1, 1))  # shape (n_beads, n_atoms, dim)

    # Initial velocities: shape (n_beads, n_atoms, dim)
    if velocity_file:
        vel_data = load_xyz_file_with_units(velocity_file, unit_type='velocity')
        v_symbols = vel_data['symbols']
        if v_symbols != symbols:
            raise ValueError("Velocity file symbols do not match input.xyz symbols.")
        v_coords = vel_data['coords']  # shape (n_atoms, dim)
        if v_coords.shape[1] != dim:
            raise ValueError("Velocity file dimensionality does not match coordinate file.")
        velocities = np.tile(v_coords, (n_beads, 1, 1))
    else:
        velocities = np.zeros((n_beads, n_atoms, dim))
        for i in range(n_atoms):
            sigma = np.sqrt(kB_au * temperature / masses[i])
            velocities[:, i, :] = np.random.normal(0.0, sigma, size=(n_beads, dim))

    return positions, velocities, masses
