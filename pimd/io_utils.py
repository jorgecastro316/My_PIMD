# io_utils.py
import numpy as np
import re
from scipy.fft import rfft, irfft

# Boltzmann constant in atomic units (Hartree/Kelvin)
kB_au = 3.166811563e-6

# Conversion factors to Bohr (1 Bohr = 0.529177210903 Å)
length_unit_to_bohr = {
    'bohr': 1.0,
    'angstrom': 1.0 / 0.529177210903,
    'ang': 1.0 / 0.529177210903,
    'a': 1.0 / 0.529177210903,
    'nm': 10.0 / 0.529177210903,
    'pm': 0.01 / 0.529177210903,
    'm': 1.0e10 / 0.529177210903
}

# Conversion factors to atomic units of velocity (1 a.u. = 2.18769126364e6 m/s)
velocity_unit_to_au = {
    'au': 1.0,
    'a.u.': 1.0,
    'bohr/fs': 1.0 / 0.0210819,
    'm/s': 1.0 / 2.18769126364e6,
    'nm/ps': 1000.0 / (2.18769126364e6 * 0.529177210903),
    'angstrom/fs': 1.0 / (0.529177210903 * 0.0210819)
}

# Conversion factors to atomic units of time
time_unit_to_au = {
    'fs': 1.0 / 0.02418884,
    'ps': 1.0e3 / 0.02418884,
    'as': 1.0e-3 / 0.02418884,
    's': 1.0e15 / 0.02418884,
    'au': 1.0,
    'a.u.': 1.0
}
# Atomic masses in atomic units (electron mass = 1)
# Source: CODATA 2018, adjusted to a.u.
# Elements up to Z=100
atomic_masses_au = {
    'X': 1.0,
    'H': 1837.1526, 'He': 7294.2995, 'Li': 12653.279, 'Be': 16424.215, 'B': 20061.504,
    'C': 21894.659, 'N': 25533.216, 'O': 29160.946, 'F': 32745.987, 'Ne': 36423.773,
    'Na': 41930.789, 'Mg': 43822.297, 'Al': 50203.875, 'Si': 52103.247, 'P': 56446.799,
    'S': 58265.449, 'Cl': 64609.314, 'Ar': 72828.645, 'K': 71032.606, 'Ca': 80205.677,
    'Sc': 81911.275, 'Ti': 87381.297, 'V': 83726.728, 'Cr': 94504.179, 'Mn': 95326.870,
    'Fe': 102329.271, 'Co': 106132.783, 'Ni': 107258.132, 'Cu': 115601.794, 'Zn': 119348.487,
    'Ga': 127430.465, 'Ge': 131252.407, 'As': 137161.521, 'Se': 140156.136, 'Br': 146889.323,
    'Kr': 152491.293, 'Rb': 159411.542, 'Sr': 165288.925, 'Y': 170866.720, 'Zr': 178482.849,
    'Nb': 183839.087, 'Mo': 190280.726, 'Tc': 192752.793, 'Ru': 197240.319, 'Rh': 203353.730,
    'Pd': 209136.291, 'Ag': 196693.210, 'Cd': 204383.254, 'In': 208945.180, 'Sn': 218638.178,
    'Sb': 221685.562, 'Te': 229330.079, 'I': 232498.542, 'Xe': 239316.955, 'Cs': 242245.796,
    'Ba': 249264.667, 'La': 255001.833, 'Ce': 259558.724, 'Pr': 264003.760, 'Nd': 268568.243,
    'Pm': 273146.394, 'Sm': 277709.992, 'Eu': 282266.050, 'Gd': 286812.660, 'Tb': 291338.828,
    'Dy': 295858.336, 'Ho': 300372.492, 'Er': 304881.252, 'Tm': 309383.771, 'Yb': 313879.325,
    'Lu': 318372.630, 'Hf': 322866.635, 'Ta': 327360.566, 'W': 331853.907, 'Re': 336346.802,
    'Os': 340838.688, 'Ir': 345330.197, 'Pt': 349820.324, 'Au': 359407.689, 'Hg': 364393.201,
    'Tl': 368880.566, 'Pb': 373368.353, 'Bi': 377855.656, 'Po': 382342.412, 'At': 386828.881,
    'Rn': 391314.443, 'Fr': 395798.336, 'Ra': 400280.000, 'Ac': 404761.920, 'Th': 409243.902,
    'Pa': 413724.923, 'U': 418204.126, 'Np': 422683.180, 'Pu': 427162.449, 'Am': 431640.233,
    'Cm': 436118.383, 'Bk': 440595.801, 'Cf': 445072.892, 'Es': 449549.650, 'Fm': 454026.233,
    'Md': 458502.552, 'No': 462978.858, 'Lr': 467454.539, 'Rf': 471930.000, 'Db': 476405.000,
    'Sg': 480880.000, 'Bh': 485355.000, 'Hs': 489830.000, 'Mt': 494305.000, 'Ds': 498780.000,
    'Rg': 503255.000, 'Cn': 507730.000, 'Nh': 512205.000, 'Fl': 516680.000, 'Mc': 521155.000,
    'Lv': 525630.000, 'Ts': 530105.000, 'Og': 534580.000
}

def temperature_to_beta(temperature_K):
    temperature_K = np.float64(temperature_K)
    return np.float64(1.0) / (temperature_K * np.float64(kB_au))


def get_atomic_masses_au(symbols):
    try:
        return np.array([atomic_masses_au[s] for s in symbols], dtype=np.float64)
    except KeyError as e:
        raise KeyError(e.args[0])


def parse_unit(comment_line, unit_type='position'):
    if unit_type == 'position':
        match = re.search(r'positions\s*:\s*(\w+)', comment_line.lower())
        if match:
            unit = match.group(1)
            if unit in length_unit_to_bohr:
                return np.float64(length_unit_to_bohr[unit])
            else:
                raise ValueError(f"Unknown length unit: {unit}")
        return np.float64(1.0)
    elif unit_type == 'velocity':
        match = re.search(r'velocities\s*:\s*(\S+)', comment_line.lower())
        if match:
            unit = match.group(1)
            if unit in velocity_unit_to_au:
                return np.float64(velocity_unit_to_au[unit])
            else:
                raise ValueError(f"Unknown velocity unit: {unit}")
        return np.float64(1.0)
    else:
        raise ValueError("Unsupported unit type")


def load_xyz_file_with_units(filename, unit_type='position'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    comment_line = lines[1]
    atom_lines = lines[2:2 + n_atoms]

    conversion = np.float64(parse_unit(comment_line, unit_type=unit_type))

    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.strip().split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:]])
    coords = np.array(coords, dtype=np.float64) * conversion
    return {'symbols': symbols, 'coords': coords}

def read_input_file(filename="input.txt"):
    params = {
        'timestep': 1.0,  # default in fs
        'velocity_file': None
    }
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            key, value = line.strip().split(None, 1)
            key = key.lower()
            if key == 'geometry_file':
                params['geometry_file'] = value
            elif key == 'velocity_file':
                params['velocity_file'] = value
            elif key == 'temperature':
                params['temperature'] = np.float64(value)
            elif key == 'n_beads':
                params['n_beads'] = int(value)
            elif key == 'n_steps':
                params['n_steps'] = int(value)
            elif key == 'timestep':
                parts = value.strip().split()
                if len(parts) == 2:
                    val, unit = parts
                    val = np.float64(val)
                    unit = unit.lower()
                    if unit in time_unit_to_au:
                        params['timestep'] = val * time_unit_to_au[unit]
                    else:
                        raise ValueError(f"Unsupported time unit: {unit}")
                else:
                    # Assume it's already in atomic units
                    params['timestep'] = np.float64(value)
            elif key == 'force':
                params['force'] = value
            else:
                raise ValueError(f"Unknown input parameter: {key}")

    for field in ['geometry_file', 'temperature', 'n_beads', 'n_steps']:
        if field not in params:
            raise ValueError(f"Missing mandatory parameter: {field}")

    return params

def parse_force_spec(force_line):
    """
    Parse a force specification string, e.g. 'harmonic 0.5' or 'double_well 1.0 0.5'

    Parameters
    ----------
    force_line : str
        Line containing the force type and its parameters

    Returns
    -------
    force_type : str
        Name of the force field
    params : list of float
        Associated parameters (if any)
    """
    parts = force_line.strip().split()
    if not parts:
        raise ValueError("Empty force field specification.")
    force_type = parts[0].lower()
    params = list(map(float, parts[1:])) if len(parts) > 1 else []
    return force_type, params


def rr2nm(array):
    array = np.array(array, dtype=np.float64)  # Ensure float64 input
    P = array.shape[0]
    shape = array.shape
    flat = array.reshape(P, -1)
    
    nm = np.zeros_like(flat, dtype=np.complex128)
    nm[:] = rfft(flat, axis=0) * np.sqrt(2.0) / np.sqrt(P)
    
    nm[0] /= np.sqrt(2.0)
    if P > 1:
        nm[-1] /= np.sqrt(2.0)

    for j in range(1, P):
        nm[j] *= -1

    nm[0] /= np.sqrt(P)
    return nm.real.reshape(shape).astype(np.float64)

def nm2rr(array):
    array = np.array(array, dtype=np.float64)
    P = array.shape[0]
    shape = array.shape
    flat = array.reshape(P, -1).copy()

    flat[0] *= np.sqrt(P)
    for j in range(1, P):
        flat[j] /= -1

    if P > 1:
        flat[-1] *= np.sqrt(2.0)
    flat[0] *= np.sqrt(2.0)

    flat *= np.sqrt(P) / np.sqrt(2.0)
    rr = irfft(flat, n=P, axis=0)
    return rr.real.reshape(shape).astype(np.float64)


def freeRPProp(q_nm, p_nm, masses, dt, omegas):
    q_nm = np.array(q_nm, dtype=np.float64)
    p_nm = np.array(p_nm, dtype=np.float64)
    masses = np.array(masses, dtype=np.float64)
    dt = np.float64(dt)
    omegas = np.array(omegas, dtype=np.float64)

    n_beads, n_atoms, dim = q_nm.shape
    m = masses.reshape(1, n_atoms, 1)

    for k in range(n_beads):
        omega = omegas[k]
        if omega == 0:
            q_nm[k] = q_nm[k] + dt * (p_nm[k] / m)  # centroid drift
            continue

        cos_ = np.cos(omega * dt)
        sin_ = np.sin(omega * dt)

        q = q_nm[k]
        p = p_nm[k]

        q_nm[k] = cos_ * q + (sin_ / (omega * m)) * p
        p_nm[k] = cos_ * p - (m * omega * sin_) * q

    return q_nm, p_nm


def print_initial_conditions(positions, velocities, masses, force_type, force_params, temperature, dt, n_steps):
    print("\n=== INITIAL CONDITIONS ===")
    print(f"Temperature (K):        {temperature:.2f}")
    print(f"Time step (a.u.):       {dt:.6f}")
    print(f"Number of steps:        {n_steps}")
    print(f"Force field:            {force_type} {force_params if force_params else ''}")
    print(f"Masses (a.u.):          {masses}")
    print(f"Initial positions:      {positions}")
    print(f"Initial velocities:     {velocities}")
    print("===========================\n")
