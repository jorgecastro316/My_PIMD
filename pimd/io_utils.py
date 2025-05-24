# io_utils.py
import numpy as np
import re

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

def get_atomic_masses_au(symbols):
    try:
        return np.array([atomic_masses_au[s] for s in symbols])
    except KeyError as e:
        raise KeyError(e.args[0])

def parse_unit(comment_line, unit_type='length'):
    if unit_type == 'length':
        match = re.search(r'length_units\s*:\s*(\w+)', comment_line.lower())
        if match:
            unit = match.group(1)
            if unit in length_unit_to_bohr:
                return length_unit_to_bohr[unit]
            else:
                raise ValueError(f"Unknown length unit: {unit}")
        return 1.0
    elif unit_type == 'velocity':
        match = re.search(r'velocity_units\s*:\s*(\S+)', comment_line.lower())
        if match:
            unit = match.group(1)
            if unit in velocity_unit_to_au:
                return velocity_unit_to_au[unit]
            else:
                raise ValueError(f"Unknown velocity unit: {unit}")
        return 1.0
    else:
        raise ValueError("Unsupported unit type")

def load_xyz_file_with_units(filename, unit_type='length'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    comment_line = lines[1]
    atom_lines = lines[2:2 + n_atoms]

    conversion = parse_unit(comment_line, unit_type=unit_type)

    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.strip().split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:]])

    coords = np.array(coords) * conversion
    return {'symbols': symbols, 'coords': coords}

def read_input_file(filename="input.txt"):
    params = {
        'velocity_file': None
    }
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            keyval = line.strip().split()
            key = keyval[0].lower()
            value = keyval[1]
            if key == 'geometry_file':
                params['geometry_file'] = value
            elif key == 'velocity_file':
                params['velocity_file'] = value
            elif key == 'temperature':
                params['temperature'] = float(value)
            elif key == 'n_beads':
                params['n_beads'] = int(value)
            elif key == 'n_steps':
                params['n_steps'] = int(value)
            elif key == 'timestep':
                timestep_val = float(value)
                timestep_unit = keyval[2].lower() if len(keyval) > 2 else 'fs'
                if timestep_unit not in time_unit_to_au:
                    raise ValueError(f"Unknown time unit: {timestep_unit}")
                params['timestep'] = timestep_val * time_unit_to_au[timestep_unit]
            else:
                raise ValueError(f"Unknown input parameter: {key}")


    for field in ['geometry_file', 'temperature', 'n_beads', 'timestep', 'n_steps']:
        if field not in params:
            raise ValueError(f"Missing mandatory parameter: {field}")

    return params
