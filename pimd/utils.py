#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:37:31 2022

@author: castrojo
"""

import numpy as np
import re
import numpy as np
from scipy.fftpack import rfft, irfft

# Boltzmann constant in atomic units (Hartree/Kelvin)
kB_au = 3.166811563e-6

# Conversion factors to Bohr (1 Bohr = 0.529177210903 Å)
length_unit_to_bohr = {
    'au': 1.0,
    'bohr': 1.0,
    'angstrom': 1.0 / 0.529177210903,
    'ang': 1.0 / 0.529177210903,
    'nm': 10.0 / 0.529177210903,
    'pm': 0.01 / 0.529177210903,
    'm': 1.0e10 / 0.529177210903
}

# Conversion factors to atomic units of velocity (1 a.u. = 2.18769126364e6 m/s)
velocity_unit_to_au = {
    'au': 1.0,
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
    'H': 1836.1526, 'He': 7294.2995, 'Li': 12653.279, 'Be': 16424.215, 'B': 20061.504,
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

def beta_to_temperature(beta_au):
    beta_au = np.float64(beta_au)
    return np.float64(1.0) / (beta_au * np.float64(kB_au))

def get_atomic_masses_au(symbols):
    try:
        return np.array([atomic_masses_au[s] for s in symbols], dtype=np.float64)
    except KeyError as e:
        raise KeyError(e.args[0])


def parse_unit(comment_line, unit_type='position'):
    if unit_type == 'position':
        match = re.search(r'positions\s*:\s*(\w+)', comment_line.lower())
        #print(match)
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
        'timestep': 1.0,  # default in atomic units
        'velocity_file': None,
        'thermostat': None,
        'tau': 1.0,  # default in atomic units
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
            elif key == 'prefix':
                params['prefix'] = value 
            elif key == 'seed':
                params['seed'] = int(value)
            elif key == 'thermostat':
                params['thermostat'] = value
            elif key == 'tau':
                parts = value.strip().split()
                if len(parts) == 2:
                    val, unit = parts
                    val = np.float64(val)
                    unit = unit.lower()
                    if unit in time_unit_to_au:
                        params['tau'] = val * time_unit_to_au[unit]
                    else:
                        raise ValueError(f"Unsupported time unit: {unit}")
                else:
                    # Assume it's already in atomic units
                    params['tau'] = np.float64(value)
        
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


# ===========================================================
# 1‐D normal mode transformation matrix
# ===========================================================
def C_matrix(Nbeads):
    C = np.zeros((Nbeads, Nbeads), float)
    C[0, :] = np.sqrt(1)
    for j in range(Nbeads):
        for k in range(1, (Nbeads // 2) + 1):
            C[k, j] = np.sqrt(2.0) * np.cos((2 * np.pi * j * k) / Nbeads)
        for k in range((Nbeads // 2) + 1, Nbeads):
            C[k, j] = np.sqrt(2.0) * np.sin((2 * np.pi * j * k) / Nbeads)
    if Nbeads % 2 == 0:
        C[Nbeads // 2, 0:Nbeads:2] = 1.0
        C[Nbeads // 2, 1:Nbeads:2] = -1.0
    return C / np.sqrt(Nbeads)

# ===========================================================
# Vectorized wrappers for any (P × dof) array
# ===========================================================

def rr2nm(q_cart):
    C = C_matrix(q_cart.shape[0])
    return C @ q_cart

def nm2rr(q_nm):
    C = C_matrix(q_nm.shape[0])
    return C.T @ q_nm


# ——————————————————————————————————————
# Free‐RP propagation block
# ——————————————————————————————————————

def omega_Cmatrix(Nbeads, beta_n):
    Wp = 1 / beta_n
    omega = np.zeros(Nbeads)
    for k in range(1, Nbeads):
        omega[k] = 2.0 * Wp * np.sin(np.pi * k / Nbeads)
    return omega

def propagate_normal_modes_general(p_nm, q_nm, omega, dt, masses, n_dim):
    n_beads, dof = p_nm.shape
    n_atoms = dof // n_dim
    mass_per_dof = np.repeat(masses, n_dim)  # shape = (dof,)
    #print(omega)
    p_nm2 = np.zeros_like(p_nm)
    q_nm2 = np.zeros_like(q_nm)

    for k in range(n_beads):
        w = omega[k]
        for i in range(dof):
            m = mass_per_dof[i]
            #print(m)
            if w == 0:
                # Centroid
                p_nm2[k, i] = p_nm[k, i]
                q_nm2[k, i] = q_nm[k, i] + dt * p_nm[k, i] / m
            else:
                cos_wdt = np.cos(w * dt)
                sin_wdt = np.sin(w * dt)
                p_nm2[k, i] = cos_wdt * p_nm[k, i] - m * w * sin_wdt * q_nm[k, i]
                q_nm2[k, i] = (sin_wdt / (m * w)) * p_nm[k, i] + cos_wdt * q_nm[k, i]

    return p_nm2, q_nm2


def thermal(Nbeads,Nparticles,Ndim,temperature, masses):
    dof = Nparticles * Ndim
    mvec = np.repeat(masses, Ndim)
    velocities = np.zeros((Nbeads,dof))
    for p in range(Nbeads):
        for b in range(dof):
            m = mvec[b]
            velocities[p,b] = np.random.normal(loc=0, scale=np.sqrt((temperature*kB_au/m)))
    return(velocities)


def Internal_Potential(q, temperature, masses, n_beads, n_atoms, n_dim):
    """
    Compute internal harmonic potential of the ring polymer.

    Parameters
    ----------
    q         : np.ndarray, shape (n_beads, n_atoms * n_dim)
    temperature : float, in Kelvin
    masses    : np.ndarray, shape (n_atoms,)
    n_atoms   : int
    n_dim     : int

    Returns
    -------
    V_int : float
        Total internal ring-polymer spring energy
    """
    q = q.reshape(-1, n_atoms, n_dim)      # shape (P, N, dim)
    beta_n = temperature_to_beta(temperature)
    omega_P = 1 / beta_n                     # ring-polymer frequency

    V_int = 0.0
    for i in range(n_atoms):
        m = masses[i]
        q_i = q[:, i, :]                   # shape (P, dim)
        dq = q_i - np.roll(q_i, shift=1, axis=0)  # cyclic diff: q_p - q_{p-1}, shape (P, dim)
        V_int += 0.5 * m * omega_P**2 * np.sum(dq**2)
    #print(f"Internal potential energy: {V_int/ n_beads}")
    return float(V_int)   ###################################double check this

def E_Estimators(temperature, forces, beads_positions, centroids_positions, potential_energies,n_beads, n_atoms, n_dim):

    beta_n = temperature_to_beta(temperature)
    beta = beta_n * n_beads  # Total beta for the ring polymer
    #beta = 1
    FQ = 0.0

    for i in range(n_atoms):
        q_beads = beads_positions[:, i*n_dim:(i+1)*n_dim]  # shape (n_beads, n_dim)
        qc = centroids_positions[0, i*n_dim:(i+1)*n_dim]    # shape (n_dim,)
        
        for j in range(n_beads):
            q = q_beads[j]
            f = forces[j, i*n_dim:(i+1)*n_dim]
            FQ += np.dot(q - qc, f)

    E_kin = (n_atoms*n_dim * 0.5) / beta - (0.5 / n_beads) * FQ

    E_pot = float(np.sum(potential_energies))/ n_beads
 
    E_tot = E_kin + E_pot
    
    return E_kin, E_pot, E_tot

def instantaneous_temperature(velocities, masses, n_beads, n_atoms, n_dim):
    r"""
    Compute the instantaneous temperature of the ring polymer:

        T = (2 * K_cl) / (N_dof * kB),
        where N_dof = P * N * dim

    Parameters
    ----------
    velocities : np.ndarray, shape (P, N * dim)
    masses     : np.ndarray, shape (N,)
    n_atoms    : int
    n_dim      : int

    Returns
    -------
    T_inst : float
        Instantaneous temperature in Kelvin
    """
    vel = np.array(velocities, dtype=np.float64).reshape(-1, n_atoms, n_dim)  # (P, N, dim)
    m = np.array(masses, dtype=np.float64).reshape(1, -1, 1)  # (1, N, 1)

    P = n_beads
    N_dof = P * n_atoms * n_dim

    K_cl = np.sum(0.5 * m * vel**2)  # scalar total kinetic energy

    return float(2.0 * K_cl / (P*N_dof * kB_au)) #The extra P in the denominator account for the fact that we are sampling at temperature T*P   
    
def classical_kinetic_energy(velocities, masses, n_atoms, n_dim):
    r"""
    Total classical kinetic energy of the ring polymer:
        K = 0.5 * sum_i sum_k m_i * v_{ik}^2

    Parameters
    ----------
    velocities : np.ndarray, shape (P, N * dim)
    masses     : np.ndarray, shape (N,)
    n_atoms    : int
    n_dim      : int

    Returns
    -------
    K_cl : float
        Total classical kinetic energy in atomic units
    """
    # Reshape velocities to (P, N, dim)
    vel = np.array(velocities, dtype=np.float64).reshape(-1, n_atoms, n_dim)
    m   = np.array(masses, dtype=np.float64).reshape(1, -1, 1)  # shape (1, N, 1)

    K_bead = 0.5 * m * vel**2  # shape (P, N, dim)
    return float(np.sum(K_bead))

def classical_potential_energy(potential_energies):
    r"""
    Total classical potential energy of the ring polymer:


    Parameters
    ----------
    potential_energies : np.ndarray, shape (n_beads, n_particles)

    Returns
    -------
    V_cl : float
        Total classical potential energy in atomic units
    """

    return float(np.sum(potential_energies))

def compute_energetics(positions,velocities,q_centroid,forces,potential_energies,masses,temperature,n_beads,n_atoms,n_dim):        
    #Classical energies
    internal_pot = Internal_Potential(positions,temperature,masses,n_beads,n_atoms,n_dim)
    #print("spring", internal_pot)
    external_pot = classical_potential_energy(potential_energies)            
    pot =  (internal_pot + external_pot)/n_beads
    kin = classical_kinetic_energy(velocities,masses,n_atoms,n_dim)/n_beads #Kinetic energy in the system
    etot = kin+pot
    system_T = instantaneous_temperature(velocities,masses,n_beads,n_atoms,n_dim) #Temperature of the system-> Not the same as the thermostat temperature
    
    #QM estimators   
    E_kin,E_pot,E_tot = E_Estimators(temperature,forces,positions,q_centroid,potential_energies,n_beads,n_atoms,n_dim)
    
    return(pot,kin,etot,E_pot,E_kin,E_tot,system_T,internal_pot)
    
def update_system(system,q,v,qc,vc,f,system_T,pot,kin,etot,E_pot,E_kin,E_tot,internal_pot):
    
    #Update the system        
    system.set_velocities(v)        
    system.set_positions(q)
    system.set_qc(qc)
    system.set_vc(vc)
    system.set_forces(f)
    system.set_Temperature(system_T)
    
    #Update Classical energies
    system.set_pot(pot)
    system.set_kin(kin)
    system.set_tot(etot)
    
    #Update Energy estimators
    system.set_E_pot(E_pot)
    system.set_E_kin(E_kin)
    system.set_E_tot(E_tot)
    system.set_int_pot(internal_pot)  

def sign(value):
    if value > 0:
        return(float(1))
    elif value < 0:
        return(float(-1))
    
    
def velocities_to_momenta(velocities, masses, n_atoms, n_dim):
    """
    Convert flattened velocities to momenta using masses.

    Parameters
    ----------
    velocities : np.ndarray, shape (n_beads, n_atoms * n_dim)
    masses     : np.ndarray, shape (n_atoms,)
    n_atoms    : int
    n_dim      : int

    Returns
    -------
    momenta : np.ndarray, shape (n_beads, n_atoms * n_dim)
    """
    v = velocities.reshape(-1, n_atoms, n_dim)               # shape (P, N, dim)
    m = masses.reshape(1, n_atoms, 1)                        # shape (1, N, 1)
    p = m * v                                                # p = m*v, shape (P, N, dim)
    return p.reshape(-1, n_atoms * n_dim)                    # flatten back to (P, N * dim)

def momenta_to_velocities(momenta, masses, n_atoms, n_dim):
    """
    Convert flattened momenta to velocities using masses.

    Parameters
    ----------
    momenta : np.ndarray, shape (n_beads, n_atoms * n_dim)
    masses  : np.ndarray, shape (n_atoms,)
    n_atoms : int
    n_dim   : int

    Returns
    -------
    velocities : np.ndarray, shape (n_beads, n_atoms * n_dim)
    """
    p = momenta.reshape(-1, n_atoms, n_dim)                  # shape (P, N, dim)
    m = masses.reshape(1, n_atoms, 1)                        # shape (1, N, 1)
    v = p / m                                                # v = p/m
    return v.reshape(-1, n_atoms * n_dim)                    # flatten back to (P, N * dim)

def classical_PE(T, K, ndim):
    """
    Energía potencial media clásica de un oscilador armónico
    T : temperatura en Kelvin
    K : constante del resorte en Hartree/bohr²
    ndim : número de dimensiones
    """
    beta = temperature_to_beta(T)
    return ndim / (2 * beta)

def quantum_PE(T, K, ndim, m=1836.1526):
    """
    Energía potencial media cuántica de un oscilador armónico
    T : temperatura en Kelvin
    K : constante del resorte en Hartree/bohr²
    ndim : número de dimensiones
    m : masa en unidades atómicas (por defecto: protón)
    """
    omega = np.sqrt(K / m)
    beta = temperature_to_beta(T)
    return ndim * (0.25 * omega / np.tanh(0.5 * beta * omega))