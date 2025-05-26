import numpy as np
from pimd.io_utils import *

def basic_verlet(positions, momenta, masses, forces_fn, dt):
    """
    Perform one step of the basic Verlet integrator using momenta.
    """
    positions = positions.astype(np.float64, copy=False)
    momenta = momenta.astype(np.float64, copy=False)
    masses = masses.astype(np.float64, copy=False)

    forces = forces_fn(positions)
    momenta += 0.5 * dt * forces
    positions += dt * momenta / masses.reshape(1, -1, 1)
    forces = forces_fn(positions)
    momenta += 0.5 * dt * forces
    return positions, momenta, forces

def path_integral_propagator(positions, momenta, masses, dt, forces_fn, beta):
    positions = positions.astype(np.float64, copy=False)
    momenta = momenta.astype(np.float64, copy=False)
    masses = masses.astype(np.float64, copy=False)

    forces = forces_fn(positions)
    momenta += 0.5 * dt * forces

    q_nm = rr2nm(positions)
    p_nm = rr2nm(momenta)

    n_beads = positions.shape[0]
    omegas = 2.0 * np.pi * np.arange(n_beads, dtype=np.float64) / beta

    q_nm, p_nm = freeRPProp(q_nm, p_nm, masses, dt, omegas)

    positions = nm2rr(q_nm)
    momenta = nm2rr(p_nm)

    forces = forces_fn(positions)
    momenta += 0.5 * dt * forces

    return positions, momenta, forces

def integrate_one_step(method, positions, velocities, masses, forces_fn, dt, temperature):
    positions = positions.astype(np.float64, copy=False)
    velocities = velocities.astype(np.float64, copy=False)
    masses = masses.astype(np.float64, copy=False)

    momenta = velocities * masses.reshape(1, -1, 1)

    method = method.lower()
    if method in ['basic', 'verlet', 'velocity_verlet']:
        positions, momenta, forces = basic_verlet(positions, momenta, masses, forces_fn, dt)
    elif method in ['pi', 'path_integral']:
        beta = temperature_to_beta(temperature)
        positions, momenta, forces = path_integral_propagator(positions, momenta, masses, dt, forces_fn, beta)
    else:
        raise ValueError(f"Unknown integrator method: {method}")

    velocities = momenta / masses.reshape(1, -1, 1)
    return positions, velocities, forces
