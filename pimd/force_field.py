import numpy as np

def harmonic_force(positions, k=1.0):
    """
    Compute forces for a 1D harmonic oscillator: V(x) = 0.5 * k * x^2

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_beads, n_atoms, dim)
    k : float
        Force constant in a.u.

    Returns
    -------
    forces : np.ndarray
        Same shape as positions
    """
    return -k * positions

def double_well_force(positions, a=1.0, b=1.0):
    """
    Compute forces for a 1D double-well potential: V(x) = a*x^4 - b*x^2

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_beads, n_atoms, dim)
    a : float
        Quartic coefficient
    b : float
        Quadratic coefficient

    Returns
    -------
    forces : np.ndarray
    """
    return - (4 * a * positions**3 - 2 * b * positions)

def zero_force(positions):
    """
    Return zero forces (used for testing free ring polymer propagation).

    Parameters
    ----------
    positions : np.ndarray

    Returns
    -------
    forces : np.ndarray
    """
    return np.zeros_like(positions)


def force_field(force_type, params):
    force_type = force_type.lower()
    if force_type == "harmonic":
        k = params[0] if params else 1.0
        return lambda x: harmonic_force(x, k=k)
    elif force_type == "double_well":
        a = params[0] if len(params) > 0 else 1.0
        b = params[1] if len(params) > 1 else 1.0
        return lambda x: double_well_force(x, a=a, b=b)
    elif force_type == "zero":
        return zero_force
    else:
        raise ValueError(f"Unknown force type: {force_type}")