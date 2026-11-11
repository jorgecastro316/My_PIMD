"""
Quick checks for the composite Morse + harmonic potential.

Usage:
    python3 analysis_scripts/test_morse_harmonic.py

This exercises the new `morse_harmonic` force-field entry by comparing the
returned forces/energies against an analytic implementation with the same
unit conversions used in the i-PI tutorial (params in eV/Å).
"""

import os
import sys
import numpy as np

# Ensure repository root is on sys.path so `pimd` resolves when run as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pimd.force_field import Force_Field, EV_TO_HA, ANGSTROM_TO_BOHR


def convert_params(params_ev_ang):
    """Convert [De(eV), a(1/Å), z0(Å), k(eV/Å^2)] to atomic units."""
    De_ev, a_inv_ang, z0_ang, k_ev_per_ang2 = params_ev_ang
    De = De_ev * EV_TO_HA
    # 1/Å -> 1/Bohr (divide by Å→Bohr)
    a = a_inv_ang / ANGSTROM_TO_BOHR
    z0 = z0_ang * ANGSTROM_TO_BOHR
    k = k_ev_per_ang2 * EV_TO_HA / (ANGSTROM_TO_BOHR ** 2)
    return De, a, z0, k


def analytic_energy_force(q_bohr, params_ev_ang):
    """
    Standalone energy/force evaluator for the composite potential.

    q_bohr: array of shape (..., 3) in Bohr.
    returns: energy (Ha), forces (Ha/Bohr) matching force_field.morse_harmonic.
    """
    De, a, z0, k = convert_params(params_ev_ang)
    xy = q_bohr[..., :2]
    z = (q_bohr[..., 2] - z0) * a

    energy = De * (z**2 - z**3 + (7.0 / 12.0) * z**4) + 0.5 * k * np.sum(xy**2, axis=-1)
    forces = np.zeros_like(q_bohr)
    forces[..., :2] = -k * xy
    forces[..., 2] = -De * a * (2.0 * z - 3.0 * z**2 + (7.0 / 3.0) * z**3)
    return energy, forces


def main():
    params = [0.2, 0.75, 1.11, 1.0]  # eV, 1/Å, Å, eV/Å^2 (tutorial defaults)
    De, a, z0, k = convert_params(params)

    ff = Force_Field([{"type": "morse_harmonic", "args": params}])

    # Equilibrium check
    q_eq = np.array([[0.0, 0.0, z0]])  # (1,3) in Bohr
    forces_ff, pe_ff = ff.ff(q_eq, n_beads=1, n_particles=1, n_dim=3,
                             type="morse_harmonic", params=params)
    pe_ref, f_ref = analytic_energy_force(q_eq, params)
    print("Equilibrium energy (should be ~0):", pe_ff.ravel()[0])
    print("Equilibrium force:", forces_ff.ravel())

    assert np.allclose(pe_ff, pe_ref), "Energy mismatch at equilibrium"
    assert np.allclose(forces_ff, f_ref), "Force mismatch at equilibrium"

    # Random point comparison
    rng = np.random.default_rng(0)
    q_rand = q_eq + rng.normal(scale=0.1, size=q_eq.shape)  # small displacement
    forces_ff, pe_ff = ff.ff(q_rand, 1, 1, 3, "morse_harmonic", params)
    pe_ref, f_ref = analytic_energy_force(q_rand, params)
    assert np.allclose(pe_ff, pe_ref, atol=1e-12), "Energy mismatch at random point"
    assert np.allclose(forces_ff, f_ref, atol=1e-10), "Force mismatch at random point"

    # Finite-difference gradient check along z
    dz = 1e-5
    q_plus = q_rand.copy()
    q_minus = q_rand.copy()
    q_plus[0, 2] += dz
    q_minus[0, 2] -= dz
    pe_plus, _ = analytic_energy_force(q_plus, params)
    pe_minus, _ = analytic_energy_force(q_minus, params)
    num_grad_z = (pe_plus - pe_minus) / (2 * dz)
    assert np.allclose(num_grad_z, -forces_ff[0, 2], rtol=1e-5, atol=1e-7), \
        "Z force fails finite-difference check"

    print("All checks passed for morse_harmonic potential.")


if __name__ == "__main__":
    main()
