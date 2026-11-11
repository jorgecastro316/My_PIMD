"""
Compute classical and quantum harmonic free energies from input frequencies.

Usage:
    python3 analytic_free_energy.py 519.4 519.4 246.4 --T 300

Inputs are vibrational frequencies in cm^-1; output is free energy in Hartree.
"""

import argparse
import math
from typing import Sequence

# Physical constants in atomic units
KB_AU = 3.166811563e-6  # Hartree/K
# Conversion: cm^-1 -> angular frequency (a.u.)
CM1_TO_AU_ANG_FREQ = 2.0 * math.pi * 2.99792458e10 * 2.4188843265857e-17


def beta_from_T(T_kelvin: float) -> float:
    return 1.0 / (KB_AU * T_kelvin)


def cm1_to_au(freq_cm1: float) -> float:
    """Convert wavenumber (cm^-1) to angular frequency in atomic units."""
    return freq_cm1 * CM1_TO_AU_ANG_FREQ


def classical_harmonic_free_energy(freqs_cm1: Sequence[float], T_kelvin: float) -> float:
    beta = beta_from_T(T_kelvin)
    omegas = [cm1_to_au(f) for f in freqs_cm1]
    return sum(math.log(beta * w) for w in omegas) / beta


def quantum_harmonic_free_energy(freqs_cm1: Sequence[float], T_kelvin: float) -> float:
    beta = beta_from_T(T_kelvin)
    omegas = [cm1_to_au(f) for f in freqs_cm1]
    terms = [0.5 * w + (1.0 / beta) * math.log(1.0 - math.exp(-beta * w)) for w in omegas]
    return sum(terms)


def main():
    parser = argparse.ArgumentParser(description="Harmonic free energies from cm^-1 frequencies.")
    parser.add_argument("freqs", nargs="+", type=float, help="Frequencies (cm^-1)")
    parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin (default: 300)")
    args = parser.parse_args()

    F_classical = classical_harmonic_free_energy(args.freqs, args.T)
    F_quantum = quantum_harmonic_free_energy(args.freqs, args.T)

    print(f"F_classical (Ha): {F_classical:.8e}")
    print(f"F_quantum   (Ha): {F_quantum:.8e}")


if __name__ == "__main__":
    main()
