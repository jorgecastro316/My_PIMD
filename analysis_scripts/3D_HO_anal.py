import argparse
import re

import numpy as np

# ============================
# Constants (atomic units)
# ============================

# speed of light in atomic units
# 1 cm^-1 = 4.556335252912e-6 Hartree
CM_TO_HA = 4.556335252912e-6

# Boltzmann constant (Hartree / Kelvin)
K_B_AU = 3.166811563e-6

# default beta if user does not override
DEFAULT_BETA = 1000.0

# ============================
# Coth with stability handling
# ============================

def coth(x):
    """Numerically stable coth(x)."""
    x = np.array(x, dtype=float)
    out = np.zeros_like(x)

    small = np.abs(x) < 1e-6
    large = ~small

    # series expansion around 0: coth(x) ≈ 1/x + x/3
    out[small] = 1.0/x[small] + x[small]/3.0

    # direct formula for normal regime
    ex = np.exp(2.0 * x[large])
    out[large] = (ex + 1.0) / (ex - 1.0)

    return out

# ============================
# Classical average potential
# <V>_cl = 3 / (2 * beta)
# ============================

def classical_avg_potential(beta):
    return 3.0 / (2.0 * beta)

# ============================
# Quantum average potential
# <V>_qm = 0.25 * sum_i [ omega_i * coth(beta*omega_i/2) ]
# ============================

def quantum_avg_potential(beta, omegas):
    """Return <V> for a 3D anisotropic HO (omegas in Hartree)."""
    x = 0.5 * beta * omegas
    term = omegas * coth(x)
    return 0.25 * np.sum(term)

# ============================
# Temperature helpers
# ============================

def beta_to_temperature(beta):
    """Convert beta (a.u.) to Kelvin."""
    return 1.0 / (K_B_AU * beta)

def parse_temperature_arg(temp_tokens):
    """
    Accepts strings like '300 K', '315Kelvin', '1000 beta', etc.
    Returns (beta, temperature, description).
    """
    if temp_tokens is None:
        beta = DEFAULT_BETA
        T = beta_to_temperature(beta)
        return beta, T, f"{beta} beta"

    text = " ".join(temp_tokens).replace("=", " ").strip()
    if not text:
        raise ValueError("Temperature specification is empty.")

    parts = text.split()
    value_str = parts[0]
    unit_str = parts[1] if len(parts) > 1 else ""

    if not unit_str:
        match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)([A-Za-z]+)", value_str)
        if not match:
            raise ValueError(f"Could not parse temperature specification '{text}'.")
        value_str, unit_str = match.groups()

    try:
        value = float(value_str)
    except ValueError as exc:
        raise ValueError(f"Invalid temperature value '{value_str}'.") from exc

    unit = unit_str.lower()
    if unit in ("k", "kelvin"):
        temperature = value
        beta = 1.0 / (K_B_AU * temperature)
        description = f"{value} K"
    elif unit in ("b", "beta"):
        beta = value
        temperature = beta_to_temperature(beta)
        description = f"{value} beta"
    else:
        raise ValueError(
            f"Unrecognized temperature unit '{unit_str}'. Use K/Kelvin or B/Beta."
        )

    return beta, temperature, description

# ============================
# Main computation
# ============================

def main():

    parser = argparse.ArgumentParser(
        description="Classical vs quantum <V> for a 3D anisotropic harmonic oscillator."
    )
    parser.add_argument(
        "-t",
        "--temperature",
        nargs="+",
        metavar=("VALUE", "UNIT"),
        help="Temperature spec like '300 K' or '1000 beta'. Units accepted: K/Kelvin, B/Beta.",
    )
    args = parser.parse_args()

    beta, tempK, temp_desc = parse_temperature_arg(args.temperature)

    # Frequencies in cm^-1
    freq_cm = np.array([2000.0, 2400.0, 1800.0])

    # Convert to Hartree
    omegas = freq_cm * CM_TO_HA

    print("=== 3D Anisotropic HO Expectation Values ===")
    print("Temperature input:", temp_desc)
    print(f"Beta (a.u.): {beta}")
    print(f"Temperature (K): {tempK:.3f}")
    print("Frequencies (cm^-1):", freq_cm)
    print("Frequencies (Hartree):", omegas)

    V_cl = classical_avg_potential(beta)
    V_qm = quantum_avg_potential(beta, omegas)

    print("\nClassical <V> (Hartree):", V_cl)
    print("Quantum   <V> (Hartree):", V_qm)
    print("Quantum/Classical ratio:", V_qm / V_cl)

if __name__ == "__main__":
    main()
