#!/usr/bin/env python3
# Quantum & classical <V> for the symmetric double well V(x)=V0(x^4 - x^2)
# Units: ħ = 1, k_B = 1 by default (dimensionless atomic-like)

import numpy as np
from numpy.linalg import eigh

# ----- physical parameters -----
hbar = 1.0
m    = 1.0
V0   = 5.0
beta = 1.0

# ----- grid / box (increase N or box if needed) -----
xmin, xmax = -8.0, 8.0
N = 900  # 300–1000 already converges <V> nicely for these params

def V_sym(x):
    return V0 * (x**4 - x**2)

def solve_schrodinger_1d(Vfunc, m, hbar, xmin, xmax, N):
    """Return eigenvalues/eigenvectors on a uniform FD grid."""
    x  = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]

    # Kinetic operator for - (ħ^2 / 2m) d2/dx2  → tri-diagonal with [2, -1, -1]
    c = (hbar**2) / (2.0 * m * dx*dx)
    T = np.zeros((N, N))
    np.fill_diagonal(T, 2.0 * c)
    np.fill_diagonal(T[1:], -c)
    np.fill_diagonal(T[:, 1:], -c)

    Vx = Vfunc(x)
    H  = T + np.diag(Vx)

    evals, evecs = eigh(H)  # evecs columns are eigenstates
    return evals, evecs, x

def quantum_V_avg(Vx, evals, evecs, beta):
    """
    <V>_quant = (1/Z) sum_n e^{-βE_n} <n|V|n>
    with <n|V|n> ≈ sum_i |ψ_n(x_i)|^2 V(x_i).
    IMPORTANT: multiply Vx along the ROWS (x-index), then sum over rows.
    """
    # Boltzmann weights with ground-state shift for stability
    E0 = evals[0]
    w  = np.exp(-beta * (evals - E0))

    # Expectation per state: sum_i |psi_i,n|^2 * V(x_i)
    Vn = np.sum((evecs**2) * Vx[:, None], axis=0)

    # Thermal average
    return np.sum(w * Vn) / np.sum(w)

def classical_V_avg(x, Vx, beta):
    """<V>_cl = ∫ V e^{-βV} / ∫ e^{-βV}  (simple trapz on the box)."""
    w = np.exp(-beta * Vx)
    Z = np.trapz(w, x)
    return np.trapz(Vx * w, x) / Z

# ---------- run ----------
evals, evecs, x = solve_schrodinger_1d(V_sym, m, hbar, xmin, xmax, N)
Vx = V_sym(x)

V_quant = quantum_V_avg(Vx, evals, evecs, beta)
V_class = classical_V_avg(x, Vx, beta)

print(f"Units: ħ={hbar}, m={m}, V0={V0}, β={beta}, box=[{xmin},{xmax}], N={N}")
print(f"Quantum  <V> = {V_quant:.10f}")
print(f"Classical<V> = {V_class:.10f}")
