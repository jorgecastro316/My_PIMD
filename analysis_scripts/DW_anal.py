#!/usr/bin/env python3
# Quantum and classical free-energy difference between two double wells:
#   V_A(x) = V0 (x^4 - x^2)
#   V_B(x) = V0 (x^4 - x^2 + x)
# Quantum:   ΔF = -(1/β) ln(Z_B / Z_A) from eigenvalues of 1D Schrödinger operator
# Classical: ΔF_cl = -(1/β) ln( ∫ e^{-β V_B} / ∫ e^{-β V_A} ), momentum cancels

import numpy as np
from numpy.linalg import eigh

# ----------------------- user parameters -----------------------
hbar = 1.0   # atomic-like units
m    = 1827.1526   # mass
V0   = 5   # potential prefactor
beta = 1052.6   # inverse temperature (k_B = 1)

# Grid/box (increase for tighter convergence)
xmin, xmax = -8.0, 8.0
N = 2000  # 800–4000 is typical; raise if you need more digits

# ----------------------- potentials ----------------------------
def V_A(x):
    return V0 * (x**4 - x**2)

def V_B(x):
    return V0 * (x**4 - x**2 + x)

# ------------------ Schrödinger solver (FD) --------------------
def eigsolve_1d(Vfunc, m, hbar, xmin, xmax, N):
    """
    Dense finite-difference eigen-solver for 1D Hamiltonian:
      H = - (hbar^2 / 2m) d2/dx2 + V(x)
    Returns eigenvalues, eigenvectors, grid x.
    """
    x  = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    c  = (hbar**2) / (2.0 * m * dx*dx)
    # tri-diagonal kinetic
    T = np.zeros((N, N))
    np.fill_diagonal(T, 2.0*c)
    np.fill_diagonal(T[1:], -c)
    np.fill_diagonal(T[:,1:], -c)
    Vx = Vfunc(x)
    H  = T + np.diag(Vx)
    evals, evecs = eigh(H)
    return evals, evecs, x

def quantum_Z_from_evals(evals, beta):
    """Z = sum_n exp(-β E_n). Use ground-state shift for stability."""
    E0 = evals[0]
    w  = np.exp(-beta * (evals - E0))
    return w.sum() * np.exp(-beta * E0)

# ----------------- classical configurational Z ------------------
def classical_Zcfg(Vx, x, beta):
    """Z_cfg = ∫ e^{-β V(x)} dx (trapz on the chosen box)."""
    w = np.exp(-beta * Vx)
    return np.trapz(w, x)

# ----------------------------- run -----------------------------
# Quantum
EA, _, x = eigsolve_1d(V_A, m, hbar, xmin, xmax, N)
EB, _, _ = eigsolve_1d(V_B, m, hbar, xmin, xmax, N)
ZA = quantum_Z_from_evals(EA, beta)
ZB = quantum_Z_from_evals(EB, beta)
dF_quant = -(1.0/beta) * np.log(ZB / ZA)

# Classical (configurational only; momentum cancels in ratio)
VAx = V_A(x)
VBx = V_B(x)
ZA_cfg = classical_Zcfg(VAx, x, beta)
ZB_cfg = classical_Zcfg(VBx, x, beta)
dF_class = -(1.0/beta) * np.log(ZB_cfg / ZA_cfg)

# ------------------------ basic reporting ----------------------
print("=== Free-energy difference ΔF (B − A) ===")
print(f"Units: ħ={hbar}, m={m}, V0={V0}, β={beta}, box=[{xmin},{xmax}], N={N}")
print(f"Quantum  ΔF   = {dF_quant:.10f}   (βΔF = {beta*dF_quant:.10f})")
print(f"Classical ΔF  = {dF_class:.10f}   (βΔF = {beta*dF_class:.10f})")

# Optional sanity checks / convergence hints
# (uncomment to print a few lowest eigenvalues)
# print("\nLowest quantum levels (A):", EA[:6])
# print("Lowest quantum levels (B):", EB[:6])

# You can test convergence by enlarging the box or N and confirming ΔF is stable.
