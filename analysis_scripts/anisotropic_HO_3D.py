import numpy as np

# ==============================
# Constants in atomic units
# ==============================
kB_au = 3.166811563e-6            # Hartree/K
c_si = 2.99792458e10             # cm/s
hbar_si = 1.054571817e-34
Ha_si = 4.3597447222071e-18
m_p_au = 1836.15267389

# Conversion: cm^-1 -> omega (energy) in Hartree
def cm_to_omega_au(freq_cm):
    omega_si = 2.0 * np.pi * c_si * freq_cm       # angular frequency in s^-1
    return omega_si * (hbar_si / Ha_si)           # convert to Hartree

# ==============================
# Free energy expressions
# ==============================
def F_classical(w, beta):
    """
    Classical Helmholtz free energy (minus constants independent of w).
    w = array of ħω_i in Hartree.
    """
    log_wprod = np.log(np.prod(w))
    term = log_wprod + 3.0 * np.log(beta) - 3.0 * np.log(2.0 * np.pi)
    return term / beta

def F_quantum(w, beta):
    """
    Exact quantum free energy of 3D anisotropic HO.
    """
    F = 0.0
    for wi in w:
        F += 0.5 * wi + (1.0 / beta) * np.log(1.0 - np.exp(-beta * wi))
    return F

# ==============================
# Recommended number of beads P
# ==============================
def recommended_beads(w, beta):
    """
    Compute recommended bead count from:
        P > (ħ ω_max) / (k_B T)

    where:
        omega_max = largest ω (in cm^-1 or Hartree)
        ħω_max (in Hartree)
        T = 1/(k_B * beta)
    """
    w_max = np.max(w)                       # Hartree
    T = 1.0 / (kB_au * beta)                # Kelvin
    P_min = (w_max) / (kB_au * T)           # because ħω_max is w_max (already energy)
    return P_min

# ==============================
# Example system
# ==============================
beta = 1000.0                               # inverse temperature (a.u.)
T = 1.0 / (kB_au * beta)                    # temperature in Kelvin
print("Temperature (K):", T)

# Frequencies in cm^-1
wA_cm = np.array([1000.0, 1200.0, 900.0])
scale = 2.0
wB_cm = scale * wA_cm

# Convert to ħω (Hartree)
wA = cm_to_omega_au(wA_cm)
wB = cm_to_omega_au(wB_cm)

# Compute free energies
FclA = F_classical(wA, beta)
FclB = F_classical(wB, beta)
dF_cl = FclB - FclA

FqmA = F_quantum(wA, beta)
FqmB = F_quantum(wB, beta)
dF_qm = FqmB - FqmA

print("DeltaF_cl =", dF_cl)
print("DeltaF_qm =", dF_qm)

# ==============================
# Recommended number of beads
# ==============================
Pmin = recommended_beads(wB, beta)
print("Recommended P (beads) > ", Pmin)
print("Rounded recommended P:", int(np.ceil(Pmin)))
