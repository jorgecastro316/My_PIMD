import numpy as np
EV_TO_HA=1/27.211386245988
ANG2BOHR=1/0.529177210903
# a (1/Å) must be converted by dividing by Å→Bohr
De=0.2*EV_TO_HA; a=0.75/ANG2BOHR; z0=1.11*ANG2BOHR
k=1.0*EV_TO_HA/(ANG2BOHR**2)
T=300.0; kB=3.166811563e-6; beta=1/(kB*T)

def Uz(z):
    dz=(z-z0)*a
    return De*(dz**2 - dz**3 + (7/12)*dz**4)

zs = np.linspace(z0-6, z0+6, 40000)
w = np.exp(-beta*Uz(zs))
# numpy <1.20 compatibility: use trapz if trapezoid is unavailable
trapz = getattr(np, "trapezoid", np.trapz)
Z = trapz(w, zs)
U_z = trapz(Uz(zs)*w, zs)/Z
U_tot = U_z + kB*T  # adds ½kBT per x,y
print(U_z, U_z/EV_TO_HA, U_tot, U_tot/EV_TO_HA)
