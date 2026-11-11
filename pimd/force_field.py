import numpy as np
from pimd.utils import *

# Convert cm^-1 → atomic angular frequency
CM1_TO_AU_ANG_FREQ = 2.0 * np.pi * 2.99792458e10 * 2.4188843265857e-17
# Energy/length conversions (Hartree, Bohr)
EV_TO_HA = 1.0 / 27.211386245988  # eV → Ha
ANGSTROM_TO_BOHR = length_unit_to_bohr["angstrom"]


class Force_Field(object):

    def __init__(self, potentials, direction="forward", *, debug=False):
        #print("vergacion")
        self.potentials = potentials
        self._reference_positions = None  # (n_particles, n_dim)
        self._masses = None               # (n_particles,)
        self._n_particles = None
        self._n_dim = None
        self.debug = bool(debug)
        self.direction = direction

    def _log(self, msg):
        if self.debug:
            print(f"[Force_Field] {msg}")

    def register_system(self, positions, masses, n_particles, n_dim, *, force=False):
        """
        Register atomic metadata so force expressions can use cached references.

        This routine stores the centroid coordinates and per-particle masses the first
        time it is called. Those cached values are consumed by force providers that
        need an equilibrium reference (e.g. :func:`anisotropic_ho`).  The defaults are
        deliberately conservative:

        • ``force=False`` keeps the very first reference geometry/mass table so that
          cloning a :class:`System` (as we do when spawning NE child trajectories)
          does *not* move the underlying potential minimum.
        • pass ``force=True`` only when you truly intend to reinitialize the reference,
          for example if you construct a *new* physical system with the same
          :class:`Force_Field` instance.

        Parameters
        ----------
        positions : ndarray
            Cartesian coordinates with shape ``(n_beads, n_particles * n_dim)``.
        masses : array_like
            Particle masses expressed in atomic units (one mass per particle).
        n_particles : int
            Number of physical particles.
        n_dim : int
            Number of spatial dimensions per particle.
        force : bool, optional
            Overwrite any cached reference geometry/masses. Defaults to ``False``.
        """
        centroid = np.mean(positions, axis=0).reshape(n_particles, n_dim)

        if self._reference_positions is not None and not force:
            # When cloning the parent System for child runs we do not want to
            # update the cached reference positions/masses, otherwise the HO
            # minima move with every child launch. Guard against incompatible
            # shapes so accidental reuse is caught early.
            current_shape = self._reference_positions.shape
            incoming_shape = centroid.shape
            if current_shape != incoming_shape:
                raise ValueError(
                    "Force_Field.register_system called with incompatible shape "
                    f"(cached {current_shape}, new {incoming_shape}). "
                    "Pass force=True if you really need to reinitialize."
                )
            self._log(
                "register_system called again; preserving cached reference geometry/masses."
            )
            return

        self._reference_positions = centroid.copy()
        self._masses = np.asarray(masses, float).reshape(n_particles)
        self._n_particles = n_particles
        self._n_dim = n_dim
        self._log(
            f"Registered system: n_particles={n_particles}, n_dim={n_dim}, "
            f"masses={self._masses}, reference_centroid_shape={self._reference_positions.shape}"
        )

    def harmonic(self, q, n_beads, n_particles, n_dim, params, debug=False):
        """
        Harmonic oscillator defined by per-dimension frequencies (in a.u.).

        The input parameters are interpreted as angular frequencies ω (not spring
        constants).  The restoring force is constructed as

            F = -m ω^2 q

        so users can pass the raw frequencies (e.g. ω_A=0.5) exactly as reported
        in the paper examples without manually squaring them.  If fewer than
        n_dim frequencies are provided the first value is broadcast to all
        dimensions.
        """
        if not params:
            raise ValueError("harmonic potential requires at least one frequency value.")

        freqs = np.asarray(params, dtype=np.float64).ravel()
        if freqs.size == 1:
            freqs = np.repeat(freqs, n_dim)
        elif freqs.size != n_dim:
            raise ValueError(
                f"harmonic expects either 1 or {n_dim} frequency values; received {freqs.size}."
            )
        if np.any(freqs <= 0):
            raise ValueError("harmonic frequencies must be positive.")

        omega2 = freqs**2  # element-wise for each dimension
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        if self._masses is None:
            masses = np.ones(n_particles, dtype=np.float64)
        else:
            masses = self._masses

        if debug:
            self._log(f"harmonic frequencies={freqs}, omega^2={omega2}")

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]
            mass_i = masses[i]
            forces[:, start:end] = -(mass_i * omega2) * q_i
            potential_energies[:, i] = 0.5 * mass_i * np.sum(omega2 * q_i**2, axis=1)

        return forces, potential_energies

    def double_well(self, q, n_beads, n_particles, n_dim, params, debug=False):
        D = params[0] if params else 1.0
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)
        if debug:
            self._log(f"double_well: D={D}")

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]
            forces[:, start:end] = -D * (4 * q_i**3 - 2 * q_i)
            potential_energies[:, i] = D * np.sum(q_i**4 - q_i**2, axis=1)

        return forces, potential_energies

    def double_well_asym(self, q, n_beads, n_particles, n_dim, params, debug=False):
        D = params[0] if params else 1.0
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)
        if debug:
            self._log(f"double_well_asym: D={D}")

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]
            forces[:, start:end] = -D * (4 * q_i**3 - 2 * q_i + 1)
            potential_energies[:, i] = D * np.sum(q_i**4 - q_i**2 + q_i, axis=1)

        return forces, potential_energies

    def anisotropic_ho(self, q, n_beads, n_particles, n_dim, params, debug=False):
        """
        Anisotropic harmonic oscillator that only requires per-dimension
        frequencies (cm^-1). Equilibrium positions/masses are inferred from the
        registered system just like the 1D harmonic uses the origin.
        """
        if params is None or len(params) < n_dim:
            raise ValueError(f"anisotropic_ho requires {n_dim} frequency values (cm^-1).")
        if self._reference_positions is None or self._masses is None:
            raise RuntimeError("Force field references not initialized; call register_system first.")

        freq = np.asarray(params[:n_dim], float)
        if np.any(freq <= 0):
            raise ValueError("Frequencies for anisotropic_ho must be positive.")

        omega2 = (freq * CM1_TO_AU_ANG_FREQ)**2
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]
            dq = q_i - self._reference_positions[i]
            mass_i = self._masses[i]
            forces[:, start:end] = -mass_i * omega2 * dq
            potential_energies[:, i] = 0.5 * mass_i * np.sum(omega2 * dq**2, axis=1)
            if debug:
                self._log(
                    f"anisotropic_ho particle {i}: mass={mass_i}, omega^2={omega2}, "
                    f"dq_mean={np.mean(dq, axis=0)}"
                )

        return forces, potential_energies

    def morse_harmonic(self, q, n_beads, n_particles, n_dim, params, debug=False):
        """
        Composite potential: quartic expansion of Morse along z plus harmonic in x/y.

        Parameters (expects notebook/i-PI ordering):
            De [eV], a [1/Å], z0 [Å], k [eV/Å^2]

        Default (if params empty) falls back to the i-PI driver values in atomic units:
            De = 0.00735 Ha, a = 1.61 1/Bohr, z0 = 2.1 Bohr, k = 0.343 Ha/Bohr^2

        Notes
        -----
        - q is in Bohr; energies returned in Hartree.
        - Only the first three Cartesian components are used (x,y,z); raises if n_dim < 3.
        """
        if n_dim < 3:
            raise ValueError("morse_harmonic requires at least 3D coordinates (x, y, z).")

        # Handle parameters and unit conversion
        if params is None or len(params) == 0:
            De = 0.00735  # Ha
            a = 1.61      # 1/Bohr
            z0 = 2.1      # Bohr
            k = 0.343     # Ha/Bohr^2
        elif len(params) >= 4:
            De = float(params[0]) * EV_TO_HA
            # 1/Å -> 1/Bohr (divide by Å→Bohr)
            a = float(params[1]) / ANGSTROM_TO_BOHR
            z0 = float(params[2]) * ANGSTROM_TO_BOHR
            k = float(params[3]) * EV_TO_HA / (ANGSTROM_TO_BOHR ** 2)
        else:
            raise ValueError("morse_harmonic expects parameters [De(eV), a(1/Å), z0(Å), k(eV/Å^2)].")

        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        # Loop over particles (same functional form applied to each)
        for i in range(n_particles):
            start = i * n_dim
            end = start + n_dim
            q_i = q[:, start:end]

            # x/y harmonic part
            xy = q_i[:, :2]
            potential_energies[:, i] = 0.5 * k * np.sum(xy ** 2, axis=1)
            forces[:, start:start + 2] = -k * xy

            # Morse-like quartic expansion along z
            z = (q_i[:, 2] - z0) * a  # dimensionless displacement
            potential_energies[:, i] += De * (z ** 2 - z ** 3 + (7.0 / 12.0) * z ** 4)
            forces[:, start + 2] = -De * a * (2.0 * z - 3.0 * z ** 2 + (7.0 / 3.0) * z ** 3)

            if debug:
                self._log(
                    f"morse_harmonic particle {i}: De={De:.6f} Ha, a={a:.4f} 1/Bohr, "
                    f"z0={z0:.4f} Bohr, k={k:.4f} Ha/Bohr^2"
                )

        return forces, potential_energies

    def shifted_harmonic(self, q, n_beads, n_particles, n_dim, params, debug=False):
        """
        3D shifted harmonic oscillator:
        V = 1/2 K |r - d|^2

        params = [K, dx, dy, dz]

        Works for any number of particles (each gets same K and shift unless extended).
        """
        if params is None or len(params) < n_dim + 1:
            raise ValueError("shifted_harmonic requires [K, dx, dy, dz] (n_dim+1 parameters).")

        K = float(params[0])
        shift = np.asarray(params[1:1+n_dim], float)  # (n_dim,)

        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        # Loop over particles
        for i in range(n_particles):
            start = i * n_dim
            end   = (i + 1) * n_dim

            q_i = q[:, start:end]  # shape: (n_beads, n_dim)

            # r - d
            dq = q_i - shift

            # Forces: -K (r - d)
            forces[:, start:end] = -K * dq

            # V = 1/2 K |r - d|^2, per bead, per particle
            potential_energies[:, i] = 0.5 * K * np.sum(dq**2, axis=1)
            if debug:
                self._log(
                    f"shifted_harmonic particle {i}: K={K}, shift={shift}, "
                    f"dq_mean={np.mean(dq, axis=0)}"
                )

        return forces, potential_energies

    def zero_force(self,q,n_beads, n_particles, n_dim, debug=False):
        forces = np.zeros((n_beads, n_dim*n_particles),float)
        potential_energies = np.zeros((n_beads,n_particles),float)
        if debug:
            self._log("zero_force called")
        return(forces,potential_energies)

    def ff(self, q, n_beads, n_particles, n_dim, type, params):
        debug_flag = self.debug

        if type == 'harmonic':
            forces, potential_energies = self.harmonic(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == 'double_well':
            forces, potential_energies = self.double_well(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == 'double_well_asym':
            forces, potential_energies = self.double_well_asym(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == 'anisotropic_ho':
            forces, potential_energies = self.anisotropic_ho(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == 'harmonic_shifted':
            forces, potential_energies = self.shifted_harmonic(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == 'morse_harmonic':
            forces, potential_energies = self.morse_harmonic(q, n_beads, n_particles, n_dim, params, debug=debug_flag)

        elif type == "zero":
            forces, potential_energies = self.zero_force(q, n_beads, n_particles, n_dim, debug=debug_flag)

        else:
            raise ValueError("Unknown force field type: {}".format(type))

        return forces, potential_energies

    def eq_force_function(self, q, n_beads, n_particles, n_dim):
        potential_type = self.potentials[0]['type']
        params = self.potentials[0]['args']
        #print(f"Potential type: {potential_type}, Parameters: {params}")
        forces, potential_energies = self.ff(q, n_beads, n_particles, n_dim, potential_type, params)

        return forces, potential_energies

    def noneq_force_function(self, q, n_beads, n_particles, n_dim, lambda_t):
        potential_type0 = self.potentials[0]['type']
        params0 = self.potentials[0]['args']
        potential_type1 = self.potentials[1]['type']
        params1 = self.potentials[1]['args']

        forces_0,potential_energies_0 = self.ff(q, n_beads, n_particles, n_dim, potential_type0, params0)
        forces_1,potential_energies_1 = self.ff(q, n_beads, n_particles, n_dim, potential_type1, params1)

        forces = (lambda_t)*forces_1 + (1 - lambda_t)*forces_0
        potential_energies = (lambda_t)*potential_energies_1 + (1 - lambda_t)*potential_energies_0

        return forces, potential_energies
