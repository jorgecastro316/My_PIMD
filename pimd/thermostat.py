import numpy as np
from pimd.utils import *

class Thermostat(object):
    def __init__(self, target_temperature=300, thermostat_strenght=10000, thermostat_type = None, rng=None):
        self.temperature = target_temperature #Temperatue at which the simulation is run -> system temperature * n_beads
        self.tau = thermostat_strenght
        self.thermostat_type = thermostat_type
        self.rng = rng
        #self.rng = np.random.mtrand.RandomState(seed)

    def _precompute_PILE_L_coeffs(self, n_beads):
        beta_n = temperature_to_beta(self.temperature)
        Wp = 1 / beta_n
        Wk_vec = np.array([2 * Wp * np.sin(k * np.pi / n_beads) for k in range(n_beads)])
        r_k_vec = np.array([1/self.tau if k == 0 else 2 * Wk_vec[k] for k in range(n_beads)])
        C1_vec = np.exp(-0.5 * self.tau_inv_dt * r_k_vec)
        C2_vec = np.sqrt(1.0 - C1_vec ** 2)
        return C1_vec, C2_vec, beta_n

    def set_PILE_L_coeffs(self, n_beads, dt):
        self.tau_inv_dt = dt
        self.C1_vec, self.C2_vec, self.beta_n = self._precompute_PILE_L_coeffs(n_beads)


    def PILE_L(self, p_nm_col: np.ndarray, time_step: float, mass: float) -> np.ndarray:
        # Ensure 1D input
        assert p_nm_col.ndim == 1, f"Expected 1D input, got {p_nm_col.shape}"

        beta_n = temperature_to_beta(self.temperature)
        sqrt_m_over_beta = np.sqrt(mass / beta_n)

        # Generate noise of same shape
        E_noise = self.rng.standard_normal(p_nm_col.shape)  # shape: (n_beads,)

        # Apply coefficients
        return self.C1_vec * p_nm_col + sqrt_m_over_beta * self.C2_vec * E_noise


    def _precompute_PILE_G_coeffs(self, n_beads):
        beta_n = temperature_to_beta(self.temperature)
        Wp = 1 / beta_n
        Wk_vec = np.array([2 * Wp * np.sin(k * np.pi / n_beads) for k in range(n_beads)])
        r_k_vec = np.array([1 / self.tau if k == 0 else 2 * Wk_vec[k] for k in range(n_beads)])
        C1_G_vec = np.exp(-0.5 * self.tau_inv_dt * r_k_vec)
        C2_G_vec = np.sqrt(1.0 - C1_G_vec ** 2)
        return C1_G_vec, C2_G_vec, beta_n


    def set_PILE_G_coeffs(self, n_beads, dt):
        self.tau_inv_dt = dt
        self.C1_G_vec, self.C2_G_vec, self.beta_n = self._precompute_PILE_G_coeffs(n_beads)


    def PILE_G(self, p_nm_col: np.ndarray, mass: float, dof: int, K: int) -> np.ndarray:
        assert p_nm_col.ndim == 1, f"Expected 1D input, got {p_nm_col.shape}"
        n_beads = p_nm_col.shape[0]
        sqrt_m_over_beta = np.sqrt(mass / self.beta_n)

        # SVR thermostat for centroid mode
        C = self.C1_G_vec[0]
        r1 = self.rng.standard_normal()
        rg = self.rng.chisquare(dof - 1) if dof > 1 else 0.0
        alpha2 = (
            C
            + ((1 - C) * (r1**2 + rg)) / (2 * K * self.beta_n)
            + 2 * r1 * np.sqrt(C * (1 - C) / (2 * self.beta_n * K))
        )
        alpha = np.sqrt(alpha2)
        threshold = r1 + np.sqrt(2 * K / (0.5 * (1 / self.beta_n)) * C / (1 - C))
        if threshold < 0:
            alpha *= -1

        # Initialize new momentum
        new_p_nm = np.empty_like(p_nm_col)

        # Centroid mode
        new_p_nm[0] = alpha * p_nm_col[0]

        # Remaining modes: Langevin-like thermalization
        E_noise = self.rng.standard_normal(n_beads - 1)
        new_p_nm[1:] = (
            self.C1_G_vec[1:] * p_nm_col[1:]
            + sqrt_m_over_beta * self.C2_G_vec[1:] * E_noise
        )

        return new_p_nm


    def Andersen(self,p, masses, dt):
        """
        Applies Andersen thermostat to a copy of ring-polymer momenta.

        Parameters
        ----------
        p : ndarray, shape (nbeads, natoms * ndim)
            Original momenta of all beads in atomic units.
        masses : ndarray, shape (natoms,)
            Masses of atoms in atomic units.
        dt : float
            Time step in atomic units.

        Returns
        -------
        p_new : ndarray, shape (nbeads, natoms * ndim)
            Momentas after Andersen thermostat collisions.
        """
        nu = 1/self.tau
        if self.rng is None:
            rng = np.random.default_rng()
        else:
            rng = self.rng

        nbeads, dof = p.shape
        natoms = len(masses)
        ndim = dof // natoms

        mvec = np.repeat(masses, ndim)
        beta_n = temperature_to_beta(self.temperature)
        k_B_Hartree = 1.0 / temperature_to_beta(1.0)

        # Standard deviation of the Maxwell-Boltzmann distribution
        stddev = np.sqrt(mvec * k_B_Hartree * self.temperature)

        p_new = p.copy()

        for b in range(nbeads):
            for i in range(dof):
                if rng.random() < nu * dt:
                    p_new[b, i] = rng.normal(0.0, stddev[i])

        return p_new
