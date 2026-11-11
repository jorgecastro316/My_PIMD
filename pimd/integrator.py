from pimd.utils import *
from pimd.utils import _C_matrix_cached
class Integrator(object):
    def __init__(self, temperature, timestep, force_field, mode=None, timer=None, n_beads=None):
        self.temperature = temperature
        self.timestep    = timestep
        self.force_field = force_field
        self.mode        = mode
        self.timer       = timer
        self.n_beads    = n_beads
        #print('Integrator initialized with temperature:', temperature, 'and timestep:', timestep)


        # Cache C and bind fast transforms (only if n_beads is known)
        if n_beads is not None:
            C = _C_matrix_cached(n_beads)
            self._rr2nm = lambda X: C @ X
            self._nm2rr = lambda X: C.T @ X

            self._beta_n  = temperature_to_beta(self.temperature)
            self._omega_k = omega_Cmatrix(n_beads, self._beta_n)
        else:
            self._rr2nm = rr2nm
            self._nm2rr = nm2rr
            self._beta_n = None
            self._omega_k = None


    def Propagate_Ring_Polymer(self, p, q, f, n_beads, n_atoms, n_dim, masses, lambda_t=0.0):
        dt = self.timestep

        # — 1) First half‐kick under external forces —
        #print("p before first half-step:", p)
        #print ("DEBUG p before first half-step:", p)
        #print ("DEBUG q before first half-step:", q)
        #print ("DEBUG f before first half-step:", f)
        if self.mode in ('SPIMD', 'spimd'):
            with (self.timer.section("noneq_force_function (pre)") if self.timer else _nullctx()):
                f_start, _ = self.force_field.noneq_force_function(
                    q, n_beads, n_atoms, n_dim, lambda_t
                )
        else:
            f_start = f

        with (self.timer.section("kick half_1") if self.timer else _nullctx()):
            p = p + 0.5 * f_start * dt
        #print("p  after first half-step:", p)

        # — 2) Transform Cartesian → normal modes (using the 1D‐based routines) —
        with (self.timer.section("rr2nm(p,q)") if self.timer else _nullctx()):
            q_nm = self._rr2nm(q)
            p_nm = self._rr2nm(p)
        #print("DEBUG p_nm (pre-prop):", p_nm)
        #print("DEBUG q_nm (pre-prop):", q_nm)

        # — 3) Build beta using kB_au, then W[k] using the reference formula —
        with (self.timer.section("omega_Cmatrix/beta") if self.timer else _nullctx()):
            beta_n  = self._beta_n  if self._beta_n  is not None else temperature_to_beta(self.temperature)
            omega_k = self._omega_k if self._omega_k is not None else omega_Cmatrix(n_beads, beta_n)


        # — 4) Free‐RP propagation in normal modes —
        with (self.timer.section("propagate_normal_modes") if self.timer else _nullctx()):
            # Ensure correct dtypes for numba
            p_nm = np.asarray(p_nm, dtype=np.float64)
            q_nm = np.asarray(q_nm, dtype=np.float64)
            omega_k = np.asarray(omega_k, dtype=np.float64)
            masses = np.asarray(masses, dtype=np.float64)
            p_nm2, q_nm2 = propagate_normal_modes_general(p_nm, q_nm, omega_k, dt, masses, n_dim)
        #print("DEBUG p_nm (post-prop):", p_nm2)
        #print("DEBUG q_nm (post-prop):", q_nm2)

        # — 5) Transform back to Cartesian —
        with (self.timer.section("nm2rr(p,q)") if self.timer else _nullctx()):
            p = self._nm2rr(p_nm2)
            q = self._nm2rr(q_nm2)
        #print("p after back-transform:", p)
        #print("q after back-transform:", q)

        # — 6) Second half‐kick under external forces —

        if self.mode == 'SPIMD' or self.mode == 'spimd':
            with (self.timer.section("noneq_force_function") if self.timer else _nullctx()):
                forces, potential_energies = self.force_field.noneq_force_function(
                    q, n_beads, n_atoms, n_dim, lambda_t)
        else:
            with (self.timer.section("eq_force_function") if self.timer else _nullctx()):
                forces, potential_energies = self.force_field.eq_force_function(
                    q, n_beads, n_atoms, n_dim)

        with (self.timer.section("kick half_2") if self.timer else _nullctx()):
            p = p + 0.5 * forces * dt
        #print("p after second half-step:", p)

        return p, q, forces, potential_energies


    def step_NVE(self, system, lambda_t=0.0):
        q       = system.positions
        v       = system.velocities
        f       = system.forces
        n_dim   = system.n_dim
        n_atoms = system.n_atoms
        n_beads = system.n_beads
        masses  = system.masses

        # Convert velocities to momenta
        with (self.timer.section("velocities_to_momenta") if self.timer else _nullctx()):
            p = velocities_to_momenta(v, masses, n_atoms, n_dim)

        #print("Momenta before propagation: ",p)
        # PIMD propagation step
        p, q, f, potential_energies = self.Propagate_Ring_Polymer(
            p, q, f, n_beads, n_atoms, n_dim, masses,lambda_t=lambda_t
        )

        # Convert momenta to velocities
        with (self.timer.section("momenta_to_velocities") if self.timer else _nullctx()):
            v = momenta_to_velocities(p, masses, n_atoms, n_dim)

        # Compute centroid, energies, etc. (as before)

        qc = np.mean(q, axis=0, keepdims=True)
        vc = np.mean(v, axis=0, keepdims=True)
        #print(np.shape(q), np.shape(v),np.shape(qc), np.shape(f))
        with (self.timer.section("compute_energetics") if self.timer else _nullctx()):
            pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                q, v, qc, f, potential_energies,
                masses, self.temperature,n_beads, n_atoms, n_dim
            )

        # Update the system object
        with (self.timer.section("update_system") if self.timer else _nullctx()):
            update_system(
                system, q, v, qc,vc,f, system_T,
                pot, kin, etot, E_pot, E_kin, E_tot, internal_pot, potential_energies
            )

    def step_PILE_L(self, system, thermostat, lambda_t=0.0):
        q       = system.positions           # (n_beads, n_atoms, n_dim)
        v       = system.velocities          # (n_beads, n_atoms, n_dim)
        f       = system.forces              # (n_beads, n_atoms * n_dim)
        n_dim   = system.n_dim
        n_atoms = system.n_atoms
        n_beads = system.n_beads
        masses  = system.masses              # length‐n_atoms array
        thermostat.set_PILE_L_coeffs(n_beads, self.timestep)

        # 1) Convert velocities to momenta (shape (n_beads, n_atoms*n_dim))
        with (self.timer.section("velocities_to_momenta") if self.timer else _nullctx()):

            p = velocities_to_momenta(v, masses, n_atoms, n_dim)

        # 2) FIRST HALF-STEP OF PILE-L in normal-mode space (vectorized transform + loop over columns):
        with (self.timer.section("first rr2nm(p,q) PILE L ") if self.timer else _nullctx()):
            p_nm = self._rr2nm(p)  # shape (n_beads, n_atoms * n_dim)

        with (self.timer.section("first Thermalization PILE L") if self.timer else _nullctx()):
            for col in range(n_atoms * n_dim):
                atom_idx = col // n_dim
                # Use cached vectorized version
                p_nm[:, col] = thermostat.PILE_L(
                    p_nm[:, col],     # shape: (n_beads,)
                    self.timestep,
                    masses[atom_idx]
                )

        #print('after 1st',p_nm)
        with (self.timer.section("first nm2rr(p,q) PILE L ") if self.timer else _nullctx()):
            p_first_half = self._nm2rr(p_nm)  # back to (n_beads, n_atoms * n_dim)
        #p_first_half = p
        #print("p_flat after first half-step PILE_L:", np.shape(p_flat))
        # 3) RING-POLYMER PROPAGATION

        p_new, q_new, f_new, potential_energies = self.Propagate_Ring_Polymer(
            p_first_half, q, f,n_beads, n_atoms, n_dim, masses,lambda_t=lambda_t)

        # 4) SECOND HALF-STEP OF PILE-L (same pattern)
        with (self.timer.section("second rr2nm(p,q) PILE L ") if self.timer else _nullctx()):
            p_nm_new = self._rr2nm(p_new)
        #print('before 2nd',p_nm_new)

        with (self.timer.section("second Thermalization PILE L") if self.timer else _nullctx()):
            for col in range(n_atoms * n_dim):
                atom_idx = col // n_dim
                # Use cached vectorized version
                p_nm_new[:, col] = thermostat.PILE_L(
                    p_nm_new[:, col],     # shape: (n_beads,)
                    self.timestep,
                    masses[atom_idx]
                )

        #print('after 2nd',p_nm_new)
        with (self.timer.section("second nm2rr(p,q) PILE L ") if self.timer else _nullctx()):
            p_second_half = self._nm2rr(p_nm_new)

        # 5) Convert momenta → velocities, compute centroid/energetics, update system
        v_new = momenta_to_velocities(p_second_half, masses, n_atoms, n_dim)
        #print('v new',v_new)
        qc = np.mean(q_new, axis=0, keepdims=True)
        vc = np.mean(v_new, axis=0, keepdims=True)
        with (self.timer.section("compute_energetics") if self.timer else _nullctx()):
            pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                q_new, v_new, qc, f_new, potential_energies,
                masses, self.temperature, n_beads, n_atoms, n_dim
            )
        with (self.timer.section("update_system") if self.timer else _nullctx()):
            update_system(
                system, q_new, v_new, qc, vc, f_new, system_T,
                pot, kin, etot, E_pot, E_kin, E_tot, internal_pot, potential_energies
            )
        return

    def step_PILE_G(self,system,thermostat,lambda_t = 0.0):

        q       = system.positions           # (n_beads, n_atoms, n_dim)
        v       = system.velocities          # (n_beads, n_atoms, n_dim)
        f       = system.forces              # (n_beads, n_atoms * n_dim)
        n_dim   = system.n_dim
        n_atoms = system.n_atoms
        n_beads = system.n_beads
        masses  = system.masses              # length‐n_atoms array
        dof = n_atoms * n_dim
        thermostat.set_PILE_G_coeffs(n_beads=self.n_beads, dt=self.timestep)

        # 1) Convert velocities to momenta (shape (n_beads, n_atoms*n_dim))

        p = velocities_to_momenta(v, masses, n_atoms, n_dim)

        # 2)  First HALF-STEP OF PILE-G
        p_nm = self._rr2nm(p)
        #Compute K from the momenta of the centroid for each atom
        with (self.timer.section("first kinetic matrix K") if self.timer else _nullctx()):
            p0 = p_nm[0]  # shape: (n_atoms * n_dim,)
            masses_expanded = np.repeat(masses, n_dim)  # shape: (n_atoms * n_dim,)
            K = 0.5 * np.sum((p0**2) / masses_expanded)

        #print('K0', K)
        #Thermalization of momenta 1st half-step
        with (self.timer.section("first Thermalization PILE G") if self.timer else _nullctx()):
            for col in range(n_atoms * n_dim):
                atom_idx = col // n_dim
                p_nm[:, col] = thermostat.PILE_G(
                    p_nm[:, col],                # shape: (n_beads,)
                    masses[atom_idx],
                    dof,
                    K
                )

        p_first_half = self._nm2rr(p_nm)
        #print('p_first_half',p_first_half)
        # 3) RING-POLYMER PROPAGATION

        p_new, q_new, f_new, potential_energies = self.Propagate_Ring_Polymer(
            p_first_half, q, f,n_beads, n_atoms, n_dim, masses,lambda_t=lambda_t)

        # 4) SECOND HALF-STEP OF PILE-L (same pattern)
        p_nm_new = self._rr2nm(p_new)
        #Compute K from the momenta of the centroid for each atom
        with (self.timer.section("second kinetic matrix K") if self.timer else _nullctx()):
            p0 = p_nm_new[0]  # shape: (n_atoms * n_dim,)
            masses_expanded = np.repeat(masses, n_dim)  # shape: (n_atoms * n_dim,)
            K = 0.5 * np.sum((p0**2) / masses_expanded)

        #print('before 2nd',p_nm_new)
        with (self.timer.section("second Thermalization PILE G") if self.timer else _nullctx()):
            for col in range(n_atoms * n_dim):
                atom_idx = col // n_dim
                p_nm_new[:, col] = thermostat.PILE_G(
                    p_nm_new[:, col],                # shape: (n_beads,)
                    masses[atom_idx],
                    dof,
                    K
                )
        #print('after 2nd',p_nm_new)
        p_second_half = self._nm2rr(p_nm_new)

        # 5) Convert momenta → velocities, compute centroid/energetics, update system
        v_new = momenta_to_velocities(p_second_half, masses, n_atoms, n_dim)
        #print('v new',v_new)
        qc = np.mean(q_new, axis=0, keepdims=True)
        vc = np.mean(v_new, axis=0, keepdims=True)

        with (self.timer.section("compute_energetics") if self.timer else _nullctx()):
            pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                q_new, v_new, qc, f_new, potential_energies,
                masses, self.temperature, n_beads, n_atoms, n_dim
            )

        with (self.timer.section("update_system") if self.timer else _nullctx()):
            update_system(
                system, q_new, v_new, qc, vc, f_new, system_T,
                pot, kin, etot, E_pot, E_kin, E_tot, internal_pot,potential_energies
            )
        return

    def step_Andersen(self, system, thermostat, lambda_t=0.0):
        q       = system.positions           # (n_beads, n_atoms, n_dim)
        v       = system.velocities          # (n_beads, n_atoms, n_dim)
        f       = system.forces              # (n_beads, n_atoms * n_dim)
        n_dim   = system.n_dim
        n_atoms = system.n_atoms
        n_beads = system.n_beads
        masses  = system.masses              # length‐n_atoms array

        # 1) Convert velocities → flattened momenta (shape (n_beads, n_atoms*n_dim))
        p = velocities_to_momenta(v, masses, n_atoms, n_dim)
        #print("shape p ",np.shape(p))
        p_nm = rr2nm(p)
        p = nm2rr(p_nm)
        # 2) FIRST HALF-STEP
        p_first_half = thermostat.Andersen(p, masses, self.timestep)

        # 3) RING-POLYMER PROPAGATION

        p_new, q_new, f_new, potential_energies = self.Propagate_Ring_Polymer(
            p_first_half, q, f,n_beads, n_atoms, n_dim, masses,lambda_t=lambda_t)

        p_2nd_nm = rr2nm(p_new)
        p_new = nm2rr(p_2nd_nm)
        # 4) SECOND HALF-STEP
        p_second_half = thermostat.Andersen(p_new, masses, self.timestep)
        # 5) Convert momenta → velocities, compute centroid/energetics, update system

        v_new = momenta_to_velocities(p_second_half, masses, n_atoms, n_dim)

        qc = np.mean(q_new, axis=0, keepdims=True)
        vc = np.mean(v_new, axis=0, keepdims=True)

        pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
            q_new, v_new, qc, f_new, potential_energies,
            masses, self.temperature, n_beads, n_atoms, n_dim
        )

        update_system(
            system, q_new, v_new, qc, vc, f_new, system_T,
            pot, kin, etot, E_pot, E_kin, E_tot, internal_pot,potential_energies
        )
        return
