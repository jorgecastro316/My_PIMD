#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:18:17 2022

@author: castrojo
"""
from pimd.integrator import Integrator
from pimd.utils import *
from pimd.file_recorder import Recorder
import numpy as np
import copy
import os
#from velocity_read import *

class Simulation(object):
    #Constructor
    def __init__(self,system,temperature,timestep,force_field,thermostat=None,mode=None,prefix='simulation',stride_out=1000,stride_pos=1000,stride_vel=1000,stride_log=1000):
        self.initializer(system,temperature,timestep,force_field,thermostat,mode,prefix,stride_out,stride_pos,stride_vel,stride_log)

    def initializer(self,system,temperature,timestep,force_field,thermostat=None,mode=None,prefix='simulation',stride_out=1000,stride_pos=1000,stride_vel=1000,stride_log=1000):
        self.force_field = force_field
        self.timestep = timestep
        self.temperature = temperature
        self.thermostat = thermostat
        self.mode = mode
        self.prefix = prefix
        #print(f"Simulation prefix: {self.prefix}")
        self.system = system
        self.n_beads = system.n_beads
        self.stride_out = stride_out
        self.stride_pos = stride_pos
        self.stride_vel = stride_vel
        self.stride_log = stride_log
        n_atoms = system.n_atoms
        n_dim = system.n_dim
        init_q = system.positions
        init_v = system.velocities
        init_qc = system.q_centroid
        masses = system.masses

        self.force_field.register_system(init_q, masses, n_atoms, n_dim)
        init_f, init_potential_energies = self.force_field.eq_force_function(init_q,self.n_beads,n_atoms,n_dim)
        init_system_T = instantaneous_temperature(init_v,masses,self.n_beads,n_atoms,n_dim)
        init_kin = classical_kinetic_energy(init_v, masses, n_atoms, n_dim)/self.n_beads
        init_internal_pot = Internal_Potential(init_q,self.temperature,masses,self.n_beads,n_atoms,n_dim)/self.n_beads
        init_external_pot = classical_potential_energy(init_potential_energies)/self.n_beads
        init_pot = init_external_pot + init_internal_pot

        init_E_kin, init_E_pot, init_E_tot = E_Estimators(temperature, init_f, init_q, init_qc, init_potential_energies,self.n_beads, n_atoms, n_dim)
        self.system.set_forces(init_f)
        self.system.set_pot(init_pot)
        self.system.set_kin(init_kin)
        self.system.set_tot(init_pot+init_kin)
        self.system.set_E_pot(init_E_pot)
        self.system.set_E_kin(init_E_kin)
        self.system.set_E_tot(init_E_tot)
        self.system.set_potential_energies(init_potential_energies)
        self.system.set_Temperature(init_system_T)

    def _ensure_finite_state(self, step, elapsed_time, context):
        """Abort the simulation if any tracked quantity contains NaN/inf."""
        checks = (
            ("positions", self.system.positions),
            ("velocities", self.system.velocities),
            ("forces", getattr(self.system, "forces", None)),
            ("q_centroid", getattr(self.system, "q_centroid", None)),
            ("v_centroid", getattr(self.system, "v_centroid", None)),
            ("potential_energies", getattr(self.system, "potential_energies", None)),
            ("kinetic_energies", getattr(self.system, "kinetic_energies", None)),
            ("kin", getattr(self.system, "kin", None)),
            ("pot", getattr(self.system, "pot", None)),
            ("tot", getattr(self.system, "tot", None)),
            ("E_kin", getattr(self.system, "E_kin", None)),
            ("E_pot", getattr(self.system, "E_pot", None)),
            ("E_tot", getattr(self.system, "E_tot", None)),
            ("Temperature", getattr(self.system, "Temperature", None)),
        )

        issues = []
        for name, value in checks:
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            finite_mask = np.isfinite(arr)
            if finite_mask.all():
                continue
            total_bad = int(arr.size - finite_mask.sum())
            num_nan = int(np.isnan(arr).sum())
            num_inf = int(np.isinf(arr).sum())
            issues.append(
                f"{name}: bad={total_bad} (NaN={num_nan}, Inf={num_inf})"
            )

        if issues:
            detail = "; ".join(issues)
            raise RuntimeError(
                f"[EMERGENCY STOP] Non-finite values detected during {context} at step "
                f"{step} (t={elapsed_time:.6f}). {detail}"
            )

    def run_PIMD(self, nsteps, record_bead_positions=None):
        counter = 0
        time_counter = 0

        timer = PerfBuckets(report_every=self.stride_log, name="PIMD")
        integrator = Integrator(self.temperature, self.timestep, self.force_field, self.mode,
                                timer=timer, n_beads=self.n_beads)

        rec_energies = None
        rec_trajectories = None
        rec_velocities = None

        # --- Initialize stride-controlled recorders ---
        if self.stride_out > 0:
            rec_energies = Recorder(self.system, self.prefix)
        if self.stride_pos > 0:
            #print(self.prefix,'_trajectories')
            rec_trajectories = Recorder(self.system, self.prefix + '_trajectories')
        if self.stride_vel > 0:
            rec_velocities = Recorder(self.system, self.prefix + '_velocities')

        # --- Normalize record_bead_positions ---
        if record_bead_positions in ("All", "all"):
            record_bead_positions = list(range(self.n_beads))
        elif record_bead_positions is None:
            record_bead_positions = []

        self._ensure_finite_state(counter, time_counter, "PIMD initial state")

        # --- Initial write ---
        with timer.section("rec.write"):
            if rec_energies is not None:
                rec_energies.write(self.system, counter, time_counter, record_energies=True, record_estimators=True)
            if rec_trajectories is not None:
                rec_trajectories.write(self.system, counter, time_counter,
                                    record_bead_positions=record_bead_positions,
                                    record_centroid_position=True)
            if rec_velocities is not None:
                rec_velocities.write(self.system, counter, time_counter,
                                    record_bead_velocities=record_bead_positions,
                                    record_centroid_velocity=True)

        # --- Main loop ---
        while counter < nsteps:
            if counter % 100000 == 0:
                print('Step:', counter)

            with timer.section("integrator_step"):
                if str(self.thermostat.thermostat_type) == 'PILE_L':
                    integrator.step_PILE_L(self.system, self.thermostat)
                elif str(self.thermostat.thermostat_type) == 'PILE_G':
                    integrator.step_PILE_G(self.system, self.thermostat)
                elif str(self.thermostat.thermostat_type) == 'Andersen':
                    integrator.step_Andersen(self.system, self.thermostat)
                else:
                    integrator.step_NVE(self.system)

            counter += 1
            time_counter += self.timestep
            self._ensure_finite_state(counter, time_counter, "PIMD dynamics")

            # --- Stride-controlled output ---
            if (
                (self.stride_out > 0 and counter % self.stride_out == 0) or
                (self.stride_pos > 0 and counter % self.stride_pos == 0) or
                (self.stride_vel > 0 and counter % self.stride_vel == 0)
            ):
                with timer.section("rec.write"):
                    if self.stride_out > 0 and counter % self.stride_out == 0:
                        rec_energies.write(self.system, counter, time_counter,
                                        record_energies=True, record_estimators=True)
                    if self.stride_pos > 0 and counter % self.stride_pos == 0:
                        rec_trajectories.write(self.system, counter, time_counter,
                                            record_bead_positions=record_bead_positions,
                                            record_centroid_position=True)
                    if self.stride_vel > 0 and counter % self.stride_vel == 0:
                        rec_velocities.write(self.system, counter, time_counter,
                                            record_bead_velocities=record_bead_positions,
                                            record_centroid_velocity=True)

            timer.step_done()


    def run_SPIMD(self, nsteps, record_bead_positions=None, compute_work=False):
        counter = 0
        time_counter = 0
        timer = PerfBuckets(report_every=self.stride_log, name="SPIMD")
        integrator = Integrator(self.temperature, self.timestep, self.force_field, mode='spimd',
                                timer=timer, n_beads=self.n_beads)
        #print("timestep:",self.timestep)
        # Initialize recorders based on stride values
        rec_energies = None
        rec_trajectories = None
        rec_velocities = None

        if self.stride_out > 0:
            rec_energies = Recorder(self.system, self.prefix)

        if self.stride_pos > 0:
            rec_trajectories = Recorder(self.system, self.prefix + '_trajectories')
            if record_bead_positions == "All" or record_bead_positions == "all":
                record_bead_positions = list(range(self.n_beads))

        if self.stride_vel > 0:
            rec_velocities = Recorder(self.system, self.prefix + '_velocities')
            if record_bead_positions == "All" or record_bead_positions == "all":
                record_bead_positions = list(range(self.n_beads))

        self._ensure_finite_state(counter, time_counter, "SPIMD initial state")

        initial_energy = float(self.system.tot) if compute_work else None

        # Initial write (step = 0)
        with timer.section("rec.write"):
            if rec_energies is not None:
                rec_energies.write(
                    self.system, counter, time_counter,
                    record_energies=True,
                    record_estimators=True
                )
            if rec_trajectories is not None:
                rec_trajectories.write(
                    self.system, counter, time_counter,
                    record_bead_positions=record_bead_positions,
                    record_centroid_position=True
                )
            if rec_velocities is not None:
                rec_velocities.write(
                    self.system, counter, time_counter,
                    record_bead_velocities=record_bead_positions,
                    record_centroid_velocity=True
                )

        # Main loop
        #print(nsteps)
        #print(self.system.tot,self.system.pot,self.system.kin,self.system.positions,self.system.velocities)
        while counter < nsteps:
            #if counter % 100000 == 0:
            #    print("Step:", counter)

            counter += 1
            lambda_t = counter / nsteps
            print
            with timer.section("integrator_step"):
                if str(self.thermostat.thermostat_type) == 'PILE_L':
                    integrator.step_PILE_L(self.system, self.thermostat, lambda_t)
                elif str(self.thermostat.thermostat_type) == 'PILE_G':
                    integrator.step_PILE_G(self.system, self.thermostat, lambda_t)
                elif str(self.thermostat.thermostat_type) == 'Andersen':
                    integrator.step_Andersen(self.system, self.thermostat, lambda_t)
                else:
                    integrator.step_NVE(self.system, lambda_t)

            time_counter += self.timestep
            self._ensure_finite_state(counter, time_counter, "SPIMD dynamics")

            # Conditional recording based on stride
            with timer.section("rec.write"):
                if rec_energies is not None and counter % self.stride_out == 0:
                    rec_energies.write(
                        self.system, counter, time_counter,
                        record_energies=True,
                        record_estimators=True
                    )
                if rec_trajectories is not None and counter % self.stride_pos == 0:
                    rec_trajectories.write(
                        self.system, counter, time_counter,
                        record_bead_positions=record_bead_positions,
                        record_centroid_position=True
                    )
                if rec_velocities is not None and counter % self.stride_vel == 0:
                    rec_velocities.write(
                        self.system, counter, time_counter,
                        record_bead_velocities=record_bead_positions,
                        record_centroid_velocity=True
                    )
            #print(lambda_t,self.system.tot,self.system.pot,self.system.kin,self.system.positions,self.system.velocities)
            timer.step_done()
        #print(self.system.tot,self.system.pot,self.system.kin,self.system.positions,self.system.velocities)
        if compute_work:
            return float(self.system.tot) - initial_energy

    def run_NEQ_PIMD(self,
                    n_samples,
                    input,
                    record_endpoints=False,
                    record_bead_positions_parent=None,
                    record_bead_positions_child=None):
        counter = 0
        time_counter = 0

        timer = PerfBuckets(report_every=self.stride_log, name="NEQ-PIMD")

        # Parent integrator
        integrator_parent = Integrator(self.temperature, self.timestep, self.force_field, self.mode,
                                timer=timer, n_beads=self.n_beads)

        if record_bead_positions_parent == "All" or record_bead_positions_parent == "all":
            record_bead_positions_parent = list(range(self.n_beads))

        if record_bead_positions_child == "All" or record_bead_positions_child == "all":
            record_bead_positions_child = list(range(self.n_beads))


        # --- Recorders for parent ---
        rec_parent_energies = None
        rec_parent_trajectories = None
        rec_parent_velocities = None

        self._ensure_finite_state(counter, time_counter, "NEQ parent initial state")
        #print("Initial forces:", self.system.forces)
        with timer.section("rec.write"):
            if self.stride_out > 0:
                rec_parent_energies = Recorder(self.system, self.prefix)
                rec_parent_energies.write(self.system, counter, time_counter,
                                        record_energies=True, record_estimators=True)

            if self.stride_pos > 0:
                rec_parent_trajectories = Recorder(self.system, self.prefix)
                rec_parent_trajectories.write(self.system, counter, time_counter,
                                            record_bead_positions=record_bead_positions_parent,
                                            record_centroid_position=False)

            if self.stride_vel > 0:
                rec_parent_velocities = Recorder(self.system, self.prefix)
                rec_parent_velocities.write(self.system, counter, time_counter,
                                            record_bead_velocities=record_bead_positions_parent,
                                            record_centroid_velocity=False)

        # Child dynamics setup
        sample_steps = input.get('sample_steps', 1000)
        thermostat_type_child = input.get('thermostat_child', 'NVE')
        tau_child = input.get('tau_child', 1.0)
        n_steps_child = input.get('n_steps_child', 1000)
        n_steps_parent = n_samples * sample_steps
        stride_out_child = input.get('stride_out_child', 0)
        stride_pos_child = input.get('stride_pos_child', 0)
        stride_vel_child = input.get('stride_vel_child', 0)

        rec_work = Recorder(self.system, self.prefix)  # always created
        #print('Vergacion')
        rec_initial = None
        #rec_final = None
        if record_endpoints:
            rec_initial = Recorder(self.system, self.prefix + "_endpoint_initial")
            #rec_final = Recorder(self.system, self.prefix + '_final')
        # === Main parent simulation loop ===

        while counter < n_steps_parent:
            if counter % 10000 == 0:
                print('Parent Step:', counter)

            # Integrate parent
            with timer.section("integrator_step_parent"):
                if str(self.thermostat.thermostat_type) == 'PILE_L':
                    integrator_parent.step_PILE_L(self.system, self.thermostat)
                elif str(self.thermostat.thermostat_type) == 'PILE_G':
                    integrator_parent.step_PILE_G(self.system, self.thermostat)
                elif str(self.thermostat.thermostat_type) == 'Andersen':
                    integrator_parent.step_Andersen(self.system, self.thermostat)
                else:
                    integrator_parent.step_NVE(self.system)

            time_counter += self.timestep
            counter += 1

            self._ensure_finite_state(
                counter,
                time_counter,
                "NEQ parent dynamics",
            )

            with timer.section("rec.write"):
                if self.stride_out > 0 and counter % self.stride_out == 0 and rec_parent_energies is not None:
                    rec_parent_energies.write(
                        self.system,
                        counter,
                        time_counter,
                        record_energies=True,
                        record_estimators=True,
                    )
                if rec_parent_trajectories is not None:
                    rec_parent_trajectories.write(
                        self.system,
                        counter,
                        time_counter,
                        record_bead_positions=record_bead_positions_parent,
                        record_centroid_position=False,
                    )
                if rec_parent_velocities is not None:
                    rec_parent_velocities.write(
                        self.system,
                        counter,
                        time_counter,
                        record_bead_velocities=record_bead_positions_parent,
                        record_centroid_velocity=False,
                    )
            #print("parent forces before", self.system.forces)
            # At sampling step: fork off child trajectory (after sample_steps parent steps)
            if sample_steps > 0 and (counter % sample_steps == 0):
                child_system = copy.deepcopy(self.system)

                #print(self.system.potential_energies)
                #print("before HPI",child_system.tot, child_system.pot, child_system.kin,child_system.positions)
                E_i = child_system.tot

                if record_endpoints:
                    rec_initial.write(child_system, counter, time_counter,
                                    record_energies=False, record_estimators=False,
                                    record_bead_positions = record_bead_positions_child,
                                    record_bead_velocities = record_bead_positions_child,
                                    record_centroid_position=True,
                                    record_centroid_velocity=True)

                # Child setup
                thermostat_child = copy.deepcopy(self.thermostat)
                thermostat_child.thermostat_type = thermostat_type_child
                thermostat_child.tau = tau_child
                child_index = counter // sample_steps - 1

                simulation_child = Simulation(
                    child_system, self.temperature, self.timestep, self.force_field,
                    thermostat_child, mode='spimd',
                    prefix=self.prefix + f'_child_{child_index}',
                    stride_out=stride_out_child,
                    stride_pos=stride_pos_child,
                    stride_vel=stride_vel_child
                )


                with timer.section("child.run"):

                    simulation_child.run_SPIMD(n_steps_child,record_bead_positions=record_bead_positions_child)

                E_e = child_system.tot

                work = E_e - E_i

                rec_work.write(child_system, counter, time_counter,
                            record_work=True, work=work)

            timer.step_done()


    def run_PIMC(
        self,
        n_samples,
        step_size=0.05,
        record_bead_positions=None,
        *,
        out_npz=None,
        out_text=True,
        gen_velocities=False,
    ):
        """
        Monte Carlo sampling of ring-polymer configurations using single-bead, single-coordinate Gaussian moves.
        Records only accepted configurations.
        """

        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature

        beta_n     = temperature_to_beta(temperature)
        rec = Recorder(system, self.prefix) if out_text else None
        if record_bead_positions == "All" or record_bead_positions == "all":
            record_bead_positions = list(range(self.n_beads))
        accepted = 0
        attempted = 0
        elapsed_time = 0.0
        qs, ps = [], []

        while accepted < n_samples:
            attempted += 1
            q_old = system.get_positions().copy()

            # Propose: single bead, single coordinate
            bead_idx  = rng.randint(nbeads)
            coord_idx = rng.randint(natoms * ndim)
            q_trial   = q_old.copy()
            q_trial[bead_idx, coord_idx] += rng.normal(loc=0.0, scale=step_size)

            # External potential for trial config (per-bead array); your API returns (forces, potentials)
            _, pot_trial = self.force_field.eq_force_function(q_trial, nbeads, natoms, ndim)

            # Ring-polymer spring energy (internal) for old and new
            V_int_old = Internal_Potential(q_old,   temperature, masses, nbeads, natoms, ndim)
            V_int_new = Internal_Potential(q_trial, temperature, masses, nbeads, natoms, ndim)

            # Total configurational "U" (external + springs), momenta integrated out
            _, pot_old = self.force_field.eq_force_function(q_old, nbeads, natoms, ndim)
            U_old = V_int_old + np.sum(pot_old)  # ensure external potential matches q_old
            U_new = V_int_new + np.sum(pot_trial)

            dS = beta_n * (U_new - U_old)

            # Stable Metropolis accept
            if dS <= 0.0 or rng.random() < np.exp(-dS):
                # Accept
                system.set_positions(q_trial)
                system.set_potential_energies(pot_trial)

                # Update derived energetics/estimators (momenta are not used in MC)
                dummy_v = np.zeros_like(system.positions)
                dummy_f = np.zeros_like(system.positions)
                qc = np.mean(system.positions, axis=0, keepdims=True)
                vc = np.zeros_like(qc)

                pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                    system.positions, dummy_v, qc, dummy_f, pot_trial,
                    masses, temperature, nbeads, natoms, ndim
                )

                update_system(
                    system, system.positions, dummy_v, qc, vc, dummy_f, system_T,
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot,pot_trial
                )

                accepted += 1

                # Note: elapsed_time is incremented by 1.0 per trial move and does not represent physical time units.
                elapsed_time += 1.0
                self._ensure_finite_state(accepted, elapsed_time, "PIMC sampling")

                if gen_velocities:
                    sampled_velocities = thermal(
                        nbeads, natoms, ndim, temperature, masses, rng=rng
                    )
                    system.set_velocities(sampled_velocities)
                    ps.append(sampled_velocities.copy())
                else:
                    ps.append(np.zeros_like(system.positions))
                qs.append(system.positions.copy())

                if rec is not None:
                    rec.write(self.system, accepted, elapsed_time, record_energies=True,
                            record_estimators=True,
                            record_bead_positions=record_bead_positions,
                            record_bead_velocities=record_bead_positions if gen_velocities else None,
                            record_centroid_position=True,
                            record_centroid_velocity=gen_velocities)

        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted}")
        print(f"Acceptance rate: {accepted / attempted:.4f}")

        self.last_acceptance_stats = {
            "mode": "pimc_single",
            "accepted": accepted,
            "attempted": attempted,
            "acceptance": accepted / attempted if attempted else 0.0,
            "step_size": step_size,
        }
        q_array = np.asarray(qs)
        p_array = np.asarray(ps)
        if out_npz:
            os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
            np.savez(
                out_npz,
                q=q_array,
                p=p_array,
                natoms=natoms,
                ndim=ndim,
                nbeads=nbeads,
                meta="candidate pool",
            )
        return q_array, p_array


    def run_PIMC_adaptive(self,
                n_samples,
                step_size=0.05,
                record_bead_positions=None,
                *,
                target_accept=0.40,      # desired acceptance rate
                adapt_interval=200,       # adapt every N attempted moves
                adapt_gain=0.25,          # sensitivity of adaptation (higher = faster)
                min_step=1e-6,            # lower bound on step size (std of proposal)
                max_step=1.0,             # upper bound on step size
                out_npz=None,
                out_text=True,
                gen_velocities=False,
                ):
        """
        Monte Carlo sampling of ring-polymer configurations using single-bead, single-coordinate
        Gaussian moves. Records only accepted configurations.

        Adaptive step size:
        - Every `adapt_interval` *attempted* moves, compute acceptance in that window and
            scale `step_size` by: step *= exp(adapt_gain * (acc_rate_window - target_accept)).
        - Keeps step_size in [min_step, max_step].

        Notes:
        - 'elapsed_time' here counts *accepted* moves (pseudo-time for logging).
        - If you need true per-attempt 'time', move its increment outside the accept block.
        """

        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature

        beta_n = temperature_to_beta(temperature)

        rec = Recorder(system, self.prefix) if out_text else None

        # Normalize "All"/"all" for bead recording
        if isinstance(record_bead_positions, str) and record_bead_positions.lower() == "all":
            record_bead_positions = list(range(self.n_beads))

        accepted = 0
        attempted = 0
        elapsed_time = 0.0

        # Window counters for adaptation
        win_attempted = 0
        win_accepted = 0
        qs, ps = [], []

        while accepted < n_samples:
            attempted += 1
            win_attempted += 1

            q_old = system.get_positions().copy()

            # Propose: single bead, single coordinate (Gaussian step with adaptive std = step_size)
            bead_idx  = rng.randint(nbeads)
            coord_idx = rng.randint(natoms * ndim)
            q_trial   = q_old.copy()
            q_trial[bead_idx, coord_idx] += rng.normal(loc=0.0, scale=step_size)

            # External potentials: old and new
            _, pot_old   = self.force_field.eq_force_function(q_old,   nbeads, natoms, ndim)
            _, pot_trial = self.force_field.eq_force_function(q_trial, nbeads, natoms, ndim)

            # Ring-polymer internal (spring) energies
            V_int_old = Internal_Potential(q_old,   temperature, masses, nbeads, natoms, ndim)
            V_int_new = Internal_Potential(q_trial, temperature, masses, nbeads, natoms, ndim)

            # Total configurational action difference (momenta integrated out)
            U_old = V_int_old + np.sum(pot_old)
            U_new = V_int_new + np.sum(pot_trial)
            dS = beta_n * (U_new - U_old)

            # Metropolis accept/reject
            if dS <= 0.0 or rng.random() < np.exp(-dS):
                # Accept
                system.set_positions(q_trial)
                system.set_potential_energies(pot_trial)

                # Update derived energetics/estimators for logging (MC has no velocities)
                dummy_v = np.zeros_like(system.positions)
                dummy_f = np.zeros_like(system.positions)
                qc = np.mean(system.positions, axis=0, keepdims=True)
                vc = np.zeros_like(qc)

                pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                    system.positions, dummy_v, qc, dummy_f, pot_trial,
                    masses, temperature, nbeads, natoms, ndim
                )

                update_system(
                    system, system.positions, dummy_v, qc, vc, dummy_f, system_T,
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot,pot_trial
                )

                accepted += 1
                win_accepted += 1

                # Pseudo time: count accepted states (kept for compatibility with your logs)
                elapsed_time += 1.0
                self._ensure_finite_state(accepted, elapsed_time, "Adaptive PIMC sampling")

                if gen_velocities:
                    sampled_velocities = thermal(
                        nbeads, natoms, ndim, temperature, masses, rng=rng
                    )
                    system.set_velocities(sampled_velocities)
                    ps.append(sampled_velocities.copy())
                else:
                    ps.append(np.zeros_like(system.positions))
                qs.append(system.positions.copy())

                if rec is not None:
                    rec.write(
                        self.system, accepted, elapsed_time,
                        record_energies=True,
                        record_estimators=True,
                        record_bead_positions=record_bead_positions,
                        record_bead_velocities=record_bead_positions if gen_velocities else None,
                        record_centroid_position=True,
                        record_centroid_velocity=gen_velocities
                    )

            # --- Adaptive step-size update (based on *attempts* in the current window) ---
            if win_attempted >= adapt_interval:
                acc_rate_win = win_accepted / max(1, win_attempted)

                # Exponential controller: step *= exp(gain*(acc - target))
                #  - If acc > target -> increase step; if acc < target -> decrease step.
                scale = np.exp(adapt_gain * (acc_rate_win - target_accept))
                step_size = float(np.clip(step_size * scale, min_step, max_step))

                # Reset window
                win_attempted = 0
                win_accepted = 0

                # Optional: print a lightweight progress line
                #print(f"[adapt] attempts={attempted}, accepted={accepted}, "
                #    f"acc_total={accepted/max(1,attempted):.3f}, "
                #    f"acc_win={acc_rate_win:.3f}, step_size={step_size:.6g}")

        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted}")
        print(f"Final acceptance rate: {accepted / attempted:.4f}")
        print(f"Final step_size: {step_size:.6g}")

        self.last_acceptance_stats = {
            "mode": "pimc_single",
            "accepted": accepted,
            "attempted": attempted,
            "acceptance": accepted / attempted if attempted else 0.0,
            "step_size": step_size,
        }
        q_array = np.asarray(qs)
        p_array = np.asarray(ps)
        if out_npz:
            os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
            np.savez(
                out_npz,
                q=q_array,
                p=p_array,
                natoms=natoms,
                ndim=ndim,
                nbeads=nbeads,
                meta="candidate pool",
            )
        return q_array, p_array

    def run_PIMC_enhanced(
        self,
        n_samples,
        step_size_single=0.05,
        step_size_disp=0.05,
        record_bead_positions=None,
        *,
        # move mix (any nonnegative weights; normalized internally)
        p_single=0.5,
        p_displacement=0.25,
        p_bisection=0.25,
        # adapt targets (per move type)
        target_accept_single=0.40,
        target_accept_disp=0.50,
        target_accept_bis=0.35,
        adapt_interval=200,
        adapt_gain=0.25,
        min_step=1e-6,
        max_step=1.0,
        # bisection specifics
        max_bis_level=2,           # 0=midpoint only; 1=+quarter points; etc.
        slice_length=None,         # if None -> 2**(max_bis_level+1); must be power of 2
        enable_bisection=True,
        out_npz=None,
        out_text=True,
        gen_velocities=False,
    ):
        """
        PIMC with Ceperley-style moves:
        • single-bead, single-coordinate Gaussian (symmetric)
        • whole-path displacement of one atom (symmetric; preserves springs)
        • multiscale bisection on a dyadic slice for one atom/coord
            (proposal from free-spring conditional; MH acceptance uses ΔV_ext ONLY)

        Records only accepted configurations. Per-move adaptive step/level control.
        Diagnostics printed at the end.
        """

        # ---------- normalize move weights ----------
        if not enable_bisection:
            p_bisection = 0.0
        total_p = p_single + p_displacement + p_bisection
        if total_p <= 0:
            raise ValueError("At least one move probability must be > 0.")
        p_single       /= total_p
        p_displacement /= total_p
        p_bisection    /= total_p

        def choose_move(rng):
            u = rng.random()
            if u < p_single:
                return "single"
            elif u < p_single + p_displacement:
                return "disp"
            else:
                return "bisection"

        # ---------- helpers ----------
        def spring_omega_n(beta_n, nbeads):
            # If ħ != 1 in your units, change to: nbeads / (beta_n * ħ)
            return nbeads / beta_n

        def normalize_record_spec(x):
            if isinstance(x, str) and x.lower() == "all":
                return list(range(self.n_beads))
            return x

        # Conditional midpoint (free-spring bridge) for one coordinate:
        # q_mid | (a,b) ~ N( (a+b)/2, 1/(2 beta m ω_n^2) )
        def sample_midpoint(rng, a, b, beta_n, mass, omega_n):
            mean = 0.5 * (a + b)
            var  = 1.0 / (2.0 * beta_n * mass * (omega_n ** 2))
            return mean + rng.normal(0.0, np.sqrt(var))

        # Recursive dyadic bisection on [i0, i1] (cyclic indices) for one atom/coord
        # Proposal uses ONLY the free-spring conditional; endpoints fixed.
        def propose_bisection_on_slice(q_old, atom_idx, coord_off, i0, i1, levels, beta_n, masses, omega_n, rng):
            q_new = q_old.copy()
            nbeads = q_old.shape[0]
            work = [(i0, i1, 0)]

            def mid_index(i, j):
                L = (j - i) % nbeads
                return (i + L // 2) % nbeads

            while work:
                L, R, lev = work.pop()
                if lev > levels:
                    continue
                M = mid_index(L, R)
                a = q_new[L, coord_off]
                b = q_new[R, coord_off]
                m = masses[atom_idx]
                q_new[M, coord_off] = sample_midpoint(rng, a, b, beta_n, m, omega_n)
                if lev < levels:
                    work.append((L, M, lev + 1))
                    work.append((M, R, lev + 1))
            return q_new

        # Whole-path displacement: translate all beads of one atom by same Δ in R^ndim
        def propose_displacement(q_old, atom_idx, nbeads, ndim, rng, step):
            q_new = q_old.copy()
            delta = rng.normal(0.0, step, size=ndim)
            base = atom_idx * ndim
            for b in range(nbeads):
                q_new[b, base:base + ndim] += delta
            return q_new

        # Single-bead, single-coordinate Gaussian
        def propose_single(q_old, nbeads, natoms, ndim, rng, step):
            q_new = q_old.copy()
            bead_idx  = rng.randint(nbeads)
            coord_idx = rng.randint(natoms * ndim)
            q_new[bead_idx, coord_idx] += rng.normal(0.0, step)
            return q_new

        # Energy helper: returns total U, external pot vector, and internal spring energy
        def total_U(q):
            _, pot = self.force_field.eq_force_function(q, nbeads, natoms, ndim)  # external per-bead
            V_int  = Internal_Potential(q, temperature, masses, nbeads, natoms, ndim)
            return V_int + np.sum(pot), pot, V_int

        # Per-move adaptive controller
        class Adapt:
            def __init__(self, step, target):
                self.step = float(step)
                self.target = float(target)
                self.win_att = 0
                self.win_acc = 0
            def note(self, accepted_flag):
                self.win_att += 1
                if accepted_flag:
                    self.win_acc += 1
            def maybe_update(self, label):
                if self.win_att >= adapt_interval:
                    acc = self.win_acc / max(1, self.win_att)
                    scale = np.exp(adapt_gain * (acc - self.target))
                    self.step = float(np.clip(self.step * scale, min_step, max_step))
                    self.win_att = 0
                    self.win_acc = 0
                    #print(f"[adapt:{label}] acc_win={acc:.3f} target={self.target:.3f} step={self.step:.6g}")

        # ---------- setup ----------
        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature

        beta_n = temperature_to_beta(temperature)
        omega_n = 1/beta_n

        record_bead_positions = normalize_record_spec(record_bead_positions)

        # bisection slice length
        if p_bisection > 0:
            if nbeads < 2:
                raise ValueError("Bisection moves require at least two beads.")
            if slice_length is None:
                largest_level = int(np.floor(np.log2(nbeads))) - 1
                bis_levels = min(max_bis_level, largest_level)
                bis_len = 2 ** (bis_levels + 1)
            else:
                bis_len = int(slice_length)
            if bis_len <= 1 or (bis_len & (bis_len - 1)) != 0:
                raise ValueError("slice_length must be a power of two ≥ 2")
            if bis_len > nbeads:
                raise ValueError("slice_length must be ≤ n_beads")
            bis_levels = min(max_bis_level, int(np.log2(bis_len)) - 1)
        else:
            bis_len = 1
            bis_levels = 0

        rec = Recorder(system, self.prefix) if out_text else None

        # adaptation trackers per move
        A_single = Adapt(step_size_single, target_accept_single)
        A_disp   = Adapt(step_size_disp,   target_accept_disp)
        bis_att = bis_acc = 0

        # diagnostics counters
        att_single = acc_single = 0
        att_disp   = acc_disp   = 0
        att_bis    = acc_bis    = 0

        accepted = 0
        attempted_total = 0
        elapsed_time = 0.0
        qs, ps = [], []

        q_curr = system.get_positions().copy()
        U_curr, pot_curr, Vint_curr = total_U(q_curr)

        while accepted < n_samples:
            move_kind = choose_move(rng)
            attempted_total += 1

            # --- propose ---
            if move_kind == "single":
                att_single += 1
                q_trial = propose_single(q_curr, nbeads, natoms, ndim, rng, A_single.step)
            elif move_kind == "disp":
                att_disp += 1
                atom_idx = rng.randint(natoms)
                q_trial = propose_displacement(q_curr, atom_idx, nbeads, ndim, rng, A_disp.step)
            else:  # bisection
                att_bis += 1
                atom_idx = rng.randint(natoms)
                coord_in_atom = rng.randint(ndim)
                coord_off = atom_idx * ndim + coord_in_atom
                i0 = rng.randint(nbeads)
                i1 = (i0 + bis_len) % nbeads
                q_trial = propose_bisection_on_slice(
                    q_curr, atom_idx, coord_off, i0, i1, bis_levels,
                    beta_n, masses, omega_n, rng
                )

            # --- energies for trial ---
            U_new, pot_new, Vint_new = total_U(q_trial)

            # --- Metropolis / MH acceptance ---
            if move_kind == "bisection":
                # Using free-spring proposal => MH ratio cancels spring action.
                # Accept on external potential change ONLY:
                dS = beta_n * (np.sum(pot_new) - np.sum(pot_curr))
            else:
                # Symmetric proposals -> standard Metropolis on total action
                dS = beta_n * (U_new - U_curr)

            accept_flag = (dS <= 0.0) or (rng.random() < np.exp(-dS))

            # adapt windows + per-move counts
            if move_kind == "single":
                A_single.note(accept_flag); A_single.maybe_update("single")
                if accept_flag: acc_single += 1
            elif move_kind == "disp":
                A_disp.note(accept_flag);   A_disp.maybe_update("disp")
                if accept_flag: acc_disp += 1
            else:
                bis_att += 1
                if accept_flag: bis_acc += 1
                if bis_att >= adapt_interval:
                    acc = bis_acc / max(1, bis_att)
                    # gently nudge levels
                    if acc < target_accept_bis * 0.8 and bis_levels > 0:
                        bis_levels -= 1
                    elif acc > target_accept_bis * 1.2 and (2 ** (bis_levels + 2)) <= nbeads:
                        bis_levels += 1
                    #print(f"[adapt:bisection] acc_win={acc:.3f} target={target_accept_bis:.3f} levels={bis_levels}")
                    bis_att = bis_acc = 0
                if accept_flag: acc_bis += 1

            if accept_flag:
                # commit current state
                q_curr, U_curr, pot_curr, Vint_curr = q_trial, U_new, pot_new, Vint_new
                system.set_positions(q_curr)
                system.set_potential_energies(pot_curr)

                # MC bookkeeping (no velocities)
                dummy_v = np.zeros_like(system.positions)
                dummy_f = np.zeros_like(system.positions)
                qc = np.mean(system.positions, axis=0, keepdims=True)
                vc = np.zeros_like(qc)

                pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = compute_energetics(
                    system.positions, dummy_v, qc, dummy_f, pot_curr,
                    masses, temperature, nbeads, natoms, ndim
                )
                update_system(
                    system, system.positions, dummy_v, qc, vc, dummy_f, system_T,
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot, pot_curr
                )

                accepted += 1
                elapsed_time += 1.0
                self._ensure_finite_state(accepted, elapsed_time, "Enhanced PIMC sampling")

                if gen_velocities:
                    sampled_velocities = thermal(
                        nbeads, natoms, ndim, temperature, masses, rng=rng
                    )
                    system.set_velocities(sampled_velocities)
                    ps.append(sampled_velocities.copy())
                else:
                    ps.append(np.zeros_like(system.positions))
                qs.append(system.positions.copy())

                if rec is not None:
                    rec.write(
                        self.system, accepted, elapsed_time,
                        record_energies=True,
                        record_estimators=True,
                        record_bead_positions=record_bead_positions,
                        record_bead_velocities=record_bead_positions if gen_velocities else None,
                        record_centroid_position=True,
                        record_centroid_velocity=gen_velocities
                    )

        # ---------- diagnostics ----------
        def safe_rate(a, b): return (a / b) if b > 0 else 0.0

        acc_total = accepted / max(1, attempted_total)
        acc_r_single = safe_rate(acc_single, att_single)
        acc_r_disp   = safe_rate(acc_disp,   att_disp)
        acc_r_bis    = safe_rate(acc_bis,    att_bis)

        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted_total}")
        print(f"Final acceptance rate (overall): {acc_total:.4f}")
        print(f"Attempts   – single: {att_single}, disp: {att_disp}, bisection: {att_bis}")
        print(f"Accepted   – single: {acc_single}, disp: {acc_disp}, bisection: {acc_bis}")
        print(f"Acc. rates – single: {acc_r_single:.4f}, disp: {acc_r_disp:.4f}, bisection: {acc_r_bis:.4f}")
        print(f"Final steps – single: {A_single.step:.6g}, disp: {A_disp.step:.6g}, bis_levels: {bis_levels}")

        self.last_acceptance_stats = {
            "mode": "pimc_enhanced",
            "accepted": accepted,
            "attempted": attempted_total,
            "acceptance": acc_total,
            "rate_single": acc_r_single,
            "rate_disp": acc_r_disp,
            "rate_bisection": acc_r_bis,
            "step_single": A_single.step,
            "step_disp": A_disp.step,
            "bis_levels": bis_levels,
        }
        q_array = np.asarray(qs)
        p_array = np.asarray(ps)
        if out_npz:
            os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
            np.savez(
                out_npz,
                q=q_array,
                p=p_array,
                natoms=natoms,
                ndim=ndim,
                nbeads=nbeads,
                meta="candidate pool",
            )
        return q_array, p_array
