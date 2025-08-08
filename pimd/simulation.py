#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:18:17 2022

@author: castrojo
"""

import numpy as np
from pimd.integrator import Integrator
from pimd.utils import *
from pimd.file_recorder import Recorder
#from velocity_read import *

class Simulation(object):
    #Constructor
    def __init__(self,system,temperature,timestep,force_field,thermostat=None,mode=None,prefix='simulation'):
        self.initializer(system,temperature,timestep,force_field,thermostat,mode,prefix)

    def initializer(self,system,temperature,timestep,force_field,thermostat=None,mode=None,prefix='simulation'):
        self.force_field = force_field
        self.timestep = timestep
        self.temperature = temperature    
        self.thermostat = thermostat
        print('Thermostat: ', thermostat)
        self.mode = mode
        self.prefix = prefix
        self.system = system
        self.n_beads = system.n_beads
        n_atoms = system.n_atoms
        n_dim = system.n_dim
        init_q = system.positions
        init_v = system.velocities
        init_qc = system.q_centroid
        init_vc = system.v_centroid
        masses = system.masses
        init_f, init_potential_energies = self.force_field.eq_force_function(init_q,self.n_beads,n_atoms,n_dim)
        init_system_T = instantaneous_temperature(init_v,masses,self.n_beads,n_atoms,n_dim)
        #init_system_T = beta_to_temperature(init_system_T)
        init_kin = classical_kinetic_energy(init_v, masses, n_atoms, n_dim)/self.n_beads
        init_internal_pot = Internal_Potential(init_q,self.temperature,masses,self.n_beads,n_atoms,n_dim)/self.n_beads
        init_external_pot = classical_potential_energy(init_potential_energies)/self.n_beads 
        init_pot = init_external_pot + init_internal_pot
        init_E_tot, init_E_kin, init_E_pot = E_Estimators(temperature, init_f, init_q, init_qc, init_potential_energies,self.n_beads, n_atoms, n_dim)
        self.system.set_forces(init_f)
        self.system.set_pot(init_pot)
        self.system.set_kin(init_kin)
        self.system.set_tot(init_pot+init_kin)
        self.system.set_E_pot(init_E_pot)
        self.system.set_E_kin(init_E_kin)
        self.system.set_E_tot(init_E_tot)
        self.system.set_potential_energies(init_potential_energies)
        self.system.set_Temperature(init_system_T)
        print(f"Using thermostat type: {self.thermostat.thermostat_type} with tau = {self.thermostat.tau}")


    def run_PIMD(self,nsteps):
        
        counter = 0
        time_counter = 0
        
        #Call the integrator
        integrator = Integrator(self.temperature,self.timestep,self.force_field,self.mode)
        
        #Call the file recorder        
        rec = Recorder(self.system,self.prefix)
        rec.write(self.system,counter,time_counter,record_energies=True,
                  record_estimators=True,
                  record_bead_positions=[0,1],
                  record_bead_velocities=[0,1],
                  record_centroid_position=True,
                  record_centroid_velocity=True)

        while counter < nsteps:
            
            #if counter%100000 == 0:
            #    print('Step: ',counter)
            #print('velocities before: ',self.system.velocities)
            if str(self.thermostat.thermostat_type) == 'PILE_L':
                integrator.step_PILE_L(self.system,self.thermostat)
            elif str(self.thermostat.thermostat_type) == 'PILE_G':
                integrator.step_PILE_G(self.system,self.thermostat)
            elif str(self.thermostat.thermostat_type) == 'Andersen':
                integrator.step_Andersen(self.system,self.thermostat)                    
            else:
                integrator.step_NVE(self.system)
            #print('velocities after: ',self.system.velocities)
            counter += 1 

            #lambda_t = counter/nsteps
            #print(lambda_t)
            #print('input',self.system.get_velocities())
            #if str(self.thermostat.thermostat_type) == 'PILE_L':
            #    integrator.step_NVT_L(self.system,self.thermostat,lambda_t)
                
            #elif str(self.thermostat.thermostat_type) == 'PILE_G':
            #    integrator.step_NVT_G(self.system,self.thermostat,lambda_t)
            
            #elif str(self.thermostat.thermostat_type) == 'Andersen':
            #    integrator.step_Andersen(self.system,self.thermostat,lambda_t)
                
            #else:
            #    integrator.step_NVE(self.system,lambda_t)
                
            #print('ouput',self.system.get_velocities())                    
            time_counter += self.timestep
            #print(self.system.qc)
            #print(self.system.Temperature)
            rec.write(self.system,counter,time_counter,record_energies=True,
                  record_estimators=True,
                  record_bead_positions=[0,1],
                  record_bead_velocities=[0,1],
                  record_centroid_position=True,
                  record_centroid_velocity=True)
                    
    def run_PIMC(self, n_samples, step_size=0.05):
        """
        Monte Carlo sampling of ring-polymer configurations using single-particle moves.
        Records only accepted configurations.

        Parameters
        ----------
        n_samples : int
            Number of accepted configurations to collect
        step_size : float
            Standard deviation of the Gaussian move proposal (in a.u.)
        """

        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature
        beta_n = temperature_to_beta(temperature)

        # Recorder
        rec = Recorder(system, self.prefix)

        # Counters
        accepted = 0
        attempted = 0
        elapsed_time = 0.0

        while accepted < n_samples:
            attempted += 1
            q_old = system.get_positions().copy()

            # Trial move
            bead_idx = rng.randint(nbeads)
            coord_idx = rng.randint(natoms * ndim)
            q_trial = q_old.copy()
            q_trial[bead_idx, coord_idx] += rng.normal(scale=step_size)

            # Compute trial potential energy
            _, pot_trial = self.force_field.eq_force_function(q_trial, nbeads, natoms, ndim)
            V_int_old = Internal_Potential(q_old, temperature, masses, nbeads, natoms, ndim)
            V_int_new = Internal_Potential(q_trial, temperature, masses, nbeads, natoms, ndim)

            U_old = V_int_old + np.sum(system.potential_energies)
            U_new = V_int_new + np.sum(pot_trial)

            dS = beta_n * (U_new - U_old)

            if rng.rand() < np.exp(-dS):
                # Accept
                system.set_positions(q_trial)
                system.set_potential_energies(pot_trial)

                # Update system energetics (velocities and forces are zero in MC)
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
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot
                )

                # Record this accepted configuration
                rec.write(system, accepted, elapsed_time,
                        record_energies=True,
                        record_estimators=True,
                        record_bead_positions=[0, 1],
                        record_centroid_position=True)

                accepted += 1

            elapsed_time += 1.0

        # Final acceptance report
        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted}")
        print(f"Acceptance rate: {accepted / attempted:.4f}")

    def run_PIMC_adaptive(self, n_samples, initial_step=0.05, target_acceptance=0.4, adapt_interval=1000000000):
        """
        Adaptive Path Integral Monte Carlo using single-particle moves.

        Parameters
        ----------
        n_samples : int
            Number of accepted configurations to collect
        initial_step : float
            Initial width of the Gaussian proposal (in a.u.)
        target_acceptance : float
            Target acceptance rate (default = 0.4)
        adapt_interval : int
            Frequency of step size adaptation (in attempted moves)
        """

        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature
        beta_n = temperature_to_beta(temperature)

        # Recorder
        rec = Recorder(system, self.prefix)

        # Initialize counters
        accepted = 0
        attempted = 0
        total_accepted = 0
        total_attempted = 0
        step_size = initial_step
        elapsed_time = 0.0

        while accepted < n_samples:
            attempted += 1
            total_attempted += 1

            q_old = system.get_positions().copy()

            # Trial move
            bead_idx = rng.randint(nbeads)
            coord_idx = rng.randint(natoms * ndim)
            q_trial = q_old.copy()
            q_trial[bead_idx, coord_idx] += rng.normal(scale=step_size)

            # Energy calculations
            _, pot_trial = self.force_field.eq_force_function(q_trial, nbeads, natoms, ndim)
            V_int_old = Internal_Potential(q_old, temperature, masses, nbeads, natoms, ndim)
            V_int_new = Internal_Potential(q_trial, temperature, masses, nbeads, natoms, ndim)

            U_old = V_int_old + np.sum(system.potential_energies)
            U_new = V_int_new + np.sum(pot_trial)

            dS = beta_n * (U_new - U_old)

            if rng.rand() < np.exp(-dS):
                # Accept
                system.set_positions(q_trial)
                system.set_potential_energies(pot_trial)

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
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot
                )

                rec.write(system, accepted, elapsed_time,
                        record_energies=True,
                        record_estimators=True,
                        record_bead_positions=[0, 1],
                        record_centroid_position=True)

                accepted += 1
                total_accepted += 1

            elapsed_time += 1.0

            # Step size adaptation
            if attempted % adapt_interval == 0:
                acceptance_rate = total_accepted / total_attempted
                adjustment = acceptance_rate / target_acceptance
                step_size *= adjustment
                step_size = min(max(step_size, 1e-4), 0.5)  # clip within bounds
                total_accepted = 0
                total_attempted = 0

        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted}")
        print(f"Final step size: {step_size:.5f}")
        print(f"Final acceptance rate: {accepted / attempted:.4f}")


    def run_PIMC_adaptive_advanced(self, n_samples, 
                        initial_step_single=0.05, 
                        initial_step_disp=0.1, 
                        target_acceptance=0.4, 
                        adapt_interval=1000, 
                        bisection_slices=3):
        """
        Adaptive PIMC with single-bead, displacement, and bisection moves.

        Parameters
        ----------
        n_samples : int
            Number of accepted configurations to collect
        initial_step_single : float
            Initial Gaussian width for single-bead moves
        initial_step_disp : float
            Initial Gaussian width for displacement moves
        target_acceptance : float
            Desired acceptance rate
        adapt_interval : int
            How often to adapt step sizes
        bisection_slices : int
            Number of beads to resample in a bisection move
        """

        system = self.system
        rng = system.rng

        nbeads = system.n_beads
        ndim = system.n_dim
        natoms = system.n_atoms
        masses = system.masses
        temperature = self.temperature
        beta_n = temperature_to_beta(temperature)

        rec = Recorder(system, self.prefix)

        accepted = 0
        attempted = 0
        elapsed_time = 0.0

        # Step sizes for each move type
        step_single = initial_step_single
        step_disp = initial_step_disp

        # Counters per move type
        acc = {"single": 0, "disp": 0, "bisection": 0}
        att = {"single": 0, "disp": 0, "bisection": 0}

        while accepted < n_samples:
            # Dynamically choose allowed move types
            move_pool = ["single", "disp"]
            if nbeads > bisection_slices:
                move_pool.append("bisection")

            move_type = rng.choice(move_pool)
            q_old = system.get_positions().copy()
            q_trial = q_old.copy()

            att[move_type] += 1
            attempted += 1

            if move_type == "single":
                bead_idx = rng.randint(nbeads)
                coord_idx = rng.randint(natoms * ndim)
                q_trial[bead_idx, coord_idx] += rng.normal(scale=step_single)

            elif move_type == "disp":
                atom_idx = rng.randint(natoms)
                shift = rng.normal(scale=step_disp, size=(ndim,))
                for b in range(nbeads):
                    q_trial[b, atom_idx*ndim:(atom_idx+1)*ndim] += shift

            elif move_type == "bisection":
                atom_idx = rng.randint(natoms)
                m = rng.randint(nbeads - bisection_slices)
                l = bisection_slices

                # endpoints
                q_start = q_old[m, atom_idx*ndim:(atom_idx+1)*ndim]
                q_end = q_old[m+l, atom_idx*ndim:(atom_idx+1)*ndim]

                # Approximate sigma^2 using free-particle propagator
                sigma2 = 1.0 / (masses[atom_idx] * temperature * nbeads)

                # Sample intermediate points linearly + Gaussian noise
                for i in range(1, l):
                    t = i / l
                    mu = (1 - t) * q_start + t * q_end
                    q_trial[m+i, atom_idx*ndim:(atom_idx+1)*ndim] = mu + rng.normal(scale=np.sqrt(sigma2), size=(ndim,))

            # Compute energy difference
            _, pot_trial = self.force_field.eq_force_function(q_trial, nbeads, natoms, ndim)
            V_int_old = Internal_Potential(q_old, temperature, masses, nbeads, natoms, ndim)
            V_int_new = Internal_Potential(q_trial, temperature, masses, nbeads, natoms, ndim)
            U_old = V_int_old + np.sum(system.potential_energies)
            U_new = V_int_new + np.sum(pot_trial)
            dS = beta_n * (U_new - U_old)

            if rng.rand() < np.exp(-dS):
                # Accept
                system.set_positions(q_trial)
                system.set_potential_energies(pot_trial)

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
                    pot, kin, etot, E_pot, E_kin, E_tot, internal_pot
                )

                rec.write(system, accepted, elapsed_time,
                        record_energies=True,
                        record_estimators=True,
                        record_bead_positions=[0, 1],
                        record_centroid_position=True)

                accepted += 1
                acc[move_type] += 1

            elapsed_time += 1.0

            # Adapt step sizes
            if attempted % adapt_interval == 0:
                # Adapt single-bead
                if att["single"]:
                    rate = acc["single"] / att["single"]
                    step_single *= rate / target_acceptance
                    step_single = np.clip(step_single, 1e-4, 0.5)

                # Adapt displacement
                if att["disp"]:
                    rate = acc["disp"] / att["disp"]
                    step_disp *= rate / target_acceptance
                    step_disp = np.clip(step_disp, 1e-4, 1.0)

                # Reset counters
                acc = dict.fromkeys(acc, 0)
                att = dict.fromkeys(att, 0)

        print(f"Accepted samples: {accepted}")
        print(f"Total trial moves: {attempted}")
        print(f"Final step_single: {step_single:.5f}")
        print(f"Final step_disp: {step_disp:.5f}")
        print(f"Final acceptance rate: {accepted / attempted:.4f}")

