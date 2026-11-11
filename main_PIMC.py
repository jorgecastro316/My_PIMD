#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:44:11 2022

@author: castrojo
"""

from pimd.system import System
from pimd.force_field import Force_Field
from pimd.simulation import Simulation
from pimd.thermostat import Thermostat
from pimd.utils import *
import sys
import time


# === Read input ===
input_file = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
inp = read_input_file(input_file)

# === Parse input ===
geometry_file = parse_file_field(inp["geometry_file"])
velocity_file = parse_file_field(inp.get("velocity_file", None))

#Initialze random number generator
seed = inp.get("seed", 420)
rng = np.random.mtrand.RandomState(seed)

#Set the system
n_beads = inp["n_beads"]
system_temperature = inp["temperature"]
print(f"System temperature: {system_temperature} K")
simulation_temperature = system_temperature * n_beads  # PIMD temperature
print(f"Simulation temperature: {simulation_temperature} K")
system = System(geometry_file, n_beads, rng, velocity_file, simulation_temperature)
print(f"System initialized with {system.n_atoms} atoms, {system.n_beads} beads, and {system.n_dim} dimensions.")

print(f"System initialized with {system.n_atoms} atoms, {system.n_beads} beads, and {system.n_dim} dimensions.")
print(f"Initial positions: {system.positions}",np.shape(system.positions))
print(f"Initial velocities: {system.velocities}",np.shape(system.velocities))
print(f"Initial centroid positions: {system.q_centroid}",np.shape(system.q_centroid))
print(f"Initial centroid velocities: {system.v_centroid}",np.shape(system.v_centroid))
print(f"Initial masses: {system.masses}",np.shape(system.masses))
print(f"Initial forces: {system.forces}",np.shape(system.forces))


#Set the force field
forces = inp["forces"]
force_field = Force_Field(forces)

#Set Thermostat
thermostat_type = inp.get("thermostat", None)
tau = inp.get("tau", 1)
thermostat = Thermostat(simulation_temperature,tau,thermostat_type,rng)

#Set the simulation.
timestep = inp["timestep"]
n_steps = inp["n_steps"]
prefix = inp.get("prefix", "simulation")
stride_out = inp.get("stride_out", 1000)
stride_pos = inp.get("stride_pos", 1000)
stride_vel = inp.get("stride_vel", 1000)
stride_log = inp.get("stride_log", 1000)
simulation = Simulation(system,simulation_temperature,timestep,force_field,thermostat,prefix=prefix,stride_out=stride_out,stride_pos=stride_pos,stride_vel=stride_vel,stride_log=stride_log)
#Run simulation
start_time = time.perf_counter()
#simulation.run_PIMC(n_steps,record_bead_positions=[0])
#simulation.run_PIMC_adaptive(n_steps,record_bead_positions=[0])
simulation.run_PIMC_enhanced(n_steps,record_bead_positions=[0])
end_time = time.perf_counter()
total_seconds = end_time - start_time
hours, rem = divmod(total_seconds, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\nSimulation completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
