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
#from pimd.integrator import Integrator
#from pimd.thermostat import Thermostat
from pimd.utils import *
from scipy.constants import *
import numpy as np


# === Read input ===
inp = read_input_file("input.txt")

def parse_file_field(value):
    if isinstance(value, str):
        tokens = value.strip().split()
        return tokens if len(tokens) > 1 else tokens[0]
    return value

geometry_file = parse_file_field(inp["geometry_file"])
velocity_file = parse_file_field(inp.get("velocity_file", None))
n_beads = inp["n_beads"]
system_temperature = inp["temperature"]
print(f"System temperature: {system_temperature} K")
simulation_temperature = system_temperature * n_beads  # PIMD temperature
print(f"Simulation temperature: {simulation_temperature} K")
timestep = inp["timestep"]
#timestep = 0.5
n_steps = inp["n_steps"]
force_type, force_params = parse_force_spec(inp.get("force", "zero"))
prefix = inp.get("prefix", "simulation")
thermostat_type = inp.get("thermostat", None)
tau = inp.get("tau", 1)
#tau = 0.5
#Initialze random number generator
seed = inp.get("seed", 420)
#rng = random_number_generator(seed)
rng = np.random.mtrand.RandomState(seed)
#rng=None
#Set the system
system = System(geometry_file, n_beads, rng, velocity_file, simulation_temperature)
print(f"System initialized with {system.n_atoms} atoms, {system.n_beads} beads, and {system.n_dim} dimensions.")
print(f"Initial positions: {system.positions}",np.shape(system.positions))
print(f"Initial velocities: {system.velocities}",np.shape(system.velocities))
print(f"Initial centroid positions: {system.q_centroid}",np.shape(system.q_centroid))
print(f"Initial centroid velocities: {system.v_centroid}",np.shape(system.v_centroid))
print(f"Initial masses: {system.masses}",np.shape(system.masses))
print(f"Initial forces: {system.forces}",np.shape(system.forces))

#Set the force field

force_field = Force_Field(force_type, force_params)

#Set Thermostat
#thermostat = Thermostat(simulation_temperature,tau,thermostat_type,seed)
thermostat = Thermostat(simulation_temperature,tau,thermostat_type,rng)
#Set the simulation.
simulation = Simulation(system,simulation_temperature,timestep,force_field,thermostat,prefix)

#Run simulation
simulation.run_PIMD(n_steps)
#simulation.run_PIMC(n_steps)
#simulation.run_PIMC_adaptive(n_steps)
#simulation.run_PIMC_adaptive_advanced(n_steps)

#simulation.run_PIMC_adaptive_staging(n_steps)

# Constantes físicas en unidades atómicas


def classical_PE(T, K, ndim):
    """
    Energía potencial media clásica de un oscilador armónico
    T : temperatura en Kelvin
    K : constante del resorte en Hartree/bohr²
    ndim : número de dimensiones
    """
    beta = temperature_to_beta(T)
    return ndim / (2 * beta)

def quantum_PE(T, K, ndim, m=1837.1526):
    """
    Energía potencial media cuántica de un oscilador armónico
    T : temperatura en Kelvin
    K : constante del resorte en Hartree/bohr²
    ndim : número de dimensiones
    m : masa en unidades atómicas (por defecto: protón)
    """
    omega = np.sqrt(K / m)
    beta = temperature_to_beta(T)
    return ndim * (0.5 * omega / np.tanh(0.5 * beta * omega))

T = 300  # Kelvin
K = 1.0  # Hartree/bohr²
ndim = 1

print("Clásica:", classical_PE(T, K, ndim))
print("Cuántica:", quantum_PE(T, K, ndim))

