# main.py
import numpy as np
from pimd.system import *
from pimd.io_utils import *

# Read input parameters
inp = read_input_file()
dt = inp['timestep']

# Load system and initialize ring polymer
system, dim = load_system(inp['geometry_file'])
positions, velocities, masses = initialize_ring_polymer(system, inp['n_beads'], inp['temperature'], velocity_file=inp['velocity_file'])

# Output for inspection
print(f"Dimensionality: {dim}D")
print(f"Initial positions shape: {positions.shape}")
print(f"Initial positions: {positions}")  # Show first bead's positions
print(f"Initial velocities: {velocities}")  # Show first bead's velocities
print(f"Initial velocities shape: {velocities.shape}")
print(f"Masses shape: {masses.shape}")
print(f"Masses: {masses}")
print(f"Time step (a.u.): {dt}")
