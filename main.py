import numpy as np
from pimd.io_utils import *
from pimd.system import *
from pimd.integrator import integrate_one_step
from pimd.force_field import *

def main():
    # === Read input ===
    inp = read_input_file("input.txt")
    geometry_file = inp["geometry_file"]
    velocity_file = inp.get("velocity_file")
    temperature = inp["temperature"]
    n_beads = inp["n_beads"]
    n_steps = inp["n_steps"]
    dt = inp["timestep"]  # already in a.u.

    # === Load and initialize ring polymer ===
    positions, velocities, masses, dim = initialize_ring_polymer(
        geometry_file=geometry_file,
        velocity_file=velocity_file,
        temperature=temperature,
        n_beads=n_beads
    )


    # === Parse and select force function ===
    force_type, force_params = parse_force_spec(inp.get("force", "zero"))
    force_fn = force_field(force_type, force_params)

    # === Print initial conditions ===
    print_initial_conditions(
        positions=positions,
        velocities=velocities,
        masses=masses,
        force_type=force_type,
        force_params=force_params,
        temperature=temperature,
        dt=dt,
        n_steps=n_steps
    )

    # === Time evolution ===
    for step in range(1,n_steps+1): # 1-based indexing for steps since step 0 is initial state
        positions, velocities, forces = integrate_one_step(
            method="path_integral",  # or "basic"
            positions=positions,
            velocities=velocities,
            masses=masses,
            forces_fn=force_fn,
            dt=dt,
            temperature=temperature
        )
        print(f"Step {step:4d}", positions)

if __name__ == "__main__":
    main()
