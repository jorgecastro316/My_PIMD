import numpy as np
from integrator import integrate_one_step

# Harmonic oscillator parameters
omega_ho = 1.0  # a.u.
mass = np.array([1.0])  # 1D single particle
dt = 0.1  # time step in a.u.
n_steps = 10
n_beads = 1
dim = 1
temperature = 315775

# Initial conditions
positions_init = np.array([[[1.0]]])  # shape (1,1,1)
velocities_init = np.array([[[0.0]]])  # shape (1,1,1)

# Harmonic force function
def harmonic_force(x):
    return -mass.reshape(1,1,1) * omega_ho**2 * x

# Copy for each method
pos_vv = positions_init.copy()
vel_vv = velocities_init.copy()

pos_pi = positions_init.copy()
vel_pi = velocities_init.copy()

# Integrate both
print(f"{'Step':<4} {'x_vv':>12} {'x_pi':>12} {'|Δx|':>12}")
for step in range(n_steps):
    pos_vv, vel_vv, _ = integrate_one_step("basic", pos_vv, vel_vv, mass, harmonic_force, dt,temperature)
    pos_pi, vel_pi, _ = integrate_one_step("path_integral", pos_pi, vel_pi, mass, harmonic_force, dt,temperature)
    delta = np.abs(pos_vv - pos_pi)
    print(f"{step:<4} {pos_vv[0,0,0]:>12.6f} {pos_pi[0,0,0]:>12.6f} {delta[0,0,0]:>12.2e}")

