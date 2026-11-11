import unittest

import numpy as np

from pimd.force_field import Force_Field
from pimd.simulation import Simulation
from pimd.system import System
from pimd.thermostat import Thermostat


class CoreTests(unittest.TestCase):
    def test_harmonic_force_and_energy(self):
        potential = {"type": "harmonic", "args": [0.5]}
        force_field = Force_Field([potential])
        coordinates = np.array([[2.0], [-1.0]])

        forces, energies = force_field.eq_force_function(
            coordinates, n_beads=2, n_particles=1, n_dim=1
        )

        np.testing.assert_allclose(forces[:, 0], [-0.5, 0.25])
        np.testing.assert_allclose(energies[:, 0], [0.5, 0.125])

    def test_seed_controls_thermal_velocities(self):
        kwargs = {
            "geometry_file": None,
            "n_beads": 4,
            "temperature": 300.0,
            "centroid_q": np.zeros(3),
            "symbols": ["H"],
        }
        system_a = System(rng=np.random.RandomState(17), **kwargs)
        system_b = System(rng=np.random.RandomState(17), **kwargs)

        np.testing.assert_array_equal(system_a.velocities, system_b.velocities)

    def test_centroid_setters_use_the_only_centroid_row(self):
        system = System(
            geometry_file=None,
            n_beads=2,
            rng=np.random.RandomState(1),
            temperature=300.0,
            centroid_q=np.zeros(3),
            symbols=["H"],
        )

        system.set_centroid_q(np.array([1.0, 2.0, 3.0]), 0)
        system.set_centroid_v(np.array([4.0, 5.0, 6.0]), 0)

        np.testing.assert_array_equal(system.q_centroid[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(system.v_centroid[0], [4.0, 5.0, 6.0])

    def test_enhanced_pimc_returns_an_active_learning_pool(self):
        rng = np.random.RandomState(23)
        n_beads = 4
        temperature = 300.0 * n_beads
        system = System(
            geometry_file=None,
            n_beads=n_beads,
            rng=rng,
            temperature=temperature,
            centroid_q=np.array([0.1]),
            symbols=["H"],
        )
        force_field = Force_Field([{"type": "harmonic", "args": [0.01]}])
        thermostat = Thermostat(temperature, 100.0, "NVE", rng)
        simulation = Simulation(
            system,
            temperature,
            0.1,
            force_field,
            thermostat,
            stride_out=0,
            stride_pos=0,
            stride_vel=0,
            stride_log=0,
        )

        positions, velocities = simulation.run_PIMC_enhanced(
            3,
            enable_bisection=False,
            out_text=False,
            gen_velocities=True,
        )

        self.assertEqual(positions.shape, (3, n_beads, 1))
        self.assertEqual(velocities.shape, (3, n_beads, 1))
        self.assertEqual(simulation.last_acceptance_stats["accepted"], 3)


if __name__ == "__main__":
    unittest.main()
