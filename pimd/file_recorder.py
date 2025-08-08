import numpy as np
import os

class Recorder(object):
    def __init__(self, system, prefix):
        self.system = system
        self.n_beads = system.n_beads
        self.n_atoms = system.n_atoms
        self.n_dim = system.n_dim
        self.symbols = system.symbols
        self.prefix = prefix

        # Store only used file paths, open on demand in `write`
        self.opened_files = {}

    def write(
        self,
        system,
        step,
        elapsed_time,
        record_energies=False,
        record_estimators=False,
        record_bead_positions=None,
        record_bead_velocities=None,
        record_centroid_position=False,
        record_centroid_velocity=False
    ):
        # ----------------------------
        # 1) Classical energy output
        # ----------------------------
        if record_energies:
            if "classical" not in self.opened_files:
                path = f"{self.prefix}.classical.{self.n_beads}.out"
                f = open(path, "w")
                f.write(
                    f"{'#time':>15s} {'Total':>12s} {'Kinetic':>12s} "
                    f"{'Potential':>12s} {'Temperature':>12s}\n"
                )
                self.opened_files["classical"] = f
            f = self.opened_files["classical"]
            f.write(
                f"{elapsed_time:17.15f} {system.tot:17.15f} "
                f"{system.kin:17.15f} {system.pot:17.15f} "
                f"{system.Temperature:17.15f}\n"
            )

        # ----------------------------
        # 2) Path‐integral estimators
        # ----------------------------
        if record_estimators:
            if "estimator" not in self.opened_files:
                path = f"{self.prefix}.estimator.{self.n_beads}.out"
                f = open(path, "w")
                f.write(
                    f"{'#time':>15s} {'E_tot':>17s} "
                    f"{'E_kin':>17s} {'E_pot':>17s}\n"
                )
                self.opened_files["estimator"] = f
            f = self.opened_files["estimator"]
            f.write(
                f"{elapsed_time:15.15f} {system.E_tot:17.15f} "
                f"{system.E_kin:17.15f} {system.E_pot:17.15f}\n"
            )

        # ---------------------------------------
        # 3) Per‐bead POSITION (*.xyz) output
        # ---------------------------------------
        if record_bead_positions:
            q = system.positions.reshape(self.n_beads, self.n_atoms, self.n_dim)
            for i in record_bead_positions:
                if 0 <= i < self.n_beads:
                    key_pos = f"bead_pos_{i}"
                    if key_pos not in self.opened_files:
                        path = f"{self.prefix}.pos{i}.xyz"
                        self.opened_files[key_pos] = open(path, "w")
                    f = self.opened_files[key_pos]

                    # Write one XYZ snapshot for bead i
                    f.write(f"{self.n_atoms}\n")
                    f.write(f"# time = {elapsed_time:.6f}, bead = {i}\n")
                    for a in range(self.n_atoms):
                        vec = q[i, a]
                        components = " ".join(f"{c:.8f}" for c in vec)
                        f.write(f"{self.symbols[a]:<2s} {components}\n")

        # ---------------------------------------
        # 4) Per‐bead VELOCITY (*.xyz) output
        # ---------------------------------------
        if record_bead_velocities:
            v = system.velocities.reshape(self.n_beads, self.n_atoms, self.n_dim)
            for i in record_bead_velocities:
                if 0 <= i < self.n_beads:
                    key_vel = f"bead_vel_{i}"
                    if key_vel not in self.opened_files:
                        path = f"{self.prefix}.vel{i}.xyz"
                        self.opened_files[key_vel] = open(path, "w")
                    f = self.opened_files[key_vel]

                    # Write one XYZ‐like snapshot for velocities of bead i
                    f.write(f"{self.n_atoms}\n")
                    f.write(f"# time = {elapsed_time:.6f}, bead = {i}\n")
                    for a in range(self.n_atoms):
                        vec = v[i, a]
                        components = " ".join(f"{c:.8f}" for c in vec)
                        f.write(f"{self.symbols[a]:<2s} {components}\n")

        # ---------------------------------------
        # 5) Centroid POSITION (*.xyz) output
        # ---------------------------------------
        if record_centroid_position:
            key_cpos = "centroid_pos"
            if key_cpos not in self.opened_files:
                path = f"{self.prefix}.xc.xyz"
                self.opened_files[key_cpos] = open(path, "w")
            f = self.opened_files[key_cpos]

            qc = system.q_centroid.reshape(self.n_atoms, self.n_dim)
            f.write(f"{self.n_atoms}\n")
            f.write(f"# time = {elapsed_time:.6f}\n")
            for a in range(self.n_atoms):
                vec = qc[a]
                components = " ".join(f"{c:.8f}" for c in vec)
                f.write(f"{self.symbols[a]:<2s} {components}\n")

        # ---------------------------------------
        # 6) Centroid VELOCITY (*.xyz) output
        # ---------------------------------------
        if record_centroid_velocity:
            key_cvel = "centroid_vel"
            if key_cvel not in self.opened_files:
                path = f"{self.prefix}.vc.xyz"
                self.opened_files[key_cvel] = open(path, "w")
            f = self.opened_files[key_cvel]

            vc = system.v_centroid.reshape(self.n_atoms, self.n_dim)
            f.write(f"{self.n_atoms}\n")
            f.write(f"# time = {elapsed_time:.6f}\n")
            for a in range(self.n_atoms):
                vec = vc[a]
                components = " ".join(f"{c:.8f}" for c in vec)
                f.write(f"{self.symbols[a]:<2s} {components}\n")

    def close(self):
        for f in self.opened_files.values():
            f.close()




