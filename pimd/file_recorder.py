import os

class Recorder(object):
    def __init__(self, system, prefix, line_buffered=True, fsync_every=None):
        """
        line_buffered: if True, open text files with line buffering (flush on newline)
        fsync_every:   if set to an integer N, call os.fsync() every N calls to write()
        """
        self.system = system
        self.n_beads = system.n_beads
        self.n_atoms = system.n_atoms
        self.n_dim = system.n_dim
        self.symbols = system.symbols
        self.prefix = prefix

        self._buffering = 1 if line_buffered else -1
        self._fsync_every = fsync_every
        self._write_count = 0

        # Store only used file paths, open on demand in `write`
        self.opened_files = {}

    def _open(self, key, path, header=None):
        """Open file with desired buffering and optional header."""
        if key not in self.opened_files:
            f = open(path, "w", buffering=self._buffering)
            if header:
                f.write(header)
                if not header.endswith("\n"):
                    f.write("\n")
            self.opened_files[key] = f
        return self.opened_files[key]

    def _flush(self, f):
        """Flush and optionally fsync at the requested cadence."""
        f.flush()
        self._write_count += 1
        if self._fsync_every and self._fsync_every > 0:
            if (self._write_count % self._fsync_every) == 0:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # On some filesystems (e.g., certain network FS), fsync may be unsupported.
                    pass

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
        record_centroid_velocity=False,
        record_work=False,
        work=0.0
    ):
        # ----------------------------
        # 1) Classical energy output
        # ----------------------------
        if record_energies:
            f = self._open(
                "classical",
                f"{self.prefix}.classical.{self.n_beads}.out",
                header=f"{'#time':>15s} {'Total':>12s} {'Kinetic':>12s} {'Potential':>12s} {'Temperature':>12s}"
            )
            f.write(
                f"{elapsed_time:17.15f} {system.tot:17.15f} "
                f"{system.kin:17.15f} {system.pot:17.15f} "
                f"{system.Temperature:17.15f}\n"
            )
            self._flush(f)

        # ----------------------------
        # 2) Path‐integral estimators
        # ----------------------------
        if record_estimators:
            f = self._open(
                "estimator",
                f"{self.prefix}.estimator.{self.n_beads}.out",
                header=f"{'#time':>15s} {'E_tot':>17s} {'E_kin':>17s} {'E_pot':>17s}"
            )
            f.write(
                f"{elapsed_time:15.15f} {system.E_tot:17.15f} "
                f"{system.E_kin:17.15f} {system.E_pot:17.15f}\n"
            )
            self._flush(f)

        # ---------------------------------------
        # 3) Per‐bead POSITION (*.xyz) output
        # ---------------------------------------
        if record_bead_positions:
            q = system.positions.reshape(self.n_beads, self.n_atoms, self.n_dim)
            for i in record_bead_positions:
                if 0 <= i < self.n_beads:
                    key_pos = f"bead_pos_{i}"
                    f = self._open(key_pos, f"{self.prefix}.pos{i}.xyz")
                    f.write(f"{self.n_atoms}\n")
                    f.write(f"# time = {elapsed_time:.6f}, bead = {i}, step = {step}\n")
                    for a in range(self.n_atoms):
                        vec = q[i, a]
                        components = " ".join(f"{c:.8f}" for c in vec)
                        f.write(f"{self.symbols[a]:<2s} {components}\n")
                    self._flush(f)

        # ---------------------------------------
        # 4) Per‐bead VELOCITY (*.xyz) output
        # ---------------------------------------
        if record_bead_velocities:
            v = system.velocities.reshape(self.n_beads, self.n_atoms, self.n_dim)
            for i in record_bead_velocities:
                if 0 <= i < self.n_beads:
                    key_vel = f"bead_vel_{i}"
                    f = self._open(key_vel, f"{self.prefix}.vel{i}.xyz")
                    f.write(f"{self.n_atoms}\n")
                    f.write(f"# time = {elapsed_time:.6f}, bead = {i}, step = {step}\n")
                    for a in range(self.n_atoms):
                        vec = v[i, a]
                        components = " ".join(f"{c:.8f}" for c in vec)
                        f.write(f"{self.symbols[a]:<2s} {components}\n")
                    self._flush(f)

        # ---------------------------------------
        # 5) Centroid POSITION (*.xyz) output
        # ---------------------------------------
        if record_centroid_position:
            f = self._open("centroid_pos", f"{self.prefix}.xc.xyz")
            qc = system.q_centroid.reshape(self.n_atoms, self.n_dim)
            f.write(f"{self.n_atoms}\n")
            f.write(f"# time = {elapsed_time:.6f}\n")
            for a in range(self.n_atoms):
                vec = qc[a]
                components = " ".join(f"{c:.8f}" for c in vec)
                f.write(f"{self.symbols[a]:<2s} {components}\n")
            self._flush(f)

        # ---------------------------------------
        # 6) Centroid VELOCITY (*.xyz) output
        # ---------------------------------------
        if record_centroid_velocity:
            f = self._open("centroid_vel", f"{self.prefix}.vc.xyz")
            vc = system.v_centroid.reshape(self.n_atoms, self.n_dim)
            f.write(f"{self.n_atoms}\n")
            f.write(f"# time = {elapsed_time:.6f}\n")
            for a in range(self.n_atoms):
                vec = vc[a]
                components = " ".join(f"{c:.8f}" for c in vec)
                f.write(f"{self.symbols[a]:<2s} {components}\n")
            self._flush(f)

        # ---------------------------------------
        # 7) Work output
        # ---------------------------------------
        if record_work:
            f = self._open(
                "work",
                f"{self.prefix}.work.out",
                header=f"{'#time':>15s} {'Work':>17s}"
            )
            f.write(f"{elapsed_time:15.15f} {work:17.15f}\n")
            self._flush(f)

    def close(self):
        for f in self.opened_files.values():
            try:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
                f.close()
            except Exception:
                pass
        self.opened_files.clear()
