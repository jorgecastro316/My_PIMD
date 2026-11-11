import numpy as np
from pimd.utils import *
class System(object):
    def __init__(self, geometry_file=None, n_beads=None, rng=None,
                 velocity_file=None, temperature=None, centroid_q=None, symbols=None):
        """
        geometry_file : str | list[str] | None
            Path(s) to .xyz geometry file(s). If None, must provide centroid_q and symbols.
        n_beads : int
            Number of beads.
        rng : np.random.RandomState
            Random number generator.
        velocity_file : str | list[str] | None
            Optional file(s) for initial velocities.
        temperature : float | None
            Temperature (K). Required if no velocity file.
        centroid_q : ndarray | None
            1D centroid coordinates (natoms*ndim,).
        symbols : list[str] | None
            Atomic symbols, required if using centroid_q.
        """
        self.initialize(geometry_file, n_beads, rng,
                        velocity_file, temperature, centroid_q, symbols)

    def initialize(self, geometry_file, n_beads, rng,
                   velocity_file=None, temperature=None, centroid_q=None, symbols=None):
        self.n_beads = n_beads
        self.rng = rng

        # === Case 1: initialize from file(s) ===
        if geometry_file is not None:
            if isinstance(geometry_file, list):
                if len(geometry_file) != n_beads:
                    raise ValueError("Number of geometry files must match number of beads.")
                first = load_xyz_file_with_units(geometry_file[0], unit_type='position')
                symbols = first['symbols']
                coords = np.array(first['coords'], dtype=np.float64)
                n_atoms, n_dim = coords.shape
                positions = np.zeros((n_beads, n_atoms * n_dim), dtype=np.float64)
                for i in range(n_beads):
                    data = load_xyz_file_with_units(geometry_file[i], unit_type='position')
                    if data['symbols'] != symbols:
                        raise ValueError("All bead geometries must have matching atomic symbols.")
                    coords = np.array(data['coords'], dtype=np.float64)
                    positions[i] = coords.flatten()
            else:
                data = load_xyz_file_with_units(geometry_file, unit_type='position')
                symbols = data['symbols']
                coords = np.array(data['coords'], dtype=np.float64)
                n_atoms, n_dim = coords.shape
                positions = np.tile(coords.reshape(1, -1), (n_beads, 1)).astype(np.float64)

        # === Case 2: initialize from centroid ===
        elif centroid_q is not None:
            if symbols is None:
                raise ValueError("Must provide symbols when initializing from centroid_q.")
            centroid_q = np.array(centroid_q, dtype=np.float64)
            n_atoms = len(symbols)
            n_dim = centroid_q.size // n_atoms
            if centroid_q.size != n_atoms * n_dim:
                raise ValueError("centroid_q size mismatch with natoms*ndim.")
            positions = np.tile(centroid_q.reshape(1, -1), (n_beads, 1)).astype(np.float64)

        else:
            raise ValueError("Must provide either geometry_file or centroid_q+symbols.")

        self.symbols = symbols
        self.n_atoms = n_atoms
        self.n_dim = n_dim

        # === Masses ===
        try:
            self.masses = np.array(get_atomic_masses_au(symbols), dtype=np.float64)
        except KeyError as e:
            raise ValueError(f"Unknown atomic symbol '{e.args[0]}' encountered.")

        # === Velocities ===
        if velocity_file:
            if isinstance(velocity_file, list):
                if len(velocity_file) != n_beads:
                    raise ValueError("Number of velocity files must match number of beads.")
                velocities = np.zeros((n_beads, n_atoms * n_dim), dtype=np.float64)
                for i in range(n_beads):
                    vdata = load_xyz_file_with_units(velocity_file[i], unit_type='velocity')
                    if vdata['symbols'] != symbols:
                        raise ValueError("Velocity file symbols do not match geometry file symbols.")
                    velocities[i] = np.array(vdata['coords'], dtype=np.float64).flatten()
            else:
                vdata = load_xyz_file_with_units(velocity_file, unit_type='velocity')
                if vdata['symbols'] != symbols:
                    raise ValueError("Velocity file symbols do not match geometry file symbols.")
                v_coords = np.array(vdata['coords'], dtype=np.float64)
                velocities = np.tile(v_coords.reshape(1, -1), (n_beads, 1)).astype(np.float64)
        else:
            if temperature is None:
                raise ValueError("Temperature must be provided if no velocity file is given.")
            masses = np.array(self.masses, dtype=np.float64).reshape(n_atoms)
            velocities = thermal(n_beads, n_atoms, n_dim, temperature, masses, rng=rng)

        # === Assign ===
        self.positions = positions
        self.velocities = velocities
        self.q_centroid = np.mean(positions, axis=0, keepdims=True)
        self.v_centroid = np.mean(velocities, axis=0, keepdims=True)

        # === Energies ===
        self.forces = np.zeros_like(self.positions)
        self.potential_energies = np.zeros((n_beads, n_atoms), float)
        self.kinetic_energies = np.zeros((n_beads, n_atoms), float)
        self.kin = self.pot = self.tot = 0
        self.E_kin = self.E_pot = self.E_tot = 0
        self.int_pot = 0
        self.Temperature = 0


    #For simulation initialization only

    def set_bead_position(self,new_position,bead_number,particle_number):
        self.positions[bead_number,particle_number*self.n_dim:self.n_dim*(particle_number+1)] = new_position

    def set_bead_velocity(self,new_velocity,bead_number,particle_number):
        self.velocities[bead_number,particle_number*self.n_dim:self.n_dim*(particle_number+1)] = new_velocity

    def set_centroid_q(self,new_qc,particle_number):
        self.q_centroid[0,particle_number*self.n_dim:self.n_dim*(particle_number+1)] = new_qc

    def set_centroid_v(self,new_vc,particle_number):
        self.v_centroid[0,particle_number*self.n_dim:self.n_dim*(particle_number+1)] = new_vc

    #Setting
    def set_positions(self,new_positions):
        self.positions = new_positions

    def set_velocities(self,new_velocities):
        self.velocities = new_velocities

    def set_qc(self,new_qc):
        self.q_centroid = new_qc

    def set_vc(self,new_vc):
        self.v_centroid = new_vc

    def set_forces(self,new_forces):
        self.forces = new_forces

    def set_potential_energies(self,new_potential_energies):
        self.potential_energies = new_potential_energies

    def set_kinetic_energies(self,new_kinetic_energies):
        self.kinetic_energies = new_kinetic_energies

    def set_kin(self,new_kin):
        self.kin = new_kin

    def set_pot(self,new_pot):
        self.pot = new_pot

    def set_tot(self,new_tot):
        self.tot = new_tot

    def set_E_kin(self,new_E_kin):
        self.E_kin = new_E_kin

    def set_E_pot(self,new_E_pot):
        self.E_pot = new_E_pot

    def set_E_tot(self,new_E_tot):
        self.E_tot = new_E_tot

    def set_int_pot(self,new_int_pot):
        self.int_pot = new_int_pot

    def set_Temperature(self,new_Temperature):
        self.Temperature = new_Temperature

    def read_bead_position(self,positions_file,line_num = 1):
        print('Reading positions from: ', positions_file)
        line = linecache.getline(positions_file,line_num)
        line = line.split()
        #Check dimension compatibility

        if len(line) != self.n_atoms*self.n_dim:
            print('Incompatible dimensions: Check your input files!')

        bead_q = np.zeros((1,self.n_atoms*self.n_dim))

        for i in range(len(line)):
            bead_q[0,i] = float(line[i])

        return(bead_q)

    def build_q(self,bead_pos_list,line_num = 1):
        positions = np.empty((self.n_beads,self.n_atoms*self.n_dim),dtype=object)

        if len(bead_pos_list) == 1:
            for i in range(self.n_beads):
                positions[i,0:(self.n_atoms*self.n_dim)] = self.read_bead_position(bead_pos_list[0],line_num)
        else:
            for i in range(self.n_beads):
                positions[i,0:(self.n_atoms*self.n_dim)] = self.read_bead_position(bead_pos_list[i],line_num)

        return(positions)

    #Getting

    def get_positions(self):
        return self.positions

    def get_velocities(self):
        return self.velocities

    def get_forces(self):
        return self.forces

    def get_centroids(self):
        return self.q_centroid

    def get_potential_energies(self):
        return self.potential_energies
