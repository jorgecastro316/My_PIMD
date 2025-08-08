import numpy as np
from pimd.utils import *

class Force_Field(object):

    def __init__(self, force_type, force_params):
        self.type = force_type 
        self.params = force_params

    def harmonic(self, q, n_beads, n_particles, n_dim, params):
        K = params[0] if params else 1.0
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim

            # shape (n_beads, n_dim) for this particle across beads
            q_i = q[:, start:end]

            # Force: F = -K * q
            forces[:, start:end] = -K * q_i

            # Potential: 0.5 * K * |q|^2 (per particle, per bead)
            potential_energies[:, i] = 0.5 * K * np.sum(q_i**2, axis=1)

        return forces, potential_energies

    def double_well(self, q, n_beads, n_particles, n_dim, params):
        D = params[0] if params else 1.0
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]  # shape (n_beads, n_dim)

            # Force: F = -∇V = -D * (-4q^3 + 2q)
            forces[:, start:end] = -D * (-4 * q_i**3 + 2 * q_i)

            # Potential: D * (q^4 - q^2)
            potential_energies[:, i] = D * np.sum(q_i**4 - q_i**2, axis=1)

        return forces, potential_energies


    def double_well_asym(self, q, n_beads, n_particles, n_dim, params):
        D = params[0] if params else 1.0
        forces = np.zeros_like(q)
        potential_energies = np.zeros((n_beads, n_particles), float)

        for i in range(n_particles):
            start = i * n_dim
            end = (i + 1) * n_dim
            q_i = q[:, start:end]  # shape (n_beads, n_dim)

            # Force: -D * (4q^3 - 2q + 1)
            forces[:, start:end] = -D * (4 * q_i**3 - 2 * q_i + 1)

            # Potential: D * (q^4 - q^2 + q)
            potential_energies[:, i] = D * np.sum(q_i**4 - q_i**2 + q_i, axis=1)

        return forces, potential_energies

    
    def zero_force(self,q,n_beads, n_particles, n_dim):
        forces = np.zeros((n_beads, n_dim*n_particles),float)
        potential_energies = np.zeros((n_beads,n_particles),float)
        return(forces,potential_energies)
 

    def eq_force_function(self, q, n_beads, n_particles, n_dim):
        forces, potential_energies = self.ff( q, n_beads, n_particles, n_dim, self.type, self.params)        

        return forces, potential_energies

    def ff(self, q, n_beads, n_particles, n_dim, type, params):
        
        if type == 'harmonic':
            forces, potential_energies = self.harmonic(q, n_beads, n_particles, n_dim, params)
        
        elif type == 'double_well':
            forces, potential_energies = self.double_well(q, n_beads, n_particles, n_dim, params)
        
        elif type == 'double_well_asym':
            forces, potential_energies = self.double_well_asym(q, n_beads, n_particles, n_dim, params)

        elif type == "zero":
            forces, potential_energies = self.zero_force(q, n_beads, n_particles, n_dim)
        
        else:
            raise ValueError("Unknown force field type: {}".format(type))
        
        return forces, potential_energies

    
    #def noneq_force_function(self, q, n_beads, n_particles, n_dim, lambda_t):
        #print(lambda_t)
    #    potential_0 = self.potential[0].split()
    #    potential_1 = self.potential[1].split()
        
    #    forces_0,potential_energies_0 = self.ff(q, n_beads, n_particles, n_dim, potential_0)
    #    forces_1,potential_energies_1 = self.ff(q, n_beads, n_particles, n_dim, potential_1)
        
    #    forces = (lambda_t)*forces_1 + (1 - lambda_t)*forces_0
    #    potential_energies = (lambda_t)*potential_energies_1 + (1 - lambda_t)*potential_energies_0
        
    #    pot = (1/n_beads)*np.sum(potential_energies)
        #print(pot)
    #    return forces, potential_energies, pot 
