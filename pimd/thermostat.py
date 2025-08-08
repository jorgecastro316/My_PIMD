import numpy as np
from pimd.utils import *

class Thermostat(object):
    def __init__(self, target_temperature=300, thermostat_strenght=10000, thermostat_type = None, rng=None):
        self.temperature = target_temperature #Temperatue at which the simulation is run -> system temperature * n_beads
        self.tau = thermostat_strenght 
        self.thermostat_type = thermostat_type
        self.rng = rng
        #self.rng = np.random.mtrand.RandomState(seed)
      

    def PILE_L(self, p_nm, time_step, mass, n_beads):
        #temperature = 300
        #print('C1' ,np.exp(-0.5*time_step/self.tau),-0.5*time_step/self.tau)        
        beta = temperature_to_beta(self.temperature) 
        Wp = 1/beta
        #print('p',p_nm,np.shape(p_nm))

        #Calculate Wk
        Wk_vec = np.array([2*Wp*np.sin(k*np.pi/n_beads) for k in range(n_beads)])
        #print('Wk ',Wk_vec,np.shape(Wk_vec))

        #Calculate r_k
        r_k_vec = np.array([1/self.tau if k == 0 else 2*Wk_vec[k] for k in range(n_beads)])
        #print('rk',r_k_vec,np.shape(r_k_vec))
        
        #Calculate C1
        C1_vec = np.exp(-0.5*time_step*r_k_vec)
        #print(0.5*time_step*r_k_vec)
        #print('C1',C1_vec,np.shape(C1_vec))

        #Calculate C2
        uno = np.array([1 for k in range(n_beads)])
        C2_vec = np.sqrt(uno - (C1_vec**2))
        #print('C2',C2_vec,np.shape(C2_vec))       
        #Generate Gaussian noise vector        
        #E_noise = self.rng.standard_normal(np.shape(p_nm))
        E_noise = self.rng.standard_normal(np.shape(p_nm))
        #print('E_noise',E_noise,np.shape(E_noise))
        
        #Update velocities
        new_p_nm = np.zeros(np.shape(p_nm))
        for k in range(n_beads):
            #print('C2',np.sqrt(Nbeads*1/B)*C2_vec[k])
            #print(C1_vec[k],p_nm[k],C1_vec[k]*p_nm[k])
            new_p_nm[k] = C1_vec[k]*p_nm[k] + np.sqrt(mass/beta)*C2_vec[k]*E_noise[k]
        
        #print('new_p',new_p_nm,np.shape(new_p_nm))
        #return p_nm
        return new_p_nm

    def PILE_G(self, p_nm, time_step, mass, n_beads, dof, K):
        
        beta = temperature_to_beta(self.temperature)
        #beta = 1
        #print('beta',beta)
        Wp = 1/beta
        #print('p',p_nm)

        #Calculate Wk
        Wk_vec = np.array([2*Wp*np.sin(k*np.pi/n_beads) for k in range(n_beads)])
        #print('Wk ',Wk_vec,np.shape(Wk_vec))
        Wk_vec = np.array([2*Wp*np.sin(k*np.pi/n_beads) for k in range(n_beads)])
        #Calculate r_k
        r_k_vec = np.array([1/self.tau if k == 0 else 2*Wk_vec[k] for k in range(n_beads)])
        #print('rk',r_k_vec,np.shape(r_k_vec))
        
        #Calculate C1
        C1_vec = np.exp(-0.5*time_step*r_k_vec)
        #print('C1',C1_vec,np.shape(C1_vec))

        #Calculate C2
        uno = np.array([1 for k in range(n_beads)])
        C2_vec = np.sqrt(uno - (C1_vec**2))
        #print('C2',C2_vec,np.shape(C2_vec))
                
         
        #SVR thermostat
        C = np.exp(-time_step * r_k_vec[0] * 0.5)
        r1 = self.rng.standard_normal()
        #rg = 0  #I am applying the SVR to a single atom at the time, so rg = 0
        rg = self.rng.chisquare(dof - 1) if dof > 1 else 0.0
        alpha2 = C + (((1 - C) * (r1**2 + rg)) / (2 * K * beta)) + 2 * r1 * np.sqrt((C * (1 - C)) / (2 * beta * K))

        
        alpha = np.sqrt(alpha2)
        
        if (r1 + np.sqrt(2 * K / (0.5*(1/beta)) * C / (1 - C))) < 0:
            alpha *= -1

        #print('alpha',alpha)
        #Update velocities
        new_p_nm = np.zeros(np.shape(p_nm))
        #Thermalize the centroid with SVR  
        new_p_nm[0] = alpha*p_nm[0]
        #rand_test = self.rng.standard_normal()
        #new_p_nm[0] = C1_vec[0]*p_nm[0] + np.sqrt(mass/beta)*C2_vec[0]*rand_test #Start from here tomorrow
        #print('nco',new_p_nm[0])
        
        #pile_l for the remaining normal modes
        for k in range(1,n_beads):        
            E_noise = self.rng.standard_normal()
            #print('E_noise',E_noise)
            new_p_nm[k] = C1_vec[k]*p_nm[k] + np.sqrt(mass/beta)*C2_vec[k]*E_noise        

        #print('palph ',new_p_nm,np.shape(new_p_nm))
        return new_p_nm
        #E_noise = self.rng.standard_normal(np.shape(p_nm))
        #print('E_noise',E_noise,np.shape(E_noise))
        
        #Update velocities
        #new_p_nm = np.zeros(np.shape(p_nm))
        #for k in range(n_beads):
            #print('C2',np.sqrt(Nbeads*1/B)*C2_vec[k])
            #print(C1_vec[k],p_nm[k],C1_vec[k]*p_nm[k])
        #    new_p_nm[k] = C1_vec[k]*p_nm[k] + np.sqrt(mass/beta)*C2_vec[k]*E_noise[k]
        
        #print('new_p',new_p_nm,np.shape(new_p_nm))
        #return p_nm
        #return new_p_nm


    def Andersen(self,p, masses, dt):
        """
        Applies Andersen thermostat to a copy of ring-polymer momenta.

        Parameters
        ----------
        p : ndarray, shape (nbeads, natoms * ndim)
            Original momenta of all beads in atomic units.
        masses : ndarray, shape (natoms,)
            Masses of atoms in atomic units.
        dt : float
            Time step in atomic units.

        Returns
        -------
        p_new : ndarray, shape (nbeads, natoms * ndim)
            Momentas after Andersen thermostat collisions.
        """
        nu = 1/self.tau
        if self.rng is None:
            rng = np.random.default_rng()
        else:
            rng = self.rng

        nbeads, dof = p.shape
        natoms = len(masses)
        ndim = dof // natoms

        mvec = np.repeat(masses, ndim)
        beta = temperature_to_beta(self.temperature)
        k_B_Hartree = 1.0 / temperature_to_beta(1.0)

        # Standard deviation of the Maxwell-Boltzmann distribution
        stddev = np.sqrt(mvec * k_B_Hartree * self.temperature)

        p_new = p.copy()

        for b in range(nbeads):
            for i in range(dof):
                if rng.random() < nu * dt:
                    p_new[b, i] = rng.normal(0.0, stddev[i])

        return p_new

    
                

