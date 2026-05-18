import numpy as np
from numba import njit

@njit
def get_total_energy_numba(lattice, L, J, h):
    term_coupling = 0
    term_external = 0
    
    for i in range(L):
        for j in range(L):
            term_coupling += lattice[i,j] * (lattice[(i+1) % L,j] + lattice[i,(j+1) % L])
            
            term_external += lattice[i,j]
            
    return - J * term_coupling - h * term_external


@njit 
def get_delta_h_numba(lattice, L, J, h, i, j):
    spin = lattice [i,j]
    neighbor_sum = (
        lattice[(i + 1) % L, j] + 
        lattice[(i - 1) % L, j] + 
        lattice[i, (j + 1) % L] + 
        lattice[i, (j - 1) % L]   
    )
    return 2 * spin * (J * neighbor_sum + h)
                
                
@njit
def run_simulation_numba(lattice, L, N, J, h_0, rho, sigma_h, beta, Delta_t, t_therma):
    M_history = np.empty(Delta_t - t_therma)
    h_past = h_0
    
    for t in range(Delta_t):
        epsilon_t = np.random.normal(0, sigma_h)
        h_t = rho * h_past + epsilon_t
        
        for _ in range(N):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            
            delta_h = get_delta_h_numba(lattice, L, J, h_t, i, j)
                        
            if delta_h <= 0 or np.random.random() < np.exp(-delta_h * beta):
                lattice[i, j] *= -1
        
        if t >= t_therma:
            M_history[t - t_therma] = np.sum(lattice)
            
        h_past = h_t
        
    return M_history, lattice


class IsingLattice:
    def __init__(self, N, T, J, Delta_t, h_0, rho, sigma_h):
        
        self.L = int(np.sqrt(N))
        self.N = self.L ** 2 
        self.T = float(T)
        self.J = float(J)
        self.h_0 = float(h_0)
        
        self.rho = float(rho)
        self.sigma_h = float(sigma_h)
        
        self.Delta_t = int(Delta_t)
        self.t_therma = int(0.1*Delta_t) 
        self.lattice = np.random.randint(0, 2, (self.L, self.L)).astype(np.int8) * 2 - 1
        
    def get_total_energy(self):
        return get_total_energy_numba(self.lattice, self.L, self.J, self.h)
        
    def get_delta_h(self, i, j):
        return get_delta_h_numba(self.lattice, self.L, self.J, self.h, i, j)
    
    def run(self):
        beta = 1.0 / self.T
        M_history, final_lattice = run_simulation_numba(
            self.lattice, self.L, self.N, self.J, self.h_0, self.rho, self.sigma_h, beta, self.Delta_t, self.t_therma
        )
        self.lattice = final_lattice
        return M_history