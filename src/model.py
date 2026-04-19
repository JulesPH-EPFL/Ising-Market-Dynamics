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
def run_simulation_numba(lattice, L, N, J, h, beta, Delta_t, t_therma):
    m_history = np.empty(Delta_t - t_therma)
    
    for t in range(Delta_t):
        for _ in range(N):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            delta_h = get_delta_h_numba(lattice, L, J, h, i, j)
                        
            if delta_h <= 0 or np.random.random() < np.exp(-delta_h * beta):
                lattice[i, j] *= -1
        
        if t >= t_therma:
            m_history[t - t_therma] = np.sum(lattice) / N
        
    return m_history, lattice


class IsingLattice:
    def __init__(self, N, T, J, h, Delta_t):
        self.L = int(np.sqrt(N))
        self.N = self.L ** 2 #number of agents
        self.T = T #"temperature"
        self.J = J #coupling constant
        self.h = h #external influence
        self.Delta_t = Delta_t #number of iterations
        self.t_therma = N #time of thermalisation
        self.lattice = np.random.randint(0,2,(self.L,self.L)).astype(np.int8) * 2 - 1 #dtype = int8 for faster executions
        
    def get_total_energy(self):
        return get_total_energy_numba(self.lattice, self.L, self.J, self.h)
        
    def get_delta_h(self, i, j):
        return get_delta_h_numba(self.lattice, self.L, self.J, self.h, i, j)
    
    def run(self):
        beta = 1.0 / self.T
        m_history, final_lattice = run_simulation_numba(
            self.lattice, self.L, self.N, self.J, self.h, beta, self.Delta_t, self.t_therma
        )
        self.lattice = final_lattice
        return m_history