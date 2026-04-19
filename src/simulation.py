from .model import IsingLattice
import numpy as np
import pandas as pd
import os

class MarketSimulator:
    def __init__(self, N, T, J, h, Delta_t, P_0, lambd):
        self.N = N
        self.T = T
        self.J = J
        self.h = h
        self.Delta_t = Delta_t
        self.P_0 = P_0 #initial price
        self.lambd = lambd #liquidity
        self.lattice = IsingLattice(N, T, J, h, Delta_t)
        
    
    def get_price_history(self):
        m_history = self.lattice.run()
        m_history = np.insert(m_history, 0, 0)
        r_history = m_history / self.lambd
        sum_ = np.cumsum(r_history)
        P_history = self.P_0 * np.exp(sum_)
        return m_history, r_history, P_history
    
    def export(self, m_history, r_history, P_history):
        time = range(len(m_history))
        storage = {
            'time': time,
            'm_history': m_history,
            'r_history': r_history,
            'P_history': P_history
        }
        df = pd.DataFrame(storage)
        
        name = f"N{self.N}_T{self.T}_J{self.J}_h{self.h}_t{self.Delta_t}_P0{self.P_0}_l{self.lambd}.parquet"
        full_path = os.path.join("data",name)
        
        df.to_parquet(full_path)
        
    def run(self):
        m_history, r_history, P_history = self.get_price_history()
        self.export(m_history, r_history, P_history)