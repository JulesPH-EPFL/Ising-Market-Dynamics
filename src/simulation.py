from .model import IsingLattice
import numpy as np
import pandas as pd
import os

class MarketSimulator:
    def __init__(self, N, T, J, h_0, Delta_t, P_0, lambd_0, alpha_neg=2.5, alpha_pos=2.0, p=2, rho=0.9, sigma_h=0.05):
        self.N = N
        self.T = T
        self.J = J
        self.h = h_0
        self.Delta_t = Delta_t
        self.P_0 = P_0 
        self.lambd_0 = lambd_0 
        self.alpha_neg = alpha_neg
        self.alpha_pos = alpha_pos
        self.p = p
        # Initialisation du réseau
        self.lattice = IsingLattice(N, T, J, Delta_t, h_0, rho, sigma_h)
        
    def get_price_history(self):
        M_total = self.lattice.run()
        M_total = np.insert(M_total, 0, 0)
        m_avg = M_total / self.N

        # Lambda sur |m| — formulation originale, elle était correcte
        lambd_history = (
            self.lambd_0 
            * np.exp(-self.alpha_neg * np.maximum(-m_avg, 0)**self.p 
                    - self.alpha_pos * np.maximum(m_avg, 0)**self.p)
        )
        lambd_history = np.clip(lambd_history, 1e-6, None)

        # Rendements bruts
        r_raw = m_avg / lambd_history

        # Centrage par fenêtre glissante — pas de look-ahead
        window = 200
        r_mean_rolling = pd.Series(r_raw).rolling(window, min_periods=1).mean().values
        r_history = r_raw - r_mean_rolling

        # Winsorisation à 5 sigma — coupe les outliers numériques absurdes
        sigma_r = np.std(r_history)
        r_history = np.clip(r_history, -5 * sigma_r, 5 * sigma_r)

        log_P = np.log(self.P_0) + np.cumsum(r_history)
        P_history = np.exp(np.clip(log_P, -300, 300))

        return m_avg, r_history, P_history
    
    def export(self, m_history, r_history, P_history):
        # Création du dossier data s'il n'existe pas
        os.makedirs("data", exist_ok=True)
        
        time = range(len(m_history))
        storage = {
            'time': time,
            'm_history': m_history,
            'r_history': r_history,
            'P_history': P_history
        }
        df = pd.DataFrame(storage)
        
        # AJOUT de alpha et p dans le nom pour ne pas écraser tes tests !
        name = (f"data/N{self.N}_T{self.T:.2f}_J{self.J}_h{self.h}_"
                f"t{self.Delta_t}_l{self.lambd_0}_an{self.alpha_neg}_ap{self.alpha_pos}_p{self.p}.parquet")
        
        df.to_parquet(name)
        return name
        
    def run(self):
        m_history, r_history, P_history = self.get_price_history()
        name = self.export(m_history, r_history, P_history)
        return name
        