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
        # 1. Récupération de la magnétisation totale (Somme des spins)
        M_total = self.lattice.run()
        M_total = np.insert(M_total, 0, 0)
        
        # 2. NORMALISATION : On passe à la magnétisation moyenne m dans [-1, 1]
        # C'est l'étape CRUCIALE pour éviter l'overflow
        m_avg = M_total / self.N
        
        # 3. Calcul de la liquidité dynamique
        # On ajoute 1e-10 pour éviter une division par zéro si la liquidité s'effondre trop
        lambd_history = self.lambd_0 * np.exp(- self.alpha_neg * np.maximum(0,-m_avg) ** self.p 
                                              - self.alpha_pos * np.maximum(0,m_avg) ** self.p)
        lambd_history = np.clip(lambd_history, 1e-10, None) 
        
        # 4. Calcul des rendements r_t
        r_history = m_avg / lambd_history
        
        # 5. Calcul du prix via les Log-Prix pour éviter l'overflow
        # ln(P_t) = ln(P_0) + sum(rendements)
        log_P_0 = np.log(self.P_0)
        log_P_history = log_P_0 + np.cumsum(r_history)
        
        # On ne repasse en exponentielle que pour l'export, 
        # en limitant les valeurs pour éviter le crash "inf"
        P_history = np.exp(np.clip(log_P_history, -700, 700)) 
        
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
        