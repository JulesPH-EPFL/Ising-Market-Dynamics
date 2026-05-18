import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

class MarketMetrics:
    
    def __init__(self, data):
        if isinstance(data, str):
            if data.endswith('.parquet'):
                self.df = pd.read_parquet(data)
            elif data.endswith('.csv'):
                self.df = pd.read_csv(data)
            else:
                raise ValueError("Format de fichier non supporté. Utilisez .csv ou .parquet")
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise TypeError("Fournissez un DataFrame ou un chemin de fichier valide.")
            
        required_cols = ['r_history', 'm_history']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"Attention : La colonne {col} est manquante dans les données.")

    # ==========================================
    # 1. METRICS OF STATISTICAL PHYSICS
    # ==========================================

    def calculate_susceptibility(self, N, T):
        """
        Calcule la susceptibilité magnétique (chi) pour vérifier la transition de phase.
        Formula : chi = (N / T) * ( <m^2> - <m>^2 )
        
        :param N: Nombre total de spins (agents)
        :param T: Température de la simulation
        :param burn_in: Nombre de pas initiaux à ignorer (pour atteindre l'équilibre)
        :return: Valeur de la susceptibilité
        """
            
        m_eq = self.df['m_history']
        
        mean_m_sq = np.mean(m_eq**2)
        sq_mean_m = (np.mean(m_eq))**2
        
        chi = (N / T) * (mean_m_sq - sq_mean_m)
        return chi

    # ==========================================
    # 2. MÉTRIQUES FINANCIÈRES (Faits Stylisés)
    # ==========================================

    def analyze_returns(self):
        """
        Calcule les statistiques descriptives des rendements (Kurtosis, Skewness, test JB).
        """
        r_eq = self.df['r_history']
        
        # Statistiques de base
        volatility = r_eq.std()
        skew = r_eq.skew()
        kurt = r_eq.kurtosis() # Si > 0, on a des queues épaisses (Fat Tails)
        
        # Test de Jarque-Bera (Test de normalité)
        jb_stat, jb_p_value = stats.jarque_bera(r_eq)
        is_normal = jb_p_value > 0.05
        
        results = {
            "Volatilité (Ecart-type)": volatility,
            "Asymétrie (Skewness)": skew,
            "Excès de Kurtosis": kurt,
            "Jarque-Bera p-value": jb_p_value,
            "Est une Loi Normale ?": is_normal
        }
        
        return results

    # ==========================================
    # 3. VISUALISATIONS
    # ==========================================

    def plot_stylized_facts(self, lags=50):
        """
        Génère un tableau de bord graphique prouvant le réalisme du marché.
        - Distribution des rendements (Queues épaisses)
        - Autocorrélation des rendements absolus (Volatility Clustering)
        """
        r_eq = self.df['r_history']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphique 1 : Histogramme des rendements vs Loi Normale
        ax = axes[0]
        ax.hist(r_eq, bins=60, density=True, alpha=0.6, color='red', edgecolor='black', label='Simulés')
        
        # Superposition d'une courbe normale théorique
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(r_eq), np.std(r_eq))
        ax.plot(x, p, 'k', linewidth=2, label='Loi Normale Théorique')
        
        ax.set_title(f"Distribution des Rendements\n(Kurtosis: {r_eq.kurtosis():.2f})")
        ax.set_xlabel("Rendements ($r_t$)")
        ax.set_ylabel("Densité")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Graphique 2 : Autocorrélation des rendements absolus (Volatility Clustering)
        ax2 = axes[1]
        sm.graphics.tsa.plot_acf(r_eq.abs(), lags=lags, ax=ax2, color='blue', alpha=0.5)
        ax2.set_title("Mémoire de la Volatilité (ACF de $|r_t|$)")
        ax2.set_xlabel("Décalage (Lags)")
        ax2.set_ylabel("Autocorrélation")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# UTILITAIRE EXTERNE (Pour chercher le Tc)
# ==========================================

def plot_susceptibility_curve(results_dict):
    """
    Trace la courbe de susceptibilité en fonction de la température 
    pour localiser empiriquement la transition de phase.
    
    :param results_dict: Dictionnaire {Température: Valeur de Chi}
    """
    T_vals = [float(t) for t in results_dict.keys()]
    chi_vals = list(results_dict.values())
    
    chi_critic = np.argmax(chi_vals)
    T_critic = T_vals[chi_critic]
    
    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, chi_vals, marker='o', linestyle='-', color='purple')
    
    # Ligne verticale pour la théorie d'Onsager
    Tc_theorique = 2.269
    plt.axvline(Tc_theorique, color='blue', linestyle='--', label=f'$T_c$ théorique (~{Tc_theorique})')
    plt.axvline(T_critic, color='red', linestyle='--', label=f'$T_c$ empirique (~{T_critic})')
    
    plt.title("Transition de Phase : Susceptibilité Magnétique")
    plt.xticks([round(t, 3) for t in T_vals])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.xlabel("Température ($T$)")
    plt.ylabel("Susceptibilité ($\chi$)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()
    
    return T_critic