# Ising Model for Financial Market Dynamics: An Econophysics Approach

## Abstract
This project explores the application of the **Ising Model**, a cornerstone of statistical mechanics, to model complex financial systems. By mapping atomic spins to economic agents (Buyers/Sellers), we simulate the emergence of collective behaviors such as speculative bubbles and market crashes. The objective is to demonstrate that **extreme market events** are not necessarily driven by external shocks but can emerge endogenously from internal dynamics and local interactions (herding). We analyze the generated time series to identify "stylized facts" of finance, notably **fat tails** and **volatility clustering**, occurring near the critical phase transition.

---

## Theoretical Framework

### 1. The Microscopic Model: The Hamiltonian
The market is represented by a 2D lattice of size $L \times L$, where each site $i$ is occupied by an agent characterized by a spin state $s_i \in {+1, -1}$:
* $s_i = +1$: Bullish agent (Buyer)
* $s_i = -1$: Bearish agent (Seller)

The configuration energy of the system is governed by the Ising Hamiltonian:

$$H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i$$

Where:
* **$J$ (Coupling Constant):** Represents the interaction strength between nearest neighbors. In a financial context, $J > 0$ models **herding behavior**.
* **$h$ (External Field):** Models the influence of external news or the **fundamental value**.
* **$\langle i,j \rangle$:** Indicates the sum over nearest-neighbor pairs.

---

### 2. Market Dynamics: Metropolis-Hastings
The system evolves via non-equilibrium dynamics using the Metropolis algorithm. At each time step, an agent attempts to flip their state. The flip is accepted with a probability $P$ following the Boltzmann weight:

$$P(s_i \to -s_i) = \min \left( 1, e^{-\beta \Delta H} \right)$$

Here, $\beta = 1/k_B T$ is the inverse temperature. In econophysics, the **temperature $T$** is interpreted as the level of noise or the **irrationality** of agents. High temperatures correspond to efficient markets (random walk), while low temperatures favor macroscopic order (bubbles).

---

### 3. From Magnetization to Price
To bridge physics and finance, we define the **total magnetization** $M_t$ at time $t$ as the aggregate excess demand:

$$M_t = \sum_{i=1}^{N} s_i(t)$$

Using the **Market Impact** hypothesis, we derive the **log-return** $r_t$ and the **price** $P_t$:

$$r_t = \ln(P_t) - \ln(P_{t-1}) \approx \frac{M_t}{\lambda}$$

$$P_t = P_0 \exp \left( \sum_{k=1}^t \frac{M_k}{\lambda} \right)$$

Where $\lambda$ represents the **market liquidity** (depth of the order book).

---

### 4. Statistical Indicators & Criticality
The core of this study lies in the **phase transition** at the critical temperature $T_c$. Near this point, the magnetic susceptibility $\chi$ diverges:

$$\chi = \frac{\partial M}{\partial h} \propto |T - T_c|^{-\gamma}$$

In finance, this divergence explains why a minute fundamental signal ($h$) can trigger a macroscopic price variation during a crash. We measure the non-Gaussianity of returns using **Kurtosis** $K$:

$$K = \frac{\mathbb{E}[(r - \mu)^4]}{\sigma^4}$$

A value of $K > 3$ confirms the presence of **Fat Tails**, a characteristic of real-world markets successfully reproduced by this model at $T \approx T_c$.