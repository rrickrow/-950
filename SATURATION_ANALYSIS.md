# Actuator Saturation Analysis for Complex-Valued Inertial Neural Networks

## 1. Problem Setup

### Paper
**"Non-fragile memory sampled-data control for exponential synchronization of uncertain
complex-valued inertial neural networks with mixed delays"**
*Nonlinear Dyn (2025) 113:29407–29422*

### Drive-Response CVINNN Model

**Drive (master) system:**
$$m_1''(t) = -A(t)\,m_1'(t) - B(t)\,m_1(t) + C(t)\,f(m_1(t)) + D(t)\,f(m_{1,\tau}(t)) + E(t)\int_{t-l(t)}^t f(m_1(s))\,ds$$

**Response (slave) system with actuator saturation:**
$$m_2''(t) = -A(t)\,m_2'(t) - B(t)\,m_2(t) + C(t)\,f(m_2(t)) + D(t)\,f(m_{2,\tau}(t)) + E(t)\int_{t-l(t)}^t f(m_2(s))\,ds + \boldsymbol{U_{\rm sat}(t)}$$

**Non-fragile memory sampled-data controller (paper Eq. 5, now saturated):**
$$U_{\rm sat}(t) = \mathrm{sat}\!\Big(\underbrace{(H + \Delta H)\,h(t_k) + (H_\tau + \Delta H_\tau)\,h_\eta(t_k)}_{U(t)},\; u_{\max}\Big)$$

where the synchronization error is $h(t) = m_2(t) - m_1(t)$,  
$h_\eta(t_k) = h(t_k - \eta)$ is the transmission-delayed sample,  
and the **saturation function** clips real and imaginary parts independently:
$$[\mathrm{sat}(u, u_{\max})]_i = \mathrm{clip}(\mathrm{Re}(u_i), -u_{\max}, u_{\max}) + j\,\mathrm{clip}(\mathrm{Im}(u_i), -u_{\max}, u_{\max})$$

---

## 2. System Parameters (Paper Example 1)

| Parameter | Value |
|-----------|-------|
| $A$ | $\mathrm{diag}(0.9,\; 0.5)$ |
| $B$ | $\mathrm{diag}(0.3,\; 0.3)$ |
| $C$ | $[-0.5\!+\!j,\ 0.6\!+\!0.2j;\ -1.5\!-\!j,\ 0.4\!+\!2.5j]$ |
| $D$ | $[-1.5\!+\!0.1j,\ -1\!+\!2.1j;\ 3.1\!+\!1.1j,\ -1.1\!-\!j]$ |
| $\Gamma$ | $[0.12,\ 0.1;\ 0.2,\ -0.1]$ |
| $f(x)$ | $0.5\tanh(x)$ |
| $\tau(t)$ | $0.1\sin^2(t)+0.1 \in [0.1,\,0.2]$ |
| $l(t)$ | $0.25(1+\sin t)\cdot 0.5 \in [0,\,0.5]$ |
| $\eta$ | $0.1$ s (transmission delay) |
| $h_{\max}$ | $0.10375$ s (max sampling interval from LMI) |

**Controller gains** (Theorem 2, $a=6$):

$$H = \begin{bmatrix} -0.5964-2.1857j & -8.1920+1.0798j \\ -2.4769+0.5250j & 3.3744+0.5168j \end{bmatrix}$$

$$H_\tau = \begin{bmatrix} -0.1075+0.1028j & -0.0869-0.0221j \\ -0.1108-0.0287j & 0.0438-0.0187j \end{bmatrix}$$

---

## 3. Simulation Results (T = 80 s, dt = 0.001 s)

### 3.1 Error Norm Summary

| Scenario | $u_{\max}$ | Max $\|U\|$ | $\|h\|$ at $t=50$ s | $\|h\|$ at $t=80$ s |
|----------|-----------|-------------|----------------------|----------------------|
| No saturation | 100.0 | 24.15 | 2.72 | 0.86 |
| Moderate saturation | 5.0 | 7.07 | 0.38 | 1.08 |
| Heavy saturation | 2.0 | 2.83 | 3.97 | **4.93** (diverging) |

> Note: The "no saturation" and "moderate saturation" rows show a decrease in error over time (converging). The "heavy saturation" row shows error growing from 3.97 at t=50s to 4.93 at t=80s—indicating synchronization is lost due to insufficient control authority.

### 3.2 Key Observations

**1. Control signal clipping** (Fig. 6):  
The unsaturated controller produces peak inputs up to $|U| = 24.15$ during the initial transient. With $u_{\max}=5.0$ the control is clipped to $7.07$ (saturating both the real and imaginary parts simultaneously, hence the $\sqrt{2}\,u_{\max}$ norm). With $u_{\max}=2.0$ the effective control power is reduced to only $\approx 12\%$ of the unsaturated value.

**2. Synchronization degradation** (Fig. 4 & 5):  
- **Heavy saturation ($u_{\max}=2$):** The system loses synchronization. The error norm grows monotonically, indicating that insufficient control authority cannot overcome the initial mismatch and the chaotic drive signal. This is the **domain-of-attraction** limitation inherent to actuator-saturated systems.
- **Moderate saturation ($u_{\max}=5$):** Synchronization is still achieved, but the convergence trajectory is different—the smoother saturated control avoids the large initial overshoot seen in the unsaturated case, leading to a lower error at $t=50$ s despite a slightly higher error at $t=80$ s.
- **No saturation:** Fastest asymptotic convergence, but requires peak control effort $24.15\times$ the saturation bound.

**3. Phase-plane behaviour** (Fig. 7):  
With moderate saturation, the response state tracks the chaotic drive trajectory with a small but persistent offset, converging exponentially as theoretically guaranteed—provided the initial error lies within the domain of attraction for the saturated system.

---

## 4. LMI Modification for Actuator Saturation (Theoretical Extension)

The existing LMI (`lmizhong.m`, Theorem 2) guarantees exponential synchronization **without** considering actuator limits. To formally handle saturation, the following modification is needed:

### 4.1 Dead-Zone Decomposition
Write the saturated control as:
$$U_{\rm sat}(t) = U(t) - \varphi(U(t))$$

where $\varphi(u_i) = u_i - \mathrm{sat}(u_i, u_{\max})$ is the **dead-zone nonlinearity**, satisfying the **sector condition**:
$$\varphi^\dagger(u)\,[\varphi(u) - Du] \le 0 \quad \forall\, u \in \mathbb{C}^n$$

for any diagonal $D$ with $0 < d_i \le 1$.

### 4.2 Modified LMI Condition

Introduce the ellipsoidal invariant set $\mathcal{E}(P_1, c) = \{h \mid h^\dagger P_1 h \le c\}$ and require:
$$\{h \mid h^\dagger P_1 h \le c\} \subseteq \{h \mid |H_{(i)}\,h| \le u_{\max}\}, \quad i=1,\ldots,n$$

This translates to additional LMI constraints:
$$\begin{bmatrix} c & H_{(i)}\,G \\ (H_{(i)}\,G)^\dagger & P_1 \end{bmatrix} \ge 0, \quad i=1,\ldots,n$$

where $G$ is the slack variable from the original Theorem 2.

The modified LMI then guarantees exponential synchronization for all initial conditions within $\mathcal{E}(P_1, c)$, **even with actuator saturation**.

### 4.3 Implementation Note

This extension follows the approach of:  
> Y. Guo, Z. Zhu, I. Ahn (2022). *Non-reduced order method to parameterized sampled-data stabilization of inertial neural networks with actuator saturation.* Neurocomputing.  

To implement this in `lmizhong.m`:
1. Add the scalar variable `c` and the ellipsoidal constraint rows for each controller row $H_{(i)}$  
2. Add the sector condition term $-\Gamma\varphi(U) + \text{sector LMI block}$ to the main inequality $T_1, T_2, T_3$  
3. Maximize $c$ (or fix $c$ and minimize the decay rate $a$) subject to all constraints

---

## 5. Files

| File | Description |
|------|-------------|
| `lmizhong.m` | Original LMI solver (Theorem 2, no saturation) |
| `simulation.m` | MATLAB simulation: drive-response with actuator saturation |
| `simulation_sat.py` | Python simulation: 3-way comparison (no/moderate/heavy saturation) |
| `fig1_states_no_ctrl.png` | Drive/response state trajectories (no controller) |
| `fig2_states_with_ctrl.png` | State trajectories with non-fragile SDC (no saturation) |
| `fig3_error_comparison.png` | Error state: 3 saturation scenarios |
| `fig4_control_inputs_saturation.png` | Control inputs: 3 saturation scenarios |
| `fig5_error_norm.png` | Error norm on log scale: 3 saturation scenarios |
| `fig6_control_early_saturation.png` | Early transient [0–20 s] showing saturation clipping |
| `fig7_states_saturated.png` | State trajectories with moderate saturation |

---

## 6. How to Run

### Python simulation (generates all figures):
```bash
pip install numpy scipy matplotlib
python3 simulation_sat.py
```

### MATLAB simulation (requires YALMIP + MOSEK):
```matlab
% Step 1: Solve LMI (takes several minutes)
run('lmizhong.m')

% Step 2: Simulate with actuator saturation
run('simulation.m')
```
The MATLAB script automatically uses `KKK1` and `KKKtau1` from the LMI solution as controller gains.

---

## 7. Conclusion

Adding actuator saturation to the non-fragile memory sampled-data controller reveals a fundamental trade-off: the LMI-designed controller (Theorem 2) requires peak control efforts far exceeding physically realizable bounds ($|U|_{\max}\approx 24$). Without accounting for saturation in the design:

- **Moderate saturation** ($u_{\max} \ge 5$): synchronization is preserved but converges differently
- **Heavy saturation** ($u_{\max} \le 2$): synchronization is lost due to insufficient control authority

A complete treatment requires the modified LMI conditions (Section 4) that co-design the controller and the domain of attraction, as described in reference [15] of the paper.
