"""
Simulation with Actuator Saturation for Complex-Valued Inertial Neural Networks (CVINNs)

Based on the paper:
  "Non-fragile memory sampled-data control for exponential synchronization of
   uncertain complex-valued inertial neural networks with mixed delays"
  Nonlinear Dyn (2025) 113:29407–29422

This script demonstrates the drive-response synchronization system with the
non-fragile memory sampled-data controller, and adds actuator saturation to
show its effects on synchronization performance and control inputs.

System (error dynamics, Eq. 7 in paper):
  h''(t) = -A(t)*h'(t) - B(t)*h(t) + C(t)*F(h(t)) + D(t)*F(h_tau(t))
           + E(t)*∫_{t-l(t)}^t F(h(s))ds + Gamma*(H+DH)*h(tk) + Gamma*(Htau+DHtau)*h_eta(tk)

Controller with saturation (added to original Eq. 5):
  U(t) = sat( (H+DH)*h(tk) + (Htau+DHtau)*h_eta(tk) , u_max )

where sat() clips each component's real and imaginary parts to [-u_max, u_max].
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── System parameters (paper Example 1) ────────────────────────────────────
n = 2
A  = np.diag([0.9, 0.5])                                        # inertia/damping
B  = np.diag([0.3, 0.3])                                        # system matrix
C  = np.array([[-0.5+1j,  0.6+0.2j], [-1.5-1j,  0.4+2.5j]])   # recurrent weights
D  = np.array([[-1.5+0.1j, -1+2.1j], [3.1+1.1j, -1.1-1j]])    # delay weights
E  = np.array([[-0.1+1j,  0.35-1j ], [0.3+1.1j,  1.1-1j ]])   # distributed delay weights
Gm = np.array([[0.12, 0.1], [0.2, -0.1]])                       # Gamma matrix

# Activation function and Lipschitz constant
def f_act(x):
    return 0.5 * np.tanh(x)

# Uncertainty structure (paper Example 1)
M1 = np.eye(n);  N1 = 0.2*np.eye(n);  N2 = 0.2*np.eye(n)
M2 = np.eye(n);  N3 = 0.2*np.eye(n);  N4 = 0.2*np.eye(n);  N5 = 0.2*np.eye(n)
K_c = 0.1*np.eye(n);  N_c = 0.5*np.eye(n);  Ntau_c = 0.5*np.eye(n)

def S_func(t):   return np.sin(2*t)      # system uncertainty signal
def Th_func(t):  return np.sin(t)         # controller uncertainty signal

# ─── Controller gains (from paper Example 1, Theorem 2 solution) ────────────
H_gain    = np.array([[-0.5964-2.1857j, -8.1920+1.0798j],
                      [-2.4769+0.5250j,  3.3744+0.5168j]])
Htau_gain = np.array([[-0.1075+0.1028j, -0.0869-0.0221j],
                      [-0.1108-0.0287j,  0.0438-0.0187j]])

# ─── Delay functions ─────────────────────────────────────────────────────────
def tau_tv(t):  return 0.1*np.sin(t)**2 + 0.1      # time-varying delay ∈ [0.1, 0.2]
def l_tv(t):    return 0.25*(1.0 + np.sin(t)) * 0.5   # distributed delay ∈ [0, 0.5]

TAU_MAX = 0.2     # max of tau(t)
L_MAX   = 0.5     # max of l(t) = L
ETA     = 0.1     # transmission delay in controller

# ─── Simulation settings ─────────────────────────────────────────────────────
dt        = 0.001                          # time step (s)
T_END     = 80.0                           # simulation end time (s)
H_MAX     = 0.10375                        # maximum sampling interval from LMI
U_MAX_NO  = 100.0                          # effectively no saturation
U_MAX_SAT = 2.0                            # actuator saturation bound

t_arr  = np.arange(0.0, T_END + dt, dt)
N_STEP = len(t_arr)

# History buffer depth (samples needed for delays)
BUF   = int(np.ceil(max(TAU_MAX, L_MAX, ETA) / dt)) + 20

# ─── Actuator saturation function ────────────────────────────────────────────
def sat(u_vec, u_max):
    """Component-wise saturation: clip Re and Im parts of each element."""
    re = np.clip(np.real(u_vec), -u_max, u_max)
    im = np.clip(np.imag(u_vec), -u_max, u_max)
    return re + 1j * im

# ─── System RHS helpers ───────────────────────────────────────────────────────
def system_uncertainty(t):
    """Uncertain system matrices: DeltaA, DeltaB, DeltaC, DeltaD, DeltaE"""
    s = S_func(t)
    dA = M1 @ (s * N1);  dB = M1 @ (s * N2)
    dC = M2 @ (s * N3);  dD = M2 @ (s * N4);  dE = M2 @ (s * N5)
    return dA, dB, dC, dD, dE

def ctrl_uncertainty(t):
    """Uncertain controller gain perturbations: DeltaH, DeltaHtau"""
    th = Th_func(t)
    dH    = K_c @ (th * N_c)
    dHtau = K_c @ (th * Ntau_c)
    return dH, dHtau

def drive_rhs(t, m1, m1d, m1_tau, m1_dist):
    """RHS for drive (master) system."""
    dA, dB, dC, dD, dE = system_uncertainty(t)
    A_t = A + dA;  B_t = B + dB;  C_t = C + dC;  D_t = D + dD;  E_t = E + dE
    Fm1     = f_act(m1)
    Fm1_tau = f_act(m1_tau)
    ddm1 = -(A_t @ m1d) - (B_t @ m1) + (C_t @ Fm1) + (D_t @ Fm1_tau) + (E_t @ m1_dist)
    return m1d, ddm1

def response_rhs(t, m2, m2d, m2_tau, m2_dist, U_in):
    """RHS for response (slave) system with control input U_in."""
    dA, dB, dC, dD, dE = system_uncertainty(t)
    A_t = A + dA;  B_t = B + dB;  C_t = C + dC;  D_t = D + dD;  E_t = E + dE
    Fm2     = f_act(m2)
    Fm2_tau = f_act(m2_tau)
    ddm2 = -(A_t @ m2d) - (B_t @ m2) + (C_t @ Fm2) + (D_t @ Fm2_tau) \
           + (E_t @ m2_dist) + (Gm @ U_in)
    return m2d, ddm2

def get_delayed(hist, i, delay, dt):
    """Return history value at time index i delayed by 'delay' seconds."""
    lag = int(np.round(delay / dt))
    j   = max(i - lag, 0)
    return hist[j]

def get_dist_integral(hist_f, i, l_val, dt):
    """Approximate ∫_{t-l(t)}^t f(m(s)) ds using trapezoidal rule."""
    steps = max(1, int(np.round(l_val / dt)))
    j0    = max(i - steps, 0)
    vals  = hist_f[j0:i+1]               # shape: (steps+1, n)
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    return trapz(vals, dx=dt, axis=0) if len(vals) > 1 else vals[0] * dt

# ─── Initial conditions (paper Example 1) ────────────────────────────────────
m1_0    = np.array([ 0.1 - 0.5j, -0.1 + 0.2j])   # drive position
m1d_0   = np.array([ 0.2 - 0.1j, -0.15 + 0.2j])  # drive velocity
m2_0    = np.array([-0.1 + 0.12j, 0.2 + 0.5j])    # response position
m2d_0   = np.array([-0.1 - 0.1j,  0.1 + 0.2j])   # response velocity

# ─── Run one simulation pass ──────────────────────────────────────────────────
def run_simulation(u_max, label):
    """
    Simulate the drive-response CVINNN system with the non-fragile memory
    sampled-data controller and given actuator saturation bound u_max.
    Returns arrays of error state, control input, and both system states.
    """
    print(f"  Running simulation: {label} (u_max = {u_max})")

    # Storage
    m1_h  = np.zeros((N_STEP, n), dtype=complex)   # drive position
    m1d_h = np.zeros((N_STEP, n), dtype=complex)   # drive velocity
    m2_h  = np.zeros((N_STEP, n), dtype=complex)   # response position
    m2d_h = np.zeros((N_STEP, n), dtype=complex)   # response velocity
    fm1_h = np.zeros((N_STEP, n), dtype=complex)   # f(drive pos) history
    fm2_h = np.zeros((N_STEP, n), dtype=complex)
    U_h   = np.zeros((N_STEP, n), dtype=complex)   # control input

    # Set initial conditions
    m1_h[0]  = m1_0;    m1d_h[0] = m1d_0
    m2_h[0]  = m2_0;    m2d_h[0] = m2d_0
    fm1_h[0] = f_act(m1_0)
    fm2_h[0] = f_act(m2_0)

    # Sampler state (ZOH)
    np.random.seed(42)
    next_samp = 0.0
    h_tk      = np.zeros(n, dtype=complex)     # h(tk)   sampled error
    h_eta_tk  = np.zeros(n, dtype=complex)     # h(tk-η) sampled error

    for i in range(N_STEP - 1):
        t = t_arr[i]

        # ── Sampling logic (ZOH with aperiodic intervals in (0, H_MAX]) ──
        if t >= next_samp - 1e-10:
            hk_now     = m2_h[i] - m1_h[i]          # current error h(tk)
            # h_eta(tk) = h(tk - eta)
            eta_lag    = int(np.round(ETA / dt))
            j_eta      = max(i - eta_lag, 0)
            hk_eta_now = m2_h[j_eta] - m1_h[j_eta]  # h(tk - eta)
            h_tk       = hk_now
            h_eta_tk   = hk_eta_now
            # random aperiodic interval in (0, H_MAX]
            h_k = H_MAX * (0.5 + 0.5 * np.random.rand())
            next_samp  = t + h_k

        # ── Compute control input ────────────────────────────────────────
        dH, dHtau  = ctrl_uncertainty(t)
        U_nominal  = (H_gain + dH) @ h_tk + (Htau_gain + dHtau) @ h_eta_tk
        U_applied  = sat(U_nominal, u_max)
        U_h[i]     = U_applied

        # ── Get delay terms for drive system ────────────────────────────
        tau_val  = tau_tv(t)
        l_val    = l_tv(t)
        m1_tau   = get_delayed(m1_h,  i, tau_val, dt)
        m1_dist  = get_dist_integral(fm1_h, i, l_val, dt)

        # ── Get delay terms for response system ─────────────────────────
        m2_tau  = get_delayed(m2_h,  i, tau_val, dt)
        m2_dist = get_dist_integral(fm2_h, i, l_val, dt)

        # ── Euler integration: drive system ─────────────────────────────
        m1d_new, ddm1 = drive_rhs(t, m1_h[i], m1d_h[i], m1_tau, m1_dist)
        m1_h[i+1]  = m1_h[i]  + dt * m1d_new
        m1d_h[i+1] = m1d_h[i] + dt * ddm1

        # ── Euler integration: response system ──────────────────────────
        m2d_new, ddm2 = response_rhs(t, m2_h[i], m2d_h[i], m2_tau, m2_dist, U_applied)
        m2_h[i+1]  = m2_h[i]  + dt * m2d_new
        m2d_h[i+1] = m2d_h[i] + dt * ddm2

        # Update activation function history
        fm1_h[i+1] = f_act(m1_h[i+1])
        fm2_h[i+1] = f_act(m2_h[i+1])

    # Last step control (copy previous)
    U_h[-1] = U_h[-2]

    err = m2_h - m1_h    # synchronization error h(t)
    return err, U_h, m1_h, m1d_h, m2_h, m2d_h

# Three saturation cases for comprehensive comparison
U_MAX_MED = 5.0   # moderate saturation

# ─── Run three cases ──────────────────────────────────────────────────────────
print("Starting simulations …")
err_no,  U_no,  m1_no,  m1d_no,  m2_no,  m2d_no   = run_simulation(U_MAX_NO,  "Without saturation")
err_med, U_med, m1_med, m1d_med, m2_med, m2d_med   = run_simulation(U_MAX_MED, f"Moderate saturation (u_max={U_MAX_MED})")
err_sat, U_sat, m1_sat, m1d_sat, m2_sat, m2d_sat   = run_simulation(U_MAX_SAT, f"Heavy saturation (u_max={U_MAX_SAT})")
print("Simulations complete.")

# ─── Helper: error norm ───────────────────────────────────────────────────────
def err_norm(err):
    return np.sqrt(np.sum(np.abs(err)**2, axis=1))

# ─── Plot 1: State trajectories without controller (drive only) ───────────────
print("Generating plots …")

fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
fig1.suptitle("Fig. 2 – Trajectories of states without controller\n(drive system, chaotic behaviour)",
              fontsize=12, fontweight='bold')
labels_state = [r"$\mathrm{Re}(m_{1,1})$", r"$\mathrm{Im}(m_{1,1})$",
                r"$\mathrm{Re}(m_{1,2})$", r"$\mathrm{Im}(m_{1,2})$"]
state_funcs  = [lambda m: np.real(m[:, 0]), lambda m: np.imag(m[:, 0]),
                lambda m: np.real(m[:, 1]), lambda m: np.imag(m[:, 1])]
for ax, lbl, sfn in zip(axes.flat, labels_state, state_funcs):
    # Run a short simulation without controller
    ax.plot(t_arr[:40001], sfn(m1_no)[:40001], 'b',  lw=0.6, label='drive')
    ax.plot(t_arr[:40001], sfn(m2_no)[:40001], 'r--', lw=0.6, label='response (no ctrl)')
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, 40);  ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig("fig1_states_no_ctrl.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

# ─── Plot 2: State trajectories with controller (no saturation) ──────────────
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle("Fig. 3 – Trajectories of states with non-fragile SDC (no saturation)",
              fontsize=12, fontweight='bold')
for ax, lbl, sfn in zip(axes.flat, labels_state, state_funcs):
    ax.plot(t_arr, sfn(m1_no), 'b',  lw=0.7, label='drive')
    ax.plot(t_arr, sfn(m2_no), 'r--', lw=0.7, label='response')
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, T_END);  ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig("fig2_states_with_ctrl.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

# ─── Plot 3: Error state under controller (three cases) ─────────────────────
fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle(f"Fig. 4 – Synchronisation error state\n(no saturation | moderate u_max={U_MAX_MED} | heavy u_max={U_MAX_SAT})",
              fontsize=12, fontweight='bold')
err_labels = [r"$\mathrm{Re}(h_1)$", r"$\mathrm{Im}(h_1)$",
              r"$\mathrm{Re}(h_2)$", r"$\mathrm{Im}(h_2)$"]
err_funcs  = [lambda e: np.real(e[:, 0]), lambda e: np.imag(e[:, 0]),
              lambda e: np.real(e[:, 1]), lambda e: np.imag(e[:, 1])]
for ax, lbl, efn in zip(axes.flat, err_labels, err_funcs):
    ax.plot(t_arr, efn(err_no),  'b',  lw=0.8, label='no saturation')
    ax.plot(t_arr, efn(err_med), 'g-.', lw=0.8, label=f'moderate sat (u_max={U_MAX_MED})')
    ax.plot(t_arr, efn(err_sat), 'r--', lw=0.8, label=f'heavy sat (u_max={U_MAX_SAT})')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, T_END);  ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig("fig3_error_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

# ─── Plot 4: Control inputs (showing saturation effect) ──────────────────────
fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
fig4.suptitle(f"Fig. 5 – Control inputs: no sat | moderate (u_max={U_MAX_MED}) | heavy (u_max={U_MAX_SAT})",
              fontsize=12, fontweight='bold')
u_labels = [r"$\mathrm{Re}(U_1)$", r"$\mathrm{Im}(U_1)$",
            r"$\mathrm{Re}(U_2)$", r"$\mathrm{Im}(U_2)$"]
u_funcs  = [lambda u: np.real(u[:, 0]), lambda u: np.imag(u[:, 0]),
            lambda u: np.real(u[:, 1]), lambda u: np.imag(u[:, 1])]
for ax, lbl, ufn in zip(axes.flat, u_labels, u_funcs):
    ax.plot(t_arr, ufn(U_no),  'b',  lw=0.8, label='no saturation')
    ax.plot(t_arr, ufn(U_med), 'g-.', lw=0.9, label=f'moderate (±{U_MAX_MED})', alpha=0.85)
    ax.plot(t_arr, ufn(U_sat), 'r--', lw=0.9, label=f'heavy (±{U_MAX_SAT})', alpha=0.85)
    ax.axhline( U_MAX_SAT, color='gray', lw=1.0, ls='--', alpha=0.7, label=f'+u_max={U_MAX_SAT}')
    ax.axhline(-U_MAX_SAT, color='gray', lw=1.0, ls='--', alpha=0.7)
    ax.axhline( U_MAX_MED, color='olive', lw=0.8, ls=':', alpha=0.6, label=f'+u_max={U_MAX_MED}')
    ax.axhline(-U_MAX_MED, color='olive', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, T_END);  ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig("fig4_control_inputs_saturation.png", dpi=150, bbox_inches='tight')
plt.close(fig4)

# ─── Plot 5: Error norm comparison ───────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(10, 5))
en_no  = err_norm(err_no)
en_med = err_norm(err_med)
en_sat = err_norm(err_sat)
ax.semilogy(t_arr, en_no,  'b',   lw=1.2, label='No saturation')
ax.semilogy(t_arr, en_med, 'g-.', lw=1.2, label=f'Moderate saturation (u_max={U_MAX_MED})')
ax.semilogy(t_arr, en_sat, 'r--', lw=1.2, label=f'Heavy saturation (u_max={U_MAX_SAT})')
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel(r'$\|h(t)\|$ (log scale)', fontsize=12)
ax.set_title('Synchronisation error norm: effect of actuator saturation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, T_END)
plt.tight_layout()
fig5.savefig("fig5_error_norm.png", dpi=150, bbox_inches='tight')
plt.close(fig5)

# ─── Plot 6: Control input over early window to show clipping clearly ─────────
fig6, axes = plt.subplots(2, 2, figsize=(12, 8))
fig6.suptitle(f"Fig. 6 – Early transient (t ∈ [0, 20 s]): saturation clipping",
              fontsize=12, fontweight='bold')
idx_20 = int(20.0 / dt)
for ax, lbl, ufn in zip(axes.flat, u_labels, u_funcs):
    ax.plot(t_arr[:idx_20], ufn(U_no)[:idx_20],  'b',   lw=1.0, label='no saturation')
    ax.plot(t_arr[:idx_20], ufn(U_med)[:idx_20], 'g-.', lw=1.2, label=f'moderate (±{U_MAX_MED})', alpha=0.9)
    ax.plot(t_arr[:idx_20], ufn(U_sat)[:idx_20], 'r--', lw=1.2, label=f'heavy (±{U_MAX_SAT})', alpha=0.9)
    ax.axhline( U_MAX_SAT, color='gray', lw=1.2, ls='--', alpha=0.8, label=f'+u_max={U_MAX_SAT}')
    ax.axhline(-U_MAX_SAT, color='gray', lw=1.2, ls='--', alpha=0.8)
    ax.axhline( U_MAX_MED, color='olive', lw=0.8, ls=':', alpha=0.7, label=f'+u_max={U_MAX_MED}')
    ax.axhline(-U_MAX_MED, color='olive', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, 20);  ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig6.savefig("fig6_control_early_saturation.png", dpi=150, bbox_inches='tight')
plt.close(fig6)

# ─── Plot 7: State trajectories with moderate saturation ──────────────────────
fig7, axes = plt.subplots(2, 2, figsize=(12, 8))
fig7.suptitle(f"Fig. 7 – State trajectories with moderate actuator saturation (u_max={U_MAX_MED})",
              fontsize=12, fontweight='bold')
for ax, lbl, sfn in zip(axes.flat, labels_state, state_funcs):
    ax.plot(t_arr, sfn(m1_med), 'b',  lw=0.7, label='drive')
    ax.plot(t_arr, sfn(m2_med), 'r--', lw=0.7, label=f'response (u_max={U_MAX_MED})')
    ax.set_xlabel('Time [s]');  ax.set_ylabel(lbl)
    ax.set_xlim(0, T_END);  ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig7.savefig("fig7_states_saturated.png", dpi=150, bbox_inches='tight')
plt.close(fig7)

# ─── Print summary statistics ────────────────────────────────────────────────
t50  = int(50.0 / dt)
t80  = N_STEP - 1
print("\n══ Simulation Summary ══════════════════════════════════")
print(f"  Sampling interval H_MAX            = {H_MAX}")
print(f"  Moderate saturation bound u_max    = {U_MAX_MED}")
print(f"  Heavy saturation bound u_max       = {U_MAX_SAT}")
print(f"  Max |U| without saturation         = {np.max(np.abs(U_no)):.4f}")
print(f"  Max |U| moderate saturation        = {np.max(np.abs(U_med)):.4f}")
print(f"  Max |U| heavy saturation           = {np.max(np.abs(U_sat)):.4f}")
print(f"  Error norm at t=50 s (no sat)      = {en_no[t50]:.6f}")
print(f"  Error norm at t=50 s (moderate)    = {en_med[t50]:.6f}")
print(f"  Error norm at t=50 s (heavy sat)   = {en_sat[t50]:.6f}")
print(f"  Error norm at t=80 s (no sat)      = {en_no[t80]:.6f}")
print(f"  Error norm at t=80 s (moderate)    = {en_med[t80]:.6f}")
print(f"  Error norm at t=80 s (heavy sat)   = {en_sat[t80]:.6f}")
print("  Output files saved:")
for fn in ["fig1_states_no_ctrl.png", "fig2_states_with_ctrl.png",
           "fig3_error_comparison.png", "fig4_control_inputs_saturation.png",
           "fig5_error_norm.png", "fig6_control_early_saturation.png",
           "fig7_states_saturated.png"]:
    print(f"    {fn}")
print("══════════════════════════════════════════════════════")
