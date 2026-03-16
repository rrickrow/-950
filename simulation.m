%% simulation.m
% Simulation of drive-response CVINNN synchronization with Actuator Saturation
%
% Paper: "Non-fragile memory sampled-data control for exponential
%         synchronization of uncertain CVINNs with mixed delays"
%         Nonlinear Dyn (2025) 113:29407-29422
%
% This script:
%   1. Runs (or loads) the LMI solver (lmizhong.m) to obtain controller gains
%      KKK1 (= Gamma*H) and KKKtau1 (= Gamma*Htau)
%   2. Simulates the drive-response CVINNN system
%   3. Compares performance with and without actuator saturation
%   4. Produces publication-quality figures
%
% Actuator saturation model (applied to Eq. 5 in the paper):
%   U_sat(t) = sat(U(t), u_max)
%   where sat(u_i, u_max) clips Re and Im parts of each component to [-u_max, u_max]
%   The response system becomes (modified Eq. 3):
%   m_bar''(t) = -A(t)*m_bar'(t) - B(t)*m_bar(t) + C(t)*F(m_bar(t))
%              + D(t)*F(m_bar_tau(t)) + E(t)*integral_F
%              + U_sat(t)    [effective: Gamma*(H+DH)*h(tk) + ... saturated]
%
% Usage:
%   Run this file after lmizhong.m (KKK1 and KKKtau1 must be in workspace).
%   Or run standalone – the script will call lmizhong.m automatically.

clear; clc; close all;

%% ─── Step 1: Obtain controller gains ────────────────────────────────────────
if ~exist('KKK1', 'var') || ~exist('KKKtau1', 'var')
    fprintf('KKK1/KKKtau1 not found in workspace. Running lmizhong.m ...\n');
    run('lmizhong.m');
    fprintf('lmizhong.m completed. Proceeding to simulation.\n');
end

% Nominal controller gain matrices (Gamma*H and Gamma*Htau from LMI)
H_eff    = KKK1;       % effective gain (includes Gamma coupling)
Htau_eff = KKKtau1;    % effective tau gain

fprintf('\nController gains (KKK1 = Gamma*H):\n');  disp(H_eff);
fprintf('Controller gains (KKKtau1 = Gamma*Htau):\n'); disp(Htau_eff);

%% ─── Step 2: System parameters ──────────────────────────────────────────────
n = 2;

% System matrices (matching lmizhong.m parameters)
% NOTE: These parameters come from lmizhong.m (alpha1, beta1, ksai1, gama1, gama2).
%       The Python simulation_sat.py uses the paper's Example 1 parameters for standalone use.
A_nom = [0.9, 0; 0, 1.2];                              % alpha1 (from lmizhong.m)
B_nom = [2.3, 0; 0, 2.3];                              % beta1  (from lmizhong.m)
C_nom = [-0.5+1i, 0.6+0.2i; 1.5-1i, 1.4+2.5i];        % ksai1  (from lmizhong.m)
D_nom = [-1.5+3.1i, 1+2.1i; 4+1.1i, 1.1-1i];          % gama1  (from lmizhong.m)
E_nom = [-0.1+0.1i, 0.35+1i; 0.3+1.1i, 1.1-1i];       % gama2  (from lmizhong.m)

% Uncertainty structure (matching lmizhong.m: Na1/Na2=0.2, Nk1/Nk2=0.3)
M1_s = eye(n);  N1_s = 0.2*eye(n);  N2_s = 0.2*eye(n);  % system: A, B  (Na1, Na2)
M2_s = eye(n);  N3_s = 0.3*eye(n);  N4_s = 0.3*eye(n);  N5_s = 0.3*eye(n); % system: C, D, E (Nk1, Nk2, Nr)
K_c  = eye(n);  N_c = 0.5*eye(n);   Nc_tau = 0.5*eye(n);% controller (H1, N1, Ntau1)

% Uncertainty signal functions
S_func  = @(t) sin(2*t);   % system uncertainty
Th_func = @(t) sin(t);     % controller uncertainty

% Activation function: f(x) = 0.5*tanh(x)
f_act = @(x) 0.5 * tanh(x);

% Delay parameters
tau_tv = @(t) 0.1*sin(t).^2 + 0.1;    % time-varying delay in [0.1, 0.2]
l_tv   = @(t) 0.25*(1 + sin(t))*0.5;  % distributed delay in [0, 0.5]
L_max  = 0.5;
tau_max = 0.2;
eta    = 0.1;   % transmission delay

%% ─── Step 3: Simulation parameters ──────────────────────────────────────────
dt    = 0.001;        % time step (s)
T_END = 80;           % simulation end time (s)
t_arr = 0 : dt : T_END;
N_STEP = length(t_arr);

% Maximum sampling interval (from LMI bisection)
H_MAX = 0.10375;

% Actuator saturation bounds to compare
U_MAX_NO  = 1000;   % effectively no saturation (baseline)
U_MAX_SAT =  2.0;   % moderate saturation bound

%% ─── Step 4: Initial conditions (paper Example 1) ───────────────────────────
m1_0   = [0.1 - 0.5i; -0.1 + 0.2i];     % drive position
m1d_0  = [0.2 - 0.1i; -0.15 + 0.2i];    % drive velocity
m2_0   = [-0.1 + 0.12i;  0.2 + 0.5i];   % response position
m2d_0  = [-0.1 - 0.1i;   0.1 + 0.2i];   % response velocity

%% ─── Step 5: Define helper functions ─────────────────────────────────────────

% Actuator saturation: component-wise clipping of Re and Im
sat_u = @(u, umax) complex( max(-umax, min(umax, real(u))), ...
                             max(-umax, min(umax, imag(u))) );

% Distributed integral (trapezoidal, using stored f-history)
function val = dist_integral(fhist, i, l_val, dt)
    steps = max(1, round(l_val / dt));
    j0    = max(1, i - steps);
    vals  = fhist(j0:i, :);   % (steps+1) x n
    if size(vals, 1) > 1
        val = trapz(vals) * dt;
    else
        val = vals(1, :) * dt;
    end
end

%% ─── Step 6: Simulation loop ─────────────────────────────────────────────────
fprintf('\nStarting simulations...\n');

for scenario = 1:2
    if scenario == 1
        u_max = U_MAX_NO;
        lbl   = 'without saturation';
    else
        u_max = U_MAX_SAT;
        lbl   = sprintf('with saturation (u_max = %.1f)', U_MAX_SAT);
    end
    fprintf('  Running scenario: %s\n', lbl);

    % Pre-allocate state storage
    m1  = zeros(N_STEP, n, 'like', 1i);  m1d  = zeros(N_STEP, n, 'like', 1i);
    m2  = zeros(N_STEP, n, 'like', 1i);  m2d  = zeros(N_STEP, n, 'like', 1i);
    fm1 = zeros(N_STEP, n, 'like', 1i);  fm2  = zeros(N_STEP, n, 'like', 1i);
    U_ctrl = zeros(N_STEP, n, 'like', 1i);

    % Set initial values
    m1(1,:)   = m1_0.';   m1d(1,:) = m1d_0.';
    m2(1,:)   = m2_0.';   m2d(1,:) = m2d_0.';
    fm1(1,:)  = f_act(m1_0).';
    fm2(1,:)  = f_act(m2_0).';

    % ZOH sampler state
    rng(42);   % fixed seed for reproducibility
    next_samp  = 0;
    h_tk       = zeros(n,1,'like',1i);
    h_eta_tk   = zeros(n,1,'like',1i);

    for i = 1 : N_STEP - 1
        t = t_arr(i);

        %% Sampling logic (aperiodic ZOH)
        if t >= next_samp - 1e-10
            % Sample current error h(tk)
            h_tk = m2(i,:).' - m1(i,:).';

            % h_eta(tk) = error at (tk - eta)
            eta_lag    = round(eta / dt);
            j_eta      = max(1, i - eta_lag);
            h_eta_tk   = m2(j_eta,:).' - m1(j_eta,:).';

            % Random aperiodic interval in (0, H_MAX]
            h_k       = H_MAX * (0.5 + 0.5*rand());
            next_samp = t + h_k;
        end

        %% Compute controller with non-fragile uncertainty
        theta_t = Th_func(t);
        DeltaH_now    = K_c * (theta_t * N_c);
        DeltaHtau_now = K_c * (theta_t * Nc_tau);

        U_nom = (H_eff + DeltaH_now) * h_tk + (Htau_eff + DeltaHtau_now) * h_eta_tk;
        U_now = sat_u(U_nom, u_max);
        U_ctrl(i,:) = U_now.';

        %% System uncertainty
        s_t  = S_func(t);
        dA = M1_s * (s_t * N1_s);
        dB = M1_s * (s_t * N2_s);
        dC = M2_s * (s_t * N3_s);
        dD = M2_s * (s_t * N4_s);
        dE = M2_s * (s_t * N5_s);   % separate N5_s for E uncertainty

        A_t = A_nom + dA;  B_t = B_nom + dB;
        C_t = C_nom + dC;  D_t = D_nom + dD;  E_t = E_nom + dE;

        %% Get delayed terms for drive system
        tau_val = tau_tv(t);
        l_val   = l_tv(t);
        tau_lag = round(tau_val / dt);
        j_tau   = max(1, i - tau_lag);

        m1_tau = m1(j_tau,:).';
        m1_dist = dist_integral(fm1, i, l_val, dt).';

        %% Get delayed terms for response system
        m2_tau  = m2(j_tau,:).';
        m2_dist = dist_integral(fm2, i, l_val, dt).';

        %% Euler integration: drive system
        Fm1    = f_act(m1(i,:).');
        Fm1tau = f_act(m1_tau);
        ddm1 = -(A_t * m1d(i,:).') - (B_t * m1(i,:).') ...
               + (C_t * Fm1) + (D_t * Fm1tau) + (E_t * m1_dist);
        m1(i+1,:)  = m1(i,:)  + dt * m1d(i,:);
        m1d(i+1,:) = m1d(i,:) + dt * ddm1.';

        %% Euler integration: response system (with saturated control)
        Fm2    = f_act(m2(i,:).');
        Fm2tau = f_act(m2_tau);
        ddm2 = -(A_t * m2d(i,:).') - (B_t * m2(i,:).') ...
               + (C_t * Fm2) + (D_t * Fm2tau) + (E_t * m2_dist) + U_now;
        m2(i+1,:)  = m2(i,:)  + dt * m2d(i,:);
        m2d(i+1,:) = m2d(i,:) + dt * ddm2.';

        fm1(i+1,:) = f_act(m1(i+1,:).');
        fm2(i+1,:) = f_act(m2(i+1,:).');
    end
    U_ctrl(end,:) = U_ctrl(end-1,:);

    % Store results
    if scenario == 1
        m1_no = m1;  m1d_no = m1d;  m2_no = m2;  m2d_no = m2d;  U_no = U_ctrl;
    else
        m1_sat = m1; m1d_sat = m1d; m2_sat = m2; m2d_sat = m2d; U_sat = U_ctrl;
    end
end

%% ─── Step 7: Error states ────────────────────────────────────────────────────
err_no  = m2_no  - m1_no;
err_sat = m2_sat - m1_sat;
err_norm_no  = sqrt(sum(abs(err_no).^2,  2));
err_norm_sat = sqrt(sum(abs(err_sat).^2, 2));

fprintf('\n══ Simulation Summary ══════════════════════════════════\n');
fprintf('  Max |U| without saturation (scenario 1):  %.4f\n', max(abs(U_no(:))));
fprintf('  Max |U| with saturation (scenario 2):     %.4f\n', max(abs(U_sat(:))));
fprintf('  Error norm at t=50 s – no sat:             %.6f\n', err_norm_no(round(50/dt)+1));
fprintf('  Error norm at t=50 s – sat:                %.6f\n', err_norm_sat(round(50/dt)+1));
fprintf('  Error norm at t=80 s – no sat:             %.6f\n', err_norm_no(end));
fprintf('  Error norm at t=80 s – sat:                %.6f\n', err_norm_sat(end));
fprintf('════════════════════════════════════════════════════════\n\n');

%% ─── Step 8: Figures ─────────────────────────────────────────────────────────
set(0,'DefaultAxesFontSize',11,'DefaultLineLineWidth',1.2);

% ── Figure 1: State trajectories without controller (first 40 s) ────────────
figure('Name','States without controller','Position',[50 50 900 600]);
comp_lbl = {'\Re(m_{1,1})', '\Im(m_{1,1})', '\Re(m_{1,2})', '\Im(m_{1,2})'};
extract   = {@(m)real(m(:,1)), @(m)imag(m(:,1)), @(m)real(m(:,2)), @(m)imag(m(:,2))};
idx40 = find(t_arr <= 40, 1, 'last');
for k = 1:4
    subplot(2,2,k);
    plot(t_arr(1:idx40), extract{k}(m1_no(1:idx40,:)), 'b',   'DisplayName','Drive');
    hold on;
    plot(t_arr(1:idx40), extract{k}(m2_no(1:idx40,:)), 'r--', 'DisplayName','Response (no ctrl)');
    xlabel('Time [s]'); ylabel(comp_lbl{k}); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle('Trajectories of states without controller (chaotic)','FontWeight','bold');

% ── Figure 2: State trajectories with controller (no saturation) ─────────────
figure('Name','States with controller (no sat)','Position',[100 50 900 600]);
for k = 1:4
    subplot(2,2,k);
    plot(t_arr, extract{k}(m1_no), 'b',   'DisplayName','Drive');
    hold on;
    plot(t_arr, extract{k}(m2_no), 'r--', 'DisplayName','Response');
    xlabel('Time [s]'); ylabel(comp_lbl{k}); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle('State trajectories with non-fragile SDC (no saturation)','FontWeight','bold');

% ── Figure 3: Error state comparison ─────────────────────────────────────────
figure('Name','Error state comparison','Position',[150 50 900 600]);
err_lbl = {'\Re(h_1)', '\Im(h_1)', '\Re(h_2)', '\Im(h_2)'};
err_ex  = {@(e)real(e(:,1)), @(e)imag(e(:,1)), @(e)real(e(:,2)), @(e)imag(e(:,2))};
for k = 1:4
    subplot(2,2,k);
    plot(t_arr, err_ex{k}(err_no),  'b',   'DisplayName','No saturation');
    hold on;
    plot(t_arr, err_ex{k}(err_sat), 'r--', 'DisplayName', ...
         sprintf('Saturation (u_{max}=%.1f)', U_MAX_SAT));
    yline(0,'k:','LineWidth',0.8);
    xlabel('Time [s]'); ylabel(err_lbl{k}); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle('Synchronisation error: no saturation vs. actuator saturation','FontWeight','bold');

% ── Figure 4: Control inputs ──────────────────────────────────────────────────
figure('Name','Control inputs','Position',[200 50 900 600]);
u_lbl = {'\Re(U_1)', '\Im(U_1)', '\Re(U_2)', '\Im(U_2)'};
u_ex  = {@(u)real(u(:,1)), @(u)imag(u(:,1)), @(u)real(u(:,2)), @(u)imag(u(:,2))};
for k = 1:4
    subplot(2,2,k);
    plot(t_arr, u_ex{k}(U_no),  'b',   'DisplayName','No saturation');
    hold on;
    plot(t_arr, u_ex{k}(U_sat), 'r--', 'DisplayName', ...
         sprintf('Saturated (u_{max}=%.1f)', U_MAX_SAT));
    yline( U_MAX_SAT, 'Color',[0.5 0.5 0.5],'LineStyle','--','DisplayName','+u_{max}');
    yline(-U_MAX_SAT, 'Color',[0.5 0.5 0.5],'LineStyle','--','HandleVisibility','off');
    xlabel('Time [s]'); ylabel(u_lbl{k}); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle(sprintf('Control inputs: no saturation vs. actuator saturation (u_{max}=%.1f)', ...
                U_MAX_SAT), 'FontWeight','bold');

% ── Figure 5: Error norm (log scale) ─────────────────────────────────────────
figure('Name','Error norm','Position',[250 50 800 400]);
semilogy(t_arr, err_norm_no,  'b',   'LineWidth',1.5, 'DisplayName','No saturation');
hold on;
semilogy(t_arr, err_norm_sat, 'r--', 'LineWidth',1.5, 'DisplayName', ...
         sprintf('Actuator saturation (u_{max}=%.1f)', U_MAX_SAT));
xlabel('Time [s]','FontSize',12);
ylabel('||h(t)|| (log scale)','FontSize',12);
title('Synchronisation error norm: effect of actuator saturation','FontWeight','bold');
legend('FontSize',10);  grid on;  xlim([0, T_END]);

% ── Figure 6: Early transient control (zoomed, showing clipping) ─────────────
figure('Name','Early transient control (saturation clipping)','Position',[300 50 900 600]);
idx20 = find(t_arr <= 20, 1, 'last');
for k = 1:4
    subplot(2,2,k);
    plot(t_arr(1:idx20), u_ex{k}(U_no(1:idx20,:)),  'b',   'DisplayName','No saturation');
    hold on;
    plot(t_arr(1:idx20), u_ex{k}(U_sat(1:idx20,:)), 'r--', 'DisplayName','Saturated');
    yline( U_MAX_SAT, 'Color',[0.5 0.5 0.5],'LineStyle','--','DisplayName', ...
           sprintf('+u_{max}=%.1f', U_MAX_SAT));
    yline(-U_MAX_SAT, 'Color',[0.5 0.5 0.5],'LineStyle','--','HandleVisibility','off');
    xlabel('Time [s]'); ylabel(u_lbl{k}); xlim([0 20]); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle(sprintf('Early transient (t ∈ [0,20] s) – saturation clipping at ±%.1f', ...
                U_MAX_SAT), 'FontWeight','bold');

% ── Figure 7: State trajectories with saturation ─────────────────────────────
figure('Name','States with saturation','Position',[350 50 900 600]);
for k = 1:4
    subplot(2,2,k);
    plot(t_arr, extract{k}(m1_sat), 'b',   'DisplayName','Drive');
    hold on;
    plot(t_arr, extract{k}(m2_sat), 'r--', 'DisplayName','Response (saturated ctrl)');
    xlabel('Time [s]'); ylabel(comp_lbl{k}); grid on;
    legend('Location','best','FontSize',8);
end
sgtitle(sprintf('State trajectories with actuator saturation (u_{max}=%.1f)', ...
                U_MAX_SAT), 'FontWeight','bold');

fprintf('All figures generated.\n');

%% ─── Helper function ─────────────────────────────────────────────────────────
function val = dist_integral(fhist, i, l_val, dt)
% Approximate integral ∫_{t-l(t)}^t f(m(s)) ds using trapezoidal rule
    steps = max(1, round(l_val / dt));
    j0    = max(1, i - steps);
    vals  = fhist(j0:i, :);
    if size(vals, 1) > 1
        val = trapz(vals) * dt;
    else
        val = vals(1, :) * dt;
    end
end
