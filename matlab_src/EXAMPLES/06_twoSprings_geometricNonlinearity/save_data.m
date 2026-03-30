% save_data.m — HB NMA backbone + FRF (4 excitation levels) for
% twoSprings_geometricNonlinearity. Saves hb_data.mat for Python comparison.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Define system (from twoSprings_geometricNonlinearity.m)
om1 = 1.13;
om2 = 2.0;
zt1 = 1e-3;
zt2 = 5e-3;

M = eye(2);
D = diag([2*zt1*om1 2*zt2*om2]);
K = diag([om1^2 om2^2]);

p = [2 0; 1 1; 0 2; 3 0; 2 1; 1 2; 0 3];
E = [3*om1^2/2 om2^2/2; om2^2 om1^2; om1^2/2 3*om2^2/2; ...
    (om1^2+om2^2)/2*[1 0; 0 1; 1 0; 0 1] ...
    ];

Fex1 = [1;1];
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);
n = oscillator.n;

%% HB parameters
H = 7;
N = 4*H+1;

%% =========================================================
%  NMA backbone (HB, autonomous)
%  Matches twoSprings_geometricNonlinearity.m exactly
%% =========================================================
analysis = 'NMA';
imod = 1;
log10a_s = -6;
log10a_e = -0.15;
inorm = 1;

om_lin = sqrt(K(imod,imod));
phi_lin = zeros(n,1); phi_lin(imod) = 1;
Psi0 = zeros((2*H+1)*n, 1);
Psi0(n+(1:n)) = phi_lin;
x0_nma = [Psi0; om_lin; 0];

ds_nma = 0.01;
Sopt_nma = struct();
Sopt_nma.termination_criterion = {@(Y) Y(end-2)<1.0};  % stop if om<1.0

X_nma = solve_and_continue(x0_nma, ...
    @(X) HB_residual(X, oscillator, H, N, analysis, inorm), ...
    log10a_s, log10a_e, ds_nma, Sopt_nma);

Psi_NM  = X_nma(1:end-3, :);
om_NM   = X_nma(end-2, :);
del_NM  = X_nma(end-1, :);
log10a_NM = X_nma(end, :);
a_NM_scale = 10.^log10a_NM;
Q_NM    = Psi_NM .* repmat(a_NM_scale, size(Psi_NM,1), 1);

% Fundamental harmonic amplitude of DOF 0 (MATLAB 1-indexed: n+1 and 2n+1)
a_NM = sqrt(Q_NM(n+1,:).^2 + Q_NM(2*n+1,:).^2);

%% =========================================================
%  FRF: upward sweep from Om_e=0.8 to Om_s=1.6, 4 exc levels
%  Matching Python continuation direction for fair comparison
%% =========================================================
analysis = 'FRF';
Om_e = 0.8;   % start (Python sweeps up from 0.8)
Om_s = 1.6;   % end

exc_lev = [3e-4 5e-4 1e-3 3e-3];
Om_cell   = cell(size(exc_lev));
a_cell    = cell(size(exc_lev));

for iex = 1:length(exc_lev)
    oscillator.Fex1 = Fex1 * exc_lev(iex);

    % Initial guess at Om_e (upward sweep — matches Python)
    Q1 = (-Om_e^2*M + 1i*Om_e*D + K) \ oscillator.Fex1;
    y0 = zeros((2*H+1)*n, 1);
    y0(n+(1:2*n)) = [real(Q1); -imag(Q1)];
    qscl = max(abs((-om1^2*M + 1i*om1*D + K) \ oscillator.Fex1));

    ds = 0.005;
    Sopt = struct('Dscale', [qscl*ones(size(y0)); Om_s], 'eps', 1e-6, 'stepmax', 3000);
    X = solve_and_continue(y0, ...
        @(X) HB_residual(X, oscillator, H, N, analysis), ...
        Om_e, Om_s, ds, Sopt);

    Om_cell{iex} = X(end, :);
    Q_tmp = X(1:end-1, :);
    % Fundamental harmonic amplitude of DOF 0
    a_cell{iex} = sqrt(Q_tmp(n+1,:).^2 + Q_tmp(2*n+1,:).^2);
end

% Store for comparison: reference level iref=3 (exc_lev=1e-3)
iref = 3;
Om_HB     = Om_cell{iref};
a_fund_HB = a_cell{iref};

%% Save
save('-mat-binary', 'hb_data.mat', ...
    'Om_HB', 'a_fund_HB', ...
    'om_NM', 'a_NM', ...
    'Om_cell', 'a_cell', 'exc_lev');
disp('Saved hb_data.mat successfully.');

%% Plot — comparable to MATLAB demo (backbone + 4 FRF levels)
fig = figure('visible','off');
hold on;
for iex = 1:length(exc_lev)
    plot(Om_cell{iex}, a_cell{iex}, 'g-', 'linewidth', 1.5);
end
plot(om_NM, a_NM, 'k-', 'linewidth', 1.5);
plot([om1 om1], [0 0.35], 'k--', 'linewidth', 0.8);
xlabel('excitation frequency');
ylabel('response amplitude |Q_{0,1}|');
title('Example 06 — Geometric Nonlinearity (HB, 4 exc levels + backbone)');
set(gca, 'xlim', [1.085 1.15], 'ylim', [0 0.35]);
hold off;
box on; grid on;

print(fig, 'matlab_frf.png', '-dpng', '-r150');
close(fig);
disp('Saved matlab_frf.png successfully.');
