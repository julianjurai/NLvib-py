% save_data.m — Run beam_tanhDryFriction_simple HB and save results + PNG.
% Used by notebooks/comparison/07_beam_tanh_friction.ipynb
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

% -------------------------------------------------------------------------
% Load system matrices
% -------------------------------------------------------------------------
load('beam', 'M', 'D', 'K', 'Fex1', 'W', 'T_tip');

% Define nonlinear element (tanh dry friction)
muN = 1.5;      % friction limit force
eps = 6e-7;     % regularization parameter (tanh-approx. of signum funct.)
nonlinear_elements{1} = struct('type', 'tanhDryFriction', ...
    'friction_limit_force', muN, 'eps', eps, 'force_direction', W);

% Define mechanical system
beam_sys = MechanicalSystem(M, D, K, nonlinear_elements, Fex1);

% Analysis parameters
analysis = 'FRF';
H = 7;
N = 2^7;
Om_s = 370;   % start frequency (rad/s)
Om_e = 110;   % end frequency   (rad/s)

% Initial guess (zero vector — same as beam_tanhDryFriction_simple.m)
x0 = zeros((2*H+1)*length(M), 1);

% Continuation options
% flag=0: sequential stepping; stepadapt=0: fixed step size
ds = 5;
Sopt = struct('flag', 0, 'stepadapt', 0, ...
    'Dscale', [1e-7*ones(size(x0)); Om_s]);

% Solve and continue w.r.t. Om
X = solve_and_continue(x0, ...
    @(X) HB_residual(X, beam_sys, H, N, analysis), ...
    Om_s, Om_e, ds, Sopt);

% Extract results
Om_HB = X(end, :);
Q_HB  = X(1:end-1, :);

% Compute tip RMS response (same formula as beam_tanhDryFriction_simple.m)
Qtip     = kron(eye(2*H+1), T_tip) * Q_HB;
a_rms_HB = sqrt(sum(Qtip.^2)) / sqrt(2);

% Save to binary .mat
save('-mat-binary', 'hb_data.mat', 'Om_HB', 'Q_HB', 'a_rms_HB');
fprintf('save_data.m: saved hb_data.mat with %d steps\n', length(Om_HB));

% -------------------------------------------------------------------------
% Generate matlab_frf.png  (semilogy — amplitudes are ~1e-8)
% -------------------------------------------------------------------------
fig = figure('visible', 'off');
hold on;
semilogy(Om_HB, a_rms_HB, 'g-', 'linewidth', 1.5);
xlabel('excitation frequency (rad/s)');
ylabel('tip displacement a_{rms} (m)');
title('Example 08 — Beam Tanh Dry Friction: HB FRF');
xlim(sort([Om_s Om_e]));
legend('HB', 'location', 'northeast');
box on; grid on;
hold off;

print(fig, 'matlab_frf.png', '-dpng', '-r150');
close(fig);
fprintf('save_data.m: saved matlab_frf.png\n');
