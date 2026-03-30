% save_data.m — Runs NMA continuation for beam_cubicSpring_NM1 and saves results + plot.
% Generates matlab_backbone.png reproducing the MATLAB NMA backbone plot.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Define system
len       = .7;
height    = .014;
thickness = .014;
E         = 2.05e11;
rho       = 7800;
BCs       = 'clamped-free';

n_nodes = 20;
beam = FE_EulerBernoulliBeam(len, height, thickness, E, rho, BCs, n_nodes);

% Cubic spring at free end (translational)
inode = n_nodes;
dir   = 'trans';
knl   = 6e9;
add_nonlinear_attachment(beam, inode, dir, 'cubicSpring', 'stiffness', knl);

n = beam.n;

%% Modal analysis of linearized system
[PHI_lin, OM2] = eig(beam.K, beam.M);
om_lin = sqrt(diag(OM2));
[om_lin, ind] = sort(om_lin);
PHI_lin = PHI_lin(:, ind);

%% Nonlinear modal analysis using harmonic balance
analysis = 'NMA';

H         = 5;
N         = 4*H+1;
imod      = 1;
log10a_s  = -5;
log10a_e  = -3;
inorm     = n-1;  % phase normalisation coordinate (tip transverse DOF, 1-indexed)

om  = om_lin(imod);
phi = PHI_lin(:, imod);
Psi = zeros((2*H+1)*n, 1);
Psi(n+(1:n)) = phi;
x0   = [Psi; om; 0];
qscl = max(abs(Psi));

ds   = .6;
Sopt = struct('Dscale', [qscl*1e-2*ones(size(x0,1)-2,1); om; 1e0; 1e0], ...
    'dynamicDscale', 1);
fscl = mean(abs(beam.K * phi));

X_HB = solve_and_continue(x0, ...
    @(X) HB_residual(X, beam, H, N, analysis, inorm, fscl), ...
    log10a_s, log10a_e, ds, Sopt);

%% Interpret solver output
Psi_HB    = X_HB(1:end-3, :);
om_HB     = X_HB(end-2, :);
del_HB    = X_HB(end-1, :);
log10a_HB = X_HB(end, :);
a_HB      = 10.^log10a_HB;
Q_HB      = Psi_HB .* repmat(a_HB, size(Psi_HB, 1), 1);

%% Compute energy
energy = zeros(size(a_HB));
for i = 1:size(X_HB, 2)
    Qi = reshape(Q_HB(:, i), n, 2*H+1);
    q0 = Qi(:, 1) + sum(Qi(:, 2:2:end), 2);
    u0 = sum(Qi(:, 3:2:end), 2) * om_HB(i);
    energy(i) = 1/2*u0'*beam.M*u0 + 1/2*q0'*beam.K*q0 + knl*q0(n-1)^4/4;
end

%% Compute tip amplitude (fundamental harmonic)
% In Q_HB rows: DC rows 1..n, cos1 rows n+1..2n, sin1 rows 2n+1..3n
% Tip transverse DOF (1-indexed): inorm = n-1
tip_cos1  = Q_HB(n + inorm, :);
tip_sin1  = Q_HB(2*n + inorm, :);
amp_tip_HB = sqrt(tip_cos1.^2 + tip_sin1.^2);

%% Save
save('-mat-binary', 'hb_data.mat', 'om_HB', 'Q_HB', 'a_HB', 'log10a_HB', 'energy', 'amp_tip_HB', 'n', 'H');
disp('Saved hb_data.mat successfully.');

%% Plot — NMA backbone: tip amplitude vs modal frequency (matching MATLAB demo style)
fig = figure('visible', 'off');
hold on;
plot(log10(energy), om_HB / (2*pi), 'k-o', 'markersize', 4, 'linewidth', 1.5);
xlabel('log10(energy)');
ylabel('modal frequency in Hz');
title('Beam + Cubic Spring — NMA Backbone');
ylim([20 50]);
legend('NMA backbone', 'location', 'northwest');
grid on;
box on;
hold off;

print(fig, 'matlab_backbone.png', '-dpng', '-r150');
close(fig);
disp('Saved matlab_backbone.png successfully.');
