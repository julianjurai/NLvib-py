% save_data.m — Full HB + Shooting + Floquet stability for Duffing oscillator.
% Generates matlab_frf.png comparable to the MATLAB demo plot.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Parameters of the Duffing oscillator (from Duffing.m)
mu    = 1;
zeta  = 0.05;
kappa = 1;
gamma = 0.1;
P     = 0.18;

%% Build SingleMassOscillator (needed for shooting_residual)
nonlinear_elements{1} = struct('type','cubicSpring','stiffness',gamma,'force_direction',1);
oscillator = SingleMassOscillator(mu, zeta, kappa, nonlinear_elements, P);

%% HB analysis parameters
H    = 7;           % harmonic order
N    = 4*H + 1;     % number of time samples per period
Om_s = 0.5;         % start frequency
Om_e = 1.6;         % end frequency

%% Initial guess (from underlying linear system)
Q    = (-Om_s^2*mu + 1i*Om_s*zeta + kappa) \ P;
x0   = [0; real(Q); -imag(Q); zeros(2*(H-1), 1)];

%% HB continuation
ds   = 0.01;
Sopt = struct('jac', 'none');
X    = solve_and_continue(x0, ...
    @(X) HB_residual_Duffing(X, mu, zeta, kappa, gamma, P, H, N), ...
    Om_s, Om_e, ds, Sopt);

Om_HB    = X(end, :);
Q_HB     = X(1:end-1, :);
a_rms_HB = sqrt(sum(Q_HB .^ 2)) / sqrt(2);

%% Save HB data
save('-mat-binary', 'hb_data.mat', 'Om_HB', 'Q_HB', 'a_rms_HB');
disp('Saved hb_data.mat successfully.');

%% Shooting continuation
Ntd  = 2^8;
ys   = [real(Q); -Om_s*imag(Q)];
ds   = 0.01;
Sopt2 = struct('Dscale', [1e0*ones(size(ys)); Om_s]);
Np   = 1;
[X_shoot, ~] = solve_and_continue(ys, ...
    @(X) shooting_residual(X, oscillator, Ntd, Np, 'FRF'), ...
    Om_s, Om_e, ds, Sopt2);

Om_shoot = X_shoot(end, :);
Ys       = X_shoot(1:end-1, :);

%% Floquet stability analysis
a_rms_shoot = zeros(size(Om_shoot));
stable      = zeros(size(Om_shoot));
mucrit      = zeros(size(Om_shoot));
BPs(1:length(Om_shoot)) = struct('Om', [], 'a_rms', [], 'type', '');
nBP = 0;

for i = 1:length(Om_shoot)
    [~, ~, ~, Y, dye_dys] = shooting_residual(X_shoot(:, i), oscillator, Ntd, Np, 'FRF');
    Qc = fft(Y(:, 1)) / Ntd;
    a_rms_shoot(i) = sqrt(sum(abs(Qc(1:H+1)).^2)) / sqrt(2) * 2;
    mucrit(i)      = eigs(dye_dys, 1, 'lm');
    stable(i)      = abs(mucrit(i)) <= 1;
    if i > 1 && stable(i) ~= stable(i-1)
        nBP = nBP + 1;
        BPs(nBP).Om    = (Om_shoot(i) + Om_shoot(i-1)) / 2;
        BPs(nBP).a_rms = (a_rms_shoot(i) + a_rms_shoot(i-1)) / 2;
        if abs(angle(mucrit(i))) < 1e-3
            BPs(nBP).type = 'TP';
        elseif abs(abs(angle(mucrit(i))) - pi) < 1e-3
            BPs(nBP).type = 'PD';
        else
            BPs(nBP).type = 'NS';
        end
    end
end
BPs(nBP+1:end) = [];

%% Plot — comparable to MATLAB demo
fig = figure('visible', 'off');
hold on;
plot(Om_HB, a_rms_HB, 'g-', 'linewidth', 1.5);
plot(Om_shoot, a_rms_shoot, 'k--', 'markersize', 5);
plot(Om_shoot(~stable), a_rms_shoot(~stable), 'rx');
xlabel('excitation frequency \Omega');
ylabel('response amplitude');
for i = 1:nBP
    switch BPs(i).type
        case 'TP'; styl = {'ro', 'markersize', 10, 'markerfacecolor', 'r'};
        case 'NS'; styl = {'^',  'markersize', 10, 'markerfacecolor', 'c', 'color', 'c'};
        case 'PD'; styl = {'ms', 'markersize', 10, 'markerfacecolor', 'm'};
    end
    plot(BPs(i).Om, BPs(i).a_rms, styl{:});
end
legend('HB', 'Shooting', 'unstable', 'location', 'northeast');
xlim(sort([Om_s Om_e]));
hold off;
box on; grid on;

print(fig, 'matlab_frf.png', '-dpng', '-r150');
close(fig);
disp('Saved matlab_frf.png successfully.');
