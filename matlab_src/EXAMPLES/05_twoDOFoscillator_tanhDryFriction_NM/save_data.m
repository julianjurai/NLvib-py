% save_data.m — Run NMA backbone for 2-DOF tanh friction and save results.
% Used by notebooks/comparison/04_two_dof_tanh_friction.ipynb
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Define system (from twoDOFoscillator_tanhDryFriction_NM.m)
mi = [.02 1];           % masses
ki = [0 40 600];        % springs
di = 0*ki;              % no dampers

eps = .05;
muN = 5;
W = [1;0];
nonlinear_elements = struct('type','tanhDryFriction',...
    'eps',eps,'friction_limit_force',muN,'force_direction',W);

oscillator = ChainOfOscillators(mi,di,ki, nonlinear_elements);
n = oscillator.n;

%% Modal analysis
[PHI_free,OM2] = eig(oscillator.K,oscillator.M);
om_free = sqrt(diag(OM2));
[om_free,ind] = sort(om_free); PHI_free = PHI_free(:,ind);

inl = find(W); B = eye(length(oscillator.M)); B(:,inl) = [];
[PHI_fixed,OM2] = eig(B'*oscillator.K*B,B'*oscillator.M*B);
om_fixed = sqrt(diag(OM2));
[om_fixed,ind] = sort(om_fixed); PHI_fixed = B*PHI_fixed(:,ind);

%% Nonlinear modal analysis using harmonic balance
analysis = 'NMA';

H = 21;
Ntd = 2^10;
imod = 1;
log10a_s = -1.5;
log10a_e = 1;
inorm = 2;

om = om_fixed(imod); phi = PHI_fixed(:,imod);
Psi = zeros((2*H+1)*n,1);
Psi(n+(1:n)) = phi;
x0 = [Psi;om;0];

ds = .01;
fscl = mean(abs(oscillator.K*phi));
X_HB = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,Ntd,analysis,inorm,fscl),...
    log10a_s,log10a_e,ds);

% Interpret solver output
Psi_HB = X_HB(1:end-3,:);
om_HB = X_HB(end-2,:);
del_HB = X_HB(end-1,:);
log10a_HB = X_HB(end,:);
a_HB = 10.^log10a_HB;
Q_HB = Psi_HB.*repmat(a_HB,size(Psi_HB,1),1);

% Compute log10(q^s(inorm)) for x-axis (same as the MATLAB plot)
log10qsinorm_HB = log10(sum(real(Q_HB(inorm:n:end,:))));

% Also save om_fixed for normalisation reference
om0_fixed = om_fixed(imod);

%% Save data
save('-mat-binary', 'hb_data.mat', 'om_HB', 'del_HB', 'log10a_HB', 'a_HB', 'Q_HB', 'log10qsinorm_HB', 'om0_fixed');
disp('save_data.m: hb_data.mat written successfully.');

%% Plot — two subplots side by side, matching MATLAB demo style
fig = figure('visible', 'off');

% Left subplot: normalised frequency vs log10(amplitude)
subplot(1, 2, 1);
hold on;
plot(log10qsinorm_HB, om_HB / om0_fixed, 'g-', 'linewidth', 1.5);
xlabel('log10(q^s(i_{norm}))');
ylabel('\omega/\omega_0');
title('Backbone Curve — Normalised Frequency');
legend('HB', 'location', 'northwest');
grid on;
box on;
hold off;

% Right subplot: modal damping ratio vs log10(amplitude)
subplot(1, 2, 2);
hold on;
plot(log10qsinorm_HB, del_HB * 1e2, 'g-', 'linewidth', 1.5);
xlabel('log10(q^s(i_{norm}))');
ylabel('modal damping ratio in %');
title('Modal Damping Ratio');
legend('HB', 'location', 'northwest');
grid on;
box on;
hold off;

print(fig, 'matlab_backbone.png', '-dpng', '-r150');
close(fig);
disp('save_data.m: matlab_backbone.png written successfully.');
