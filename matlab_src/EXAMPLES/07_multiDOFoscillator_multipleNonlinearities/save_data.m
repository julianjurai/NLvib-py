%========================================================================
% save_data.m — Octave wrapper for example 07 (Multi-DOF Multiple Nonlinearities).
% Runs HB continuation, saves hb_data.mat, and generates matlab_frf.png.
%========================================================================
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Define system
mi = [1 1 1];
ki = [1 1 1 1];
di = .02*ki;
Fex1 = [0;1;0];

knl = 20; muN = 1;
W1 = [1;0;0];
nonlinear_elements{1} = struct('type','elasticDryFriction',...
    'stiffness',knl,'friction_limit_force',muN,'ishysteretic',true,...
    'force_direction',W1);
W2 = [-1;1;0];
nonlinear_elements{2} = struct('type','elasticDryFriction',...
    'stiffness',knl,'friction_limit_force',muN,'ishysteretic',true,...
    'force_direction',W2);
dKfric = knl*(W1*W1'+W2*W2');

W3 = [0;1;0];
nonlinear_elements{3} = struct('type','cubicSpring',...
    'stiffness',1.,'force_direction',W3);
W4 = [0;-1;1];
nonlinear_elements{4} = struct('type','cubicSpring',...
    'stiffness',1.,'force_direction',W4);

W5 = [0;0;1];
nonlinear_elements{5} = struct('type','unilateralSpring',...
    'stiffness',1.,'gap',.25,'force_direction',W5);

nonlinearMDOFoscillator = ChainOfOscillators(mi,di,ki,...
    nonlinear_elements,Fex1);

%% Compute frequency response using Harmonic Balance
analysis = 'FRF';
H = 7;
N = 2^10;
Om_s = .5;
Om_e = 2;

% Initial guess from underlying linear system (with friction stiffness)
Q1_lin = (-Om_s^2*nonlinearMDOFoscillator.M + ...
    1i*Om_s*nonlinearMDOFoscillator.D + ...
    nonlinearMDOFoscillator.K + dKfric)\Fex1;
x0 = zeros((2*H+1)*length(Q1_lin),1);
x0(length(Q1_lin)+(1:2*length(Q1_lin))) = [real(Q1_lin);-imag(Q1_lin)];

ds = .005;
X = solve_and_continue(x0,...
    @(Y) HB_residual(Y,nonlinearMDOFoscillator,H,N,analysis),...
    Om_s,Om_e,ds);

% Interpret solver output
n = nonlinearMDOFoscillator.n;
Om_HB  = X(end,:);
Q_dof1 = X(1:n:end-1,:);
Q_dof2 = X(2:n:end-1,:);
Q_dof3 = X(3:n:end-1,:);

% RMS amplitudes (MATLAB formula: sqrt(sum(Q.^2))/sqrt(2))
a_rms_dof1 = sqrt(sum(Q_dof1.^2))/sqrt(2);
a_rms_dof2 = sqrt(sum(Q_dof2.^2))/sqrt(2);
a_rms_dof3 = sqrt(sum(Q_dof3.^2))/sqrt(2);

%% Save to .mat file (binary format for scipy.io compatibility)
save('-mat-binary', 'hb_data.mat', 'Om_HB', 'Q_dof1', 'Q_dof2', 'Q_dof3', ...
    'a_rms_dof1', 'a_rms_dof2', 'a_rms_dof3');

fprintf('Saved hb_data.mat with %d continuation steps\n', length(Om_HB));
fprintf('Om_HB range: [%.4f, %.4f]\n', min(Om_HB), max(Om_HB));
fprintf('Max a_rms DOF1: %.6f\n', max(a_rms_dof1));
fprintf('Max a_rms DOF2: %.6f\n', max(a_rms_dof2));
fprintf('Max a_rms DOF3: %.6f\n', max(a_rms_dof3));

%% Generate matlab_frf.png (style matching MATLAB demo)
fig = figure('visible','off');
hold on;
plot(Om_HB, a_rms_dof1, 'b-',  'linewidth', 1.5);
plot(Om_HB, a_rms_dof2, 'r-',  'linewidth', 1.5);
plot(Om_HB, a_rms_dof3, 'g-',  'linewidth', 1.5);
xlabel('excitation frequency');
ylabel('response amplitude (RMS)');
title('Multi-DOF Multiple Nonlinearities — HB FRF');
legend('q1','q2','q3','location','northeast');
xlim([Om_s Om_e]);
hold off;
box on; grid on;

print(fig, 'matlab_frf.png', '-dpng', '-r150');
close(fig);
disp('Saved matlab_frf.png successfully.');
