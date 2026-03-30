% save_data.m — Full HB + Shooting + Floquet stability for twoDOFoscillator_cubicSpring.
% Generates matlab_frf.png comparable to the MATLAB demo plot.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));

%% Define system
mi   = [1 .05];
ki   = [1 .0453 0];
di   = [.002 .013 0];
Fex1 = [.11;0];

nonlinear_elements{1} = struct('type','cubicSpring','stiffness',1,  'force_direction',[1;0]);
nonlinear_elements{2} = struct('type','cubicSpring','stiffness',.0042,'force_direction',[1;-1]);

oscillator = ChainOfOscillators(mi,di,ki,nonlinear_elements,Fex1);
n = oscillator.n;

%% HB continuation
analysis = 'FRF';
H = 7; N = 4*H+1;
Om_s = .8; Om_e = 1.4;

Q1 = (-Om_s^2*oscillator.M + 1i*Om_s*oscillator.D + oscillator.K)\Fex1;
y0 = zeros((2*H+1)*length(Q1),1);
y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];

ds = .02;
Sopt = struct('Dscale',[1e-0*ones(size(y0));Om_s]);
[X_HB,~] = solve_and_continue(y0,...
    @(X) HB_residual(X,oscillator,H,N,analysis),...
    Om_s,Om_e,ds,Sopt);

Om_HB    = X_HB(end,:);
Q_HB     = X_HB(1:end-1,:);
a_rms_HB = sqrt(sum(Q_HB(1:2:end,:).^2))/sqrt(2);

%% Shooting continuation
Ntd = 2^8;
ys  = [real(Q1); -Om_s*imag(Q1)];
ds  = .02;
Sopt2 = struct('Dscale',[1e0*ones(size(ys));Om_s]);
Np = 1;
[X_shoot,~] = solve_and_continue(ys,...
    @(X) shooting_residual(X,oscillator,Ntd,Np,analysis),...
    Om_s,Om_e,ds,Sopt2);

Om_shoot    = X_shoot(end,:);
Ys          = X_shoot(1:end-1,:);

%% Floquet stability analysis
a_rms_shoot = zeros(size(Om_shoot));
stable      = zeros(size(Om_shoot));
mucrit      = zeros(size(Om_shoot));
BPs(1:length(Om_shoot)) = struct('Om',[],'a_rms',[],'type','');
nBP = 0;

for i = 1:length(Om_shoot)
    [~,~,~,Y,dye_dys] = shooting_residual(X_shoot(:,i),oscillator,Ntd,Np,analysis);
    Qc = fft(Y(:,1))/Ntd;
    a_rms_shoot(i) = sqrt(sum(abs(Qc(1:H+1)).^2))/sqrt(2)*2;
    mucrit(i)  = eigs(dye_dys,1,'lm');
    stable(i)  = abs(mucrit(i)) <= 1;
    if i > 1 && stable(i) ~= stable(i-1)
        nBP = nBP + 1;
        BPs(nBP).Om    = (Om_shoot(i)+Om_shoot(i-1))/2;
        BPs(nBP).a_rms = (a_rms_shoot(i)+a_rms_shoot(i-1))/2;
        if abs(angle(mucrit(i))) < 1e-3
            BPs(nBP).type = 'TP';
        elseif abs(abs(angle(mucrit(i)))-pi) < 1e-3
            BPs(nBP).type = 'PD';
        else
            BPs(nBP).type = 'NS';
        end
    end
end
BPs(nBP+1:end) = [];

%% Save data
save('-mat-binary','hb_data.mat','Om_HB','Q_HB','a_rms_HB');
disp('Saved hb_data.mat successfully.');

%% Plot — comparable to MATLAB demo
fig = figure('visible','off');
hold on;
plot(Om_HB, a_rms_HB, 'g-', 'linewidth', 1.5);
plot(Om_shoot, a_rms_shoot, 'k--', 'markersize', 5);
plot(Om_shoot(~stable), a_rms_shoot(~stable), 'rx');
xlabel('excitation frequency');
ylabel('response amplitude');
for i = 1:nBP
    switch BPs(i).type
        case 'TP'; styl = {'ro','markersize',10,'markerfacecolor','r'};
        case 'NS'; styl = {'^','markersize',10,'markerfacecolor','c','color','c'};
        case 'PD'; styl = {'ms','markersize',10,'markerfacecolor','m'};
    end
    plot(BPs(i).Om, BPs(i).a_rms, styl{:});
end
legend('HB','Shooting','unstable','location','northeast');
xlim(sort([Om_s Om_e]));
hold off;
box on; grid on;

print(fig, 'matlab_frf.png', '-dpng', '-r150');
close(fig);
disp('Saved matlab_frf.png successfully.');
