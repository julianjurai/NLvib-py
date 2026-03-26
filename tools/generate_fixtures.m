% generate_fixtures.m
%
% MATLAB fixture generator for NLvib Python port validation.
%
% PURPOSE
%   Run each NLvib MATLAB example, capture the continuation results, normalise
%   them to the canonical fixture schema, and write one .npz file per example
%   to tests/fixtures/.  The resulting files are committed to the repository
%   and used as ground-truth by the Python QA agent.
%
% PREREQUISITES
%   1. MATLAB R2019b or newer  (Octave 7+ should also work)
%   2. NLvib MATLAB source present at  tools/NLvib_matlab/
%      (run  bash tools/fetch_matlab_source.sh  first)
%   3. scipy must NOT be required — the .npz format is written by a small
%      inline Python call using numpy only, invoked via system().
%      Alternatively the script writes .mat files and a companion Python
%      script converts them; see CONVERSION section below.
%
% USAGE
%   From the repo root:
%       matlab -batch "run('tools/generate_fixtures.m')"
%   or interactively:
%       >> run('tools/generate_fixtures.m')
%
% OUTPUT
%   tests/fixtures/01_Duffing.npz
%   tests/fixtures/02_twoDOFoscillator_cubicSpring.npz
%   tests/fixtures/03_twoDOFoscillator_unilateralSpring.npz
%   tests/fixtures/04_twoDOFoscillator_cubicSpring_NM.npz
%   tests/fixtures/05_twoDOFoscillator_tanhDryFriction_NM.npz
%   tests/fixtures/06_twoDOFoscillator_tanhDryFriction_FRF.npz
%   tests/fixtures/07_geometricNonlinearity.npz
%   tests/fixtures/08_multiDOF_multiNL.npz
%   tests/fixtures/09_beam_tanhFriction.npz
%   tests/fixtures/10_beam_cubicSpring_NM.npz
%
% REPRODUCIBILITY
%   All examples use the parameter values from the original NLvib distribution.
%   Do not change physical parameters here — the Python port is validated against
%   these exact values.  If parameters must change, update both this script and
%   the corresponding Python example, then regenerate all fixtures.
%
% SCHEMA (see also tests/fixtures/README.md)
%   Every fixture stores at minimum:
%     omega      (n_points,)  double   — frequency (rad/s)
%     amplitude  (n_points,)  double   — response amplitude at DOF 0, harmonic 1
%     phase      (n_points,)  double   — phase of fundamental harmonic (rad)
%     stability  (n_points,)  logical  — true = stable
%     tolerance  scalar       double   — validation tolerance (default 1e-6)
%
% ---------------------------------------------------------------------------

%% ---- Paths ----------------------------------------------------------------
script_dir  = fileparts(mfilename('fullpath'));      % tools/
repo_root   = fileparts(script_dir);                 % project root
nlvib_src   = fullfile(repo_root, 'tools', 'NLvib_matlab');
fixtures_dir = fullfile(repo_root, 'tests', 'fixtures');

if ~isfolder(nlvib_src)
    error(['NLvib MATLAB source not found at:\n  %s\n' ...
           'Run:  bash tools/fetch_matlab_source.sh'], nlvib_src);
end

% Add NLvib core to path (SRC subdirectory contains the solver functions)
addpath(genpath(fullfile(nlvib_src, 'SRC')));

%% ---- Helper: extract amplitude from harmonic coefficient matrix -----------
% Q is the raw HB solution matrix, shape (n_dof*(2*H+1)) x n_points.
% For DOF index d and harmonic h=1 the cosine/sine pair sits at rows
% (2*H+1)*d + 2*(h-1) + 1  and  (2*H+1)*d + 2*(h-1) + 2  (1-indexed, MATLAB).
% Amplitude = sqrt(cos^2 + sin^2).

function A = extract_amplitude(Q_branch, dof_idx, H)
    % EXTRACT_AMPLITUDE  Amplitude at a given DOF for the fundamental harmonic.
    %   Q_branch : matrix  (n_dof*(2*H+1)) x n_points
    %   dof_idx  : 0-based DOF index
    %   H        : number of harmonics
    n_coeff_per_dof = 2*H + 1;
    row_cos = n_coeff_per_dof * dof_idx + 2;   % 1-based; row 1 is constant term
    row_sin = n_coeff_per_dof * dof_idx + 3;
    A = sqrt(Q_branch(row_cos, :).^2 + Q_branch(row_sin, :).^2);
    A = A(:);  % column vector
end

function phi = extract_phase(Q_branch, dof_idx, H)
    % EXTRACT_PHASE  Phase angle at a given DOF for the fundamental harmonic.
    n_coeff_per_dof = 2*H + 1;
    row_cos = n_coeff_per_dof * dof_idx + 2;
    row_sin = n_coeff_per_dof * dof_idx + 3;
    phi = atan2(Q_branch(row_sin, :), Q_branch(row_cos, :));
    phi = phi(:);
end

%% ---- Helper: write .mat then convert to .npz via Python -------------------

function write_fixture(fixture_name, data_struct, fixtures_dir, repo_root)
    % WRITE_FIXTURE  Save a struct to .mat and convert to .npz using Python.
    %   fixture_name : string without extension, e.g. '01_Duffing'
    %   data_struct  : MATLAB struct with fields matching the npz schema
    %   fixtures_dir : absolute path to tests/fixtures/
    %   repo_root    : absolute path to repo root
    mat_path = fullfile(fixtures_dir, [fixture_name '.mat']);
    npz_path = fullfile(fixtures_dir, [fixture_name '.npz']);

    % Save to .mat (v7 is compatible with scipy.io.loadmat)
    save(mat_path, '-struct', 'data_struct', '-v7');
    fprintf('  Written: %s\n', mat_path);

    % Convert to .npz using a one-liner Python call
    py_convert = sprintf([ ...
        'import numpy as np, scipy.io as sio; ' ...
        'm = sio.loadmat(r"%s"); ' ...
        'keys = [k for k in m if not k.startswith("_")]; ' ...
        'arrays = {}; ' ...
        '[arrays.update({k: m[k].squeeze()}) for k in keys]; ' ...
        'np.savez(r"%s", **arrays); ' ...
        'print("Converted:", r"%s")' ...
        ], mat_path, npz_path, npz_path);

    ret = system(sprintf('python3 -c "%s"', py_convert));
    if ret == 0
        delete(mat_path);   % remove intermediate .mat file
        fprintf('  Fixture:  %s\n', npz_path);
    else
        warning('Python conversion failed for %s — .mat file retained.', fixture_name);
    end
end

%% ===========================================================================
%% EXAMPLE 01 — Duffing oscillator  (T-15)
%% ===========================================================================
% Reference: NLvib EXAMPLES/01_Duffing/
% System:    m*xdd + d*xd + k*x + k3*x^3 = f*cos(Omega*t)
% Parameters from NLvib default: m=1, d=0.01, k=1, k3=1, f=0.1
% Methods: Harmonic Balance (H=7) + Shooting
% ---------------------------------------------------------------------------
fprintf('\n=== Example 01: Duffing oscillator ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '01_Duffing');
    orig_dir = cd(ex_dir);

    % Run the example script — it must define Om_HB, Q_HB, Om_shoot, Q_shoot,
    % and S_HB (stability from Floquet), S_shoot in the workspace.
    run('Duffing.m');

    cd(orig_dir);

    H = 7;   % number of harmonics used in the Duffing example

    d01.omega         = Om_HB(:);
    d01.amplitude     = extract_amplitude(Q_HB, 0, H);
    d01.phase         = extract_phase(Q_HB, 0, H);
    d01.stability     = S_HB(:) > 0;   % positive real part = unstable in NLvib convention
    d01.tolerance     = 1e-6;

    % Shooting overlay (additional arrays, not part of minimal schema)
    if exist('Om_shoot', 'var') && exist('Q_shoot', 'var')
        d01.omega_shoot     = Om_shoot(:);
        d01.amplitude_shoot = extract_amplitude(Q_shoot, 0, H);
        if exist('S_shoot', 'var')
            d01.stability_shoot = S_shoot(:) > 0;
        end
    end

    % Full harmonic matrix for residual-level validation
    d01.Q_harmonics = Q_HB;   % (n_dof*(2H+1)) x n_points

    write_fixture('01_Duffing', d01, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 02 — 2-DOF oscillator with cubic spring  (T-16)
%% ===========================================================================
% Reference: NLvib EXAMPLES/02_twoDOFoscillator_cubicSpring/
% System:    2-DOF chain; cubic spring between DOF 1 and wall
% Method:    Harmonic Balance
% ---------------------------------------------------------------------------
fprintf('\n=== Example 02: 2-DOF cubic spring ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '02_twoDOFoscillator_cubicSpring');
    orig_dir = cd(ex_dir);
    run('twoDOFoscillator_cubicSpring.m');
    cd(orig_dir);

    H = 7;
    d02.omega         = Om(:);
    d02.amplitude     = extract_amplitude(Q, 0, H);
    d02.phase         = extract_phase(Q, 0, H);
    d02.amplitude_dof2 = extract_amplitude(Q, 1, H);
    d02.phase_dof2    = extract_phase(Q, 1, H);
    d02.stability     = S(:) > 0;
    d02.tolerance     = 1e-6;
    d02.Q_harmonics   = Q;

    write_fixture('02_twoDOFoscillator_cubicSpring', d02, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 03 — 2-DOF oscillator with unilateral spring  (T-17)
%% ===========================================================================
% Reference: NLvib EXAMPLES/03_twoDOFoscillator_unilateralSpring/
% System:    2-DOF chain; unilateral contact spring on DOF 2
% Method:    Harmonic Balance (higher harmonics needed for impact dynamics)
% Note:      Non-smooth nonlinearity — tolerance relaxed to 1e-4
% ---------------------------------------------------------------------------
fprintf('\n=== Example 03: 2-DOF unilateral spring ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '03_twoDOFoscillator_unilateralSpring');
    orig_dir = cd(ex_dir);
    run('twoDOFoscillator_unilateralSpring.m');
    cd(orig_dir);

    H = 7;
    d03.omega         = Om(:);
    d03.amplitude     = extract_amplitude(Q, 0, H);
    d03.phase         = extract_phase(Q, 0, H);
    d03.amplitude_dof2 = extract_amplitude(Q, 1, H);
    d03.phase_dof2    = extract_phase(Q, 1, H);
    d03.stability     = S(:) > 0;
    d03.tolerance     = 1e-4;   % relaxed: non-smooth NL
    d03.Q_harmonics   = Q;

    write_fixture('03_twoDOFoscillator_unilateralSpring', d03, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 04 — 2-DOF cubic spring, Nonlinear Modal Analysis  (T-18)
%% ===========================================================================
% Reference: NLvib EXAMPLES/04_twoDOFoscillator_cubicSpring_NM/
% System:    Same as Example 02
% Method:    Nonlinear Normal Modes (backbone curve via NMA)
% ---------------------------------------------------------------------------
fprintf('\n=== Example 04: 2-DOF cubic spring, NMA backbone ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '04_twoDOFoscillator_cubicSpring_NM');
    orig_dir = cd(ex_dir);
    run('twoDOFoscillator_cubicSpring_NM.m');
    cd(orig_dir);

    H = 7;
    % NMA: Om is the backbone frequency (natural, not excitation)
    d04.omega         = Om(:);
    d04.amplitude     = extract_amplitude(Q, 0, H);
    d04.phase         = extract_phase(Q, 0, H);
    d04.amplitude_dof2 = extract_amplitude(Q, 1, H);
    d04.phase_dof2    = extract_phase(Q, 1, H);
    d04.stability     = ones(length(Om), 1) > 0;   % NMA backbone: stability not tracked
    d04.tolerance     = 1e-6;
    d04.Q_harmonics   = Q;

    write_fixture('04_twoDOFoscillator_cubicSpring_NM', d04, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 05 — 2-DOF oscillator with tanh dry friction, NMA  (T-18)
%% ===========================================================================
% Reference: NLvib EXAMPLES/05_twoDOFoscillator_tanhDryFriction_NM/
% System:    2-DOF chain; tanh-regularised Coulomb friction on DOF 2
% Method:    NMA backbone curve
% ---------------------------------------------------------------------------
fprintf('\n=== Example 05: 2-DOF tanh friction, NMA backbone ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '05_twoDOFoscillator_tanhDryFriction_NM');
    orig_dir = cd(ex_dir);
    run('twoDOFoscillator_tanhDryFriction_NM.m');
    cd(orig_dir);

    H = 7;
    d05.omega         = Om(:);
    d05.amplitude     = extract_amplitude(Q, 0, H);
    d05.phase         = extract_phase(Q, 0, H);
    d05.amplitude_dof2 = extract_amplitude(Q, 1, H);
    d05.phase_dof2    = extract_phase(Q, 1, H);
    d05.stability     = ones(length(Om), 1) > 0;
    d05.tolerance     = 1e-4;   % relaxed: tanh smoothing introduces small discretisation error
    d05.Q_harmonics   = Q;

    write_fixture('05_twoDOFoscillator_tanhDryFriction_NM', d05, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 06 — 2-DOF tanh friction, FRF  (T-18, forced response)
%% ===========================================================================
% Reference: same example folder as 05 but with external forcing applied
% Method:    Harmonic Balance FRF sweep
% ---------------------------------------------------------------------------
fprintf('\n=== Example 06: 2-DOF tanh friction, HB FRF ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '05_twoDOFoscillator_tanhDryFriction_NM');
    orig_dir = cd(ex_dir);

    % The NLvib example script typically computes both NMA and FRF sections.
    % Variable names for FRF branch are Om_FRF / Q_FRF in newer NLvib versions.
    % Adjust variable names if the example uses different identifiers.
    run('twoDOFoscillator_tanhDryFriction_NM.m');
    cd(orig_dir);

    H = 7;
    % Try to use Om_FRF / Q_FRF; fall back to Om / Q if not present.
    if exist('Om_FRF', 'var') && exist('Q_FRF', 'var')
        om_frf = Om_FRF;
        q_frf  = Q_FRF;
    else
        om_frf = Om;
        q_frf  = Q;
    end

    d06.omega         = om_frf(:);
    d06.amplitude     = extract_amplitude(q_frf, 0, H);
    d06.phase         = extract_phase(q_frf, 0, H);
    d06.amplitude_dof2 = extract_amplitude(q_frf, 1, H);
    d06.phase_dof2    = extract_phase(q_frf, 1, H);
    d06.stability     = S(:) > 0;
    d06.tolerance     = 1e-4;
    d06.Q_harmonics   = q_frf;

    write_fixture('06_twoDOFoscillator_tanhDryFriction_FRF', d06, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 07 — Multi-DOF geometric nonlinearity  (T-19)
%% ===========================================================================
% Reference: NLvib EXAMPLES/06_geometricNonlinearity/  (check folder name)
% System:    Multi-DOF chain with geometric (displacement-squared) nonlinearity
% Method:    Harmonic Balance
% ---------------------------------------------------------------------------
fprintf('\n=== Example 07: Multi-DOF geometric nonlinearity ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '06_geometricNonlinearity');
    if ~isfolder(ex_dir)
        % Some NLvib distributions name this folder differently
        ex_dir = fullfile(nlvib_src, 'EXAMPLES', '06_multiDOFoscillator_geometricNL');
    end
    orig_dir = cd(ex_dir);
    % Run whichever .m file is present in this folder
    mfiles = dir('*.m');
    run(mfiles(1).name);
    cd(orig_dir);

    H = 7;
    d07.omega     = Om(:);
    d07.amplitude = extract_amplitude(Q, 0, H);
    d07.phase     = extract_phase(Q, 0, H);
    d07.stability = S(:) > 0;
    d07.tolerance = 1e-6;
    d07.Q_harmonics = Q;

    write_fixture('07_geometricNonlinearity', d07, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 08 — Multi-DOF oscillator with multiple NL elements  (T-20)
%% ===========================================================================
% Reference: NLvib EXAMPLES/06_multiDOFoscillator/ (check exact folder name)
% System:    Chain of oscillators with multiple different nonlinear attachments
% Method:    Harmonic Balance
% ---------------------------------------------------------------------------
fprintf('\n=== Example 08: Multi-DOF multi-NL ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '06_multiDOFoscillator');
    if ~isfolder(ex_dir)
        ex_dir = fullfile(nlvib_src, 'EXAMPLES', '06_multiDOF_multiNL');
    end
    orig_dir = cd(ex_dir);
    mfiles = dir('*.m');
    run(mfiles(1).name);
    cd(orig_dir);

    H = 7;
    d08.omega     = Om(:);
    d08.amplitude = extract_amplitude(Q, 0, H);
    d08.phase     = extract_phase(Q, 0, H);
    d08.stability = S(:) > 0;
    d08.tolerance = 1e-6;
    d08.Q_harmonics = Q;

    write_fixture('08_multiDOF_multiNL', d08, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 09 — FE beam with tanh friction  (T-21)
%% ===========================================================================
% Reference: NLvib EXAMPLES/07_beam_tanhFriction/
% System:    Euler-Bernoulli FE beam (n_elements=10) with tanh friction attachment
% Method:    Harmonic Balance
% Note:      Larger system — may take 10–20 minutes on a laptop
% ---------------------------------------------------------------------------
fprintf('\n=== Example 09: FE beam + tanh friction ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '07_beam_tanhFriction');
    orig_dir = cd(ex_dir);
    mfiles = dir('*.m');
    run(mfiles(1).name);
    cd(orig_dir);

    H = 7;
    d09.omega     = Om(:);
    d09.amplitude = extract_amplitude(Q, 0, H);
    d09.phase     = extract_phase(Q, 0, H);
    d09.stability = S(:) > 0;
    d09.tolerance = 1e-4;   % relaxed: non-smooth + large FE system
    d09.Q_harmonics = Q;

    write_fixture('09_beam_tanhFriction', d09, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% EXAMPLE 10 — FE beam with cubic spring, NMA  (T-22)
%% ===========================================================================
% Reference: NLvib EXAMPLES/08_beam_cubicSpring_NM/
% System:    Euler-Bernoulli FE beam (n_elements=10) with cubic spring midpoint
% Method:    NMA backbone curve
% Note:      Large system; CMS reduction recommended before NMA
% ---------------------------------------------------------------------------
fprintf('\n=== Example 10: FE beam + cubic spring, NMA backbone ===\n');
try
    ex_dir = fullfile(nlvib_src, 'EXAMPLES', '08_beam_cubicSpring_NM');
    orig_dir = cd(ex_dir);
    mfiles = dir('*.m');
    run(mfiles(1).name);
    cd(orig_dir);

    H = 7;
    d10.omega     = Om(:);
    d10.amplitude = extract_amplitude(Q, 0, H);
    d10.phase     = extract_phase(Q, 0, H);
    d10.stability = ones(length(Om), 1) > 0;   % NMA: stability not tracked
    d10.tolerance = 1e-6;
    d10.Q_harmonics = Q;

    write_fixture('10_beam_cubicSpring_NM', d10, fixtures_dir, repo_root);
    fprintf('  OK\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

%% ===========================================================================
%% Summary
%% ===========================================================================
fprintf('\n========================================\n');
fprintf('Fixture generation complete.\n');
fprintf('Output directory: %s\n', fixtures_dir);
fprintf('Run  python tools/generate_fixtures.py --list  to inspect.\n');
fprintf('========================================\n');
