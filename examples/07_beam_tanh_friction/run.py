"""
Example 07 — FE Euler-Bernoulli Beam with Tanh Dry Friction.

Cantilever beam (clamped-free) with tanh dry friction at midspan (node 3).
Matches MATLAB 08_beam_tanhDryFriction/beam_tanhDryFriction_simple.m exactly.

System parameters (MATLAB: beam_tanhDryFriction_advanced.m / beam.mat)
-----------------
n_elements = 8       (9 nodes, 16 free DOFs)
L          = 2.0  m
height     = 0.1  m  (bending direction)
thickness  = 0.3  m
E          = 185e9 Pa
rho        = 7830  kg/m^3
bc         = "clamped-free"
I_area     = height**3 * thickness / 12
A_sect     = height * thickness

Nonlinearity: tanh_dry_friction(f0=1.5, c=1/6e-7≈1.67e6) at node 3, dof_type="w"
Excitation  : F = 0.2 at tip (node 8), dof_type="w", harmonic 1
Frequency   : omega in [110, 370] rad/s, n_harmonics = 7

Note on Jacobian FD step
------------------------
The tanh regularisation (eps=6e-7) makes Q ~ 1e-8.  The default finite-
difference step sqrt(machine_eps) ≈ 1.5e-8 would be larger than Q itself,
producing a meaningless Jacobian.  We patch _FD_STEP = 1.5e-15 so the
perturbation is always small relative to the solution magnitude.

Outputs
-------
examples/07_beam_tanh_friction/output/frf.png
  - FRF at tip node (node 8), a_rms vs omega
examples/07_beam_tanh_friction/output/mode_shape.png
  - Mode shape (transverse displacement DOFs) at resonance peak

Printed summary
---------------
- Linear first natural frequency
- Resonance frequency and tip a_rms at resonance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Patch FD step BEFORE importing hb_residual so the override takes effect.
# Default sqrt(eps_machine) ≈ 1.5e-8 is larger than Q ~ 1e-8 for this example.
import nlvib.solvers.harmonic_balance as _hb_mod
_hb_mod._FD_STEP = 1.5e-15

from scipy.optimize import fsolve

from nlvib.systems.fe_beam import FE_EulerBernoulliBeam
from nlvib.nonlinearities.elements import tanh_dry_friction
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System definition (MATLAB: beam_tanhDryFriction_advanced.m / beam.mat)
# ---------------------------------------------------------------------------
N_ELEMENTS = 8         # 9 nodes → 8 elements → 16 free DOFs (clamped-free)
L_BEAM     = 2.0       # m  (MATLAB: len=2)
H_HEIGHT   = 0.1       # m  (MATLAB: height = 0.05*len = 0.1)
T_THICK    = 0.3       # m  (MATLAB: thickness = 3*height = 0.3)
E_MOD      = 185e9     # Pa (MATLAB: E=185e9)
RHO        = 7830.0    # kg/m^3
BC         = "clamped-free"

I_AREA = H_HEIGHT**3 * T_THICK / 12   # m^4
A_SECT = H_HEIGHT * T_THICK            # m^2

FRICTION_NODE = 3      # MATLAB: inode=4 (1-indexed) → node 3 (0-indexed)
MU_N          = 1.5    # MATLAB: muN=1.5
EPS_TANH      = 6e-7   # MATLAB: eps=6e-7
C_TANH        = 1.0 / EPS_TANH

FORCE_NODE  = 8        # tip node (0-indexed, last free node)
FORCE_AMP   = 0.2      # MATLAB: Fex1 amplitude

N_HARMONICS = 7        # MATLAB: H=7
OMEGA_MIN   = 110.0    # MATLAB: Om_s=370 (traces backward); sweep [110, 370]
OMEGA_MAX   = 370.0    # MATLAB: Om_e=110

beam = FE_EulerBernoulliBeam(
    n_elements=N_ELEMENTS,
    L=L_BEAM,
    E=E_MOD,
    I_area=I_AREA,
    rho=RHO,
    A=A_SECT,
    bc=BC,
)

friction_dof = beam.find_dof(FRICTION_NODE, "w")
nl_element   = tanh_dry_friction(f0=MU_N, c=C_TANH, dof_index=friction_dof)
beam.add_nonlinear_attachment(FRICTION_NODE, "w", nl_element)

beam.add_forcing(FORCE_NODE, "w", FORCE_AMP)

tip_dof    = beam.find_dof(FORCE_NODE, "w")
excitation = {"dof": tip_dof, "amplitude": FORCE_AMP, "harmonic": 1}

# ---------------------------------------------------------------------------
# Linear eigenfrequency (first mode) for reference
# ---------------------------------------------------------------------------
K_dense = beam.K.toarray()
M_dense = beam.M.toarray()
eigvals, _ = eigh(K_dense, M_dense)
omega1_linear = float(np.sqrt(np.abs(eigvals[0])))
print(f"Linear first natural frequency: {omega1_linear:.3f} rad/s  (expected ~123.341 rad/s)")

# ---------------------------------------------------------------------------
# Initial solution — linear guess + scipy.optimize.fsolve
# ---------------------------------------------------------------------------
# Manual Newton diverges for this problem because the tanh regularisation
# (c ≈ 1.67e6) creates very small Q ~ 1e-8, and the step-limiter prevents
# convergence.  scipy.optimize.fsolve uses a more robust trust-region
# dogleg method that converges reliably for this example.
# ---------------------------------------------------------------------------
n_dof   = beam.n_dof
n_total = n_dof * (2 * N_HARMONICS + 1)

# Linearized tanh stiffness for initial guess (valid for q << 1/c)
k_eff_tanh = MU_N * C_TANH  # = f0 * c = MU_N / EPS_TANH ≈ 2.5e6 N/m
K_eff = K_dense.copy()
K_eff[friction_dof, friction_dof] += k_eff_tanh

# Start continuation below the resonance (omega=110, well-conditioned side)
omega_start = OMEGA_MIN
D_dense = beam.D.toarray()
Fex = np.zeros(n_dof)
Fex[tip_dof] = FORCE_AMP
Q1_complex = np.linalg.solve(
    -(omega_start**2) * M_dense + 1j * omega_start * D_dense + K_eff, Fex
)
Q0_guess = np.zeros(n_total, dtype=np.float64)
Q0_guess[n_dof * 1 : n_dof * 2] =  np.real(Q1_complex)
Q0_guess[n_dof * 2 : n_dof * 3] = -np.imag(Q1_complex)

def _res(Q: np.ndarray) -> np.ndarray:
    return hb_residual(Q, omega_start, beam, N_HARMONICS, excitation)[0]

def _jac(Q: np.ndarray) -> np.ndarray:
    return hb_residual(Q, omega_start, beam, N_HARMONICS, excitation)[1]

Q0, _info, ier, msg = fsolve(_res, Q0_guess, fprime=_jac, maxfev=500, full_output=True)
R, _ = hb_residual(Q0, omega_start, beam, N_HARMONICS, excitation)
print(f"fsolve ier={ier}: {msg}")
print(f"Initial residual at omega={omega_start:.1f}: {np.linalg.norm(R):.3e}")
if np.linalg.norm(R) > 1e-6:
    raise RuntimeError(
        f"Initial Newton solve failed: residual = {np.linalg.norm(R):.3e}. "
        "Cannot start continuation."
    )

# ---------------------------------------------------------------------------
# Arc-length continuation — sweep from OMEGA_MIN up to OMEGA_MAX
# ---------------------------------------------------------------------------
def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    return hb_residual(Q, omega, beam, N_HARMONICS, excitation)

solver = ContinuationSolver()
opts = ContinuationOptions(
    verbose=True,
    ds_initial=5.0,
    ds_min=0.5,
    ds_max=8.0,    # cap at 8 rad/s so the resonance peak (~195 rad/s) is sampled
    max_steps=300,
    newton_tol=1e-6,
    lambda_min=OMEGA_MIN - 5.0,
    lambda_max=OMEGA_MAX + 5.0,
)
result = solver.run(residual_fn, Q0, omega_start, opts)

print(f"Continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract a_rms at tip DOF
# MATLAB formula: a_rms = sqrt(sum(Qtip^2)) / sqrt(2)
# where Qtip = [c0, c1, s1, c2, s2, ...] for tip DOF across all harmonics
# ---------------------------------------------------------------------------
solutions = result.solutions  # (n_steps, n_total + 1)
omegas    = solutions[:, -1]
Q_all     = solutions[:, :-1]  # (n_steps, n_dof*(2H+1))

# Reshape to (n_steps, 2H+1, n_dof) and extract tip DOF
Qtip_all = Q_all.reshape(Q_all.shape[0], 2 * N_HARMONICS + 1, n_dof)[:, :, tip_dof]
a_rms    = np.sqrt(np.sum(Qtip_all**2, axis=1)) / np.sqrt(2)

# Filter to swept range
mask   = (omegas >= OMEGA_MIN) & (omegas <= OMEGA_MAX)
omegas_plot = omegas[mask]
a_rms_plot  = a_rms[mask]

if len(a_rms_plot) > 0:
    peak_idx       = int(np.argmax(a_rms_plot))
    peak_a_rms     = float(a_rms_plot[peak_idx])
    resonance_freq = float(omegas_plot[peak_idx])
else:
    peak_a_rms     = float("nan")
    resonance_freq = float("nan")

# ---------------------------------------------------------------------------
# Plot: FRF at tip  (matches MATLAB save_data.m style)
# ---------------------------------------------------------------------------
fig_frf, ax_frf = plt.subplots(figsize=(8, 5))
ax_frf.semilogy(omegas_plot, a_rms_plot, "g-", linewidth=1.5, label="HB")
ax_frf.legend(loc="upper right")
ax_frf.set_xlabel("excitation frequency")
ax_frf.set_ylabel("tip displacement amplitude")
ax_frf.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax_frf.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

frf_path = OUTPUT_DIR / "frf.png"
fig_frf.tight_layout()
fig_frf.savefig(frf_path, dpi=150)
plt.close(fig_frf)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Mode shape at resonance peak
# ---------------------------------------------------------------------------
if len(a_rms_plot) > 0:
    global_peak_idx = int(np.where(mask)[0][peak_idx])
    Q_peak = solutions[global_peak_idx, :n_total]
else:
    global_peak_idx = 0
    Q_peak = solutions[0, :n_total]

node_positions = [0.0]
mode_shape_amp = [0.0]

for node_i in range(1, N_ELEMENTS + 1):
    try:
        reduced_dof = beam.find_dof(node_i, "w")
        cos1_val    = Q_peak[n_dof * 1 + reduced_dof]
        sin1_val    = Q_peak[n_dof * 2 + reduced_dof]
        amp_val     = float(np.sqrt(cos1_val**2 + sin1_val**2))
        node_pos    = node_i * L_BEAM / N_ELEMENTS
        node_positions.append(node_pos)
        mode_shape_amp.append(amp_val)
    except ValueError:
        pass

node_positions_arr = np.array(node_positions)
mode_shape_arr     = np.array(mode_shape_amp)

fig_mode, ax_mode = plt.subplots(figsize=(8, 4))
ax_mode.plot(node_positions_arr, np.zeros_like(node_positions_arr), "k--",
             linewidth=0.6, label="undeformed")
ax_mode.plot(node_positions_arr, mode_shape_arr, "o-", color="tab:green",
             linewidth=1.5, label=f"mode shape at resonance ({resonance_freq:.1f} rad/s)")
ax_mode.fill_between(node_positions_arr, mode_shape_arr, alpha=0.2, color="tab:green")
ax_mode.set_xlabel("Position along beam (m)")
ax_mode.set_ylabel("Transverse amplitude (m)")
ax_mode.set_title("Example 07 — Mode Shape at Resonance Peak")
ax_mode.legend()
ax_mode.grid(True, alpha=0.3)

mode_path = OUTPUT_DIR / "mode_shape.png"
fig_mode.tight_layout()
fig_mode.savefig(mode_path, dpi=150)
plt.close(fig_mode)
print(f"Plot saved: {mode_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  Example 07 — Beam Tanh Friction Summary")
print("=" * 55)
print(f"  n_dof                        : {n_dof}")
print(f"  Friction DOF (node {FRICTION_NODE})        : {friction_dof}  (expected 4)")
print(f"  Tip DOF (node {FORCE_NODE})             : {tip_dof}  (expected 14)")
print(f"  Linear omega_1               : {omega1_linear:.3f} rad/s")
print(f"  Resonance frequency (HB)     : {resonance_freq:.3f} rad/s")
print(f"  Tip a_rms at resonance       : {peak_a_rms:.6e} m")
print("=" * 55)
