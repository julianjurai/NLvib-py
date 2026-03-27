"""
Example 05 — Geometric Nonlinearity (hardening FRF).

2-DOF system with geometric (polynomial) stiffness nonlinearity in modal coordinates.
Matches MATLAB 06_twoSprings_geometricNonlinearity.m.
Demonstrates hardening-type frequency response via HB + arc-length continuation.

System parameters (MATLAB: 06_twoSprings_geometricNonlinearity.m)
-----------------
om1=1.13, om2=2, zt1=1e-3, zt2=5e-3
M=eye(2), K=diag([om1^2, om2^2]), D=diag([2*zt1*om1, 2*zt2*om2])
Polynomial nonlinearity on both DOFs (geometric coupling)
Excitation   : F=[1;1]*exc_lev, harmonic 1; exc_lev in [3e-4, 5e-4, 1e-3, 3e-3]
Frequency    : arc-length continuation traces full branch including fold,
               starting at omega=0.8, upper limit omega=1.6; n_harmonics=7

Outputs
-------
examples/05_geometric_nonlinearity/output/frf.png
  - FRF with 4 excitation levels overlaid (green curves), xlim=[0.8, 1.6]
  - Matches MATLAB plot style: all green, no stable/unstable coloring

Printed summary
---------------
- Peak amplitude per excitation level
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package is importable when run as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nlvib.systems.base import MechanicalSystem
from nlvib.nonlinearities.elements import polynomial_stiffness
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System definition
# MATLAB: 06_twoSprings_geometricNonlinearity.m
# om1=1.13, om2=2, zt1=1e-3, zt2=5e-3
# M=eye(2), K=diag([om1^2,om2^2]), D=diag([2*zt1*om1, 2*zt2*om2])
# NOT a chain of oscillators — diagonal (uncoupled modal) matrices
# ---------------------------------------------------------------------------
om1 = 1.13                       # MATLAB: first modal frequency
om2 = 2.0                        # MATLAB: second modal frequency
zt1 = 1e-3                       # MATLAB: first modal damping ratio
zt2 = 5e-3                       # MATLAB: second modal damping ratio

M_mat = np.diag([1.0, 1.0])
K_mat = np.diag([om1**2, om2**2])
D_mat = np.diag([2 * zt1 * om1, 2 * zt2 * om2])

# MATLAB: Om_e=0.8 (start of x-axis), Om_s=1.6 (end of x-axis)
OMEGA_LOW    = 0.8               # low-frequency end of plot
OMEGA_HIGH   = 1.6               # high-frequency end of plot
N_HARMONICS  = 7                 # MATLAB: H=7

# MATLAB: exc_lev = [3e-4 5e-4 1e-3 3e-3], Fex1=[1;1]
EXC_LEVELS   = [3e-4, 5e-4, 1e-3, 3e-3]
Fex1_dir     = np.array([1.0, 1.0])  # MATLAB: Fex1=[1;1]

system = MechanicalSystem(M_mat, D_mat, K_mat)

# Polynomial nonlinearity for DOF 0:
# fnl0 = 3*om1^2/2*q0^2 + om1^2/2*q1^2 + om2^2*q0*q1
#       + (om1^2+om2^2)/2*q0^3 + (om1^2+om2^2)/2*q0*q1^2
# dof_indices=[0,1] -> q_local = [q0, q1]
_exp0 = np.array([[2, 0], [0, 2], [1, 1], [3, 0], [1, 2]], dtype=np.intp)
_coeff0 = np.array([
    3 * om1**2 / 2,
    om1**2 / 2,
    om2**2,
    (om1**2 + om2**2) / 2,
    (om1**2 + om2**2) / 2,
])
system.add_nonlinear_element(polynomial_stiffness(_exp0, _coeff0, np.array([0, 1], dtype=np.intp)))

# Polynomial nonlinearity for DOF 1:
# fnl1 = 3*om2^2/2*q1^2 + om2^2/2*q0^2 + om1^2*q0*q1
#       + (om1^2+om2^2)/2*q1^3 + (om1^2+om2^2)/2*q0^2*q1
# dof_indices=[1,0] -> q_local = [q1, q0]
_exp1 = np.array([[2, 0], [0, 2], [1, 1], [3, 0], [1, 2]], dtype=np.intp)
_coeff1 = np.array([
    3 * om2**2 / 2,
    om2**2 / 2,
    om1**2,
    (om1**2 + om2**2) / 2,
    (om1**2 + om2**2) / 2,
])
system.add_nonlinear_element(polynomial_stiffness(_exp1, _coeff1, np.array([1, 0], dtype=np.intp)))

n_dof = system.n_dof
n_total = n_dof * (2 * N_HARMONICS + 1)


# ---------------------------------------------------------------------------
# Arc-length continuation for each excitation level.
# The solver starts at omega=OMEGA_LOW=0.8 with a linear-system initial guess
# and sweeps upward (t_lam = +1).  The arc-length predictor-corrector
# automatically traces around the hardening fold, capturing the full branch
# including the unstable segment.  lambda_max=OMEGA_HIGH+margin stops the
# continuation once past the fold; lambda_min is not set so the solver can
# come back below OMEGA_LOW if the fold extends there.
# ---------------------------------------------------------------------------
def _run_frf(force_amp: float) -> tuple[np.ndarray, np.ndarray]:
    """Run HB arc-length continuation for a given force amplitude.

    Returns (omegas, amp_dof0) where amp_dof0 is the fundamental-harmonic
    magnitude of DOF 0, matching MATLAB: sqrt(Q(n+1)^2 + Q(2n+1)^2).
    """
    # Build combined excitation vector: Fex1=[1;1]*exc_lev applied as cosine at harmonic 1
    # hb_residual accepts a pre-built F_ext vector of shape (n_total,)
    F_ext = np.zeros(n_total, dtype=np.float64)
    cosine_block_start = (2 * 1 - 1) * n_dof  # harmonic 1, cosine block
    F_ext[cosine_block_start + 0] = force_amp  # DOF 0
    F_ext[cosine_block_start + 1] = force_amp  # DOF 1

    def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
        # Pass combined excitation vector (single call avoids doubling system matrices)
        return hb_residual(Q, omega, system, N_HARMONICS, F_ext)

    # Initial guess: linear solution at omega_start = OMEGA_LOW
    # MATLAB uses Q1 = (-Om_e^2*M + 1i*Om_e*D + K)\(Fex1*exc_lev) for upward sweep
    omega_start = OMEGA_LOW
    Fex_vec = Fex1_dir * force_amp
    Q1_complex = np.linalg.solve(
        -(omega_start**2) * M_mat + 1j * omega_start * D_mat + K_mat, Fex_vec
    )
    Q0 = np.zeros(n_total, dtype=np.float64)
    Q0[n_dof * 1 + 0] = float(np.real(Q1_complex[0]))   # cos1, DOF 0
    Q0[n_dof * 2 + 0] = -float(np.imag(Q1_complex[0]))  # sin1, DOF 0
    Q0[n_dof * 1 + 1] = float(np.real(Q1_complex[1]))   # cos1, DOF 1
    Q0[n_dof * 2 + 1] = -float(np.imag(Q1_complex[1]))  # sin1, DOF 1

    # Warm up with Newton iterations at omega_start using combined residual
    for _newton in range(30):
        R, J = residual_fn(Q0, omega_start)
        if np.linalg.norm(R) < 1e-10:
            break
        try:
            dQ = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
        Q0 += dQ

    solver = ContinuationSolver()
    opts = ContinuationOptions(
        verbose=False,
        ds_initial=0.005,
        ds_min=1e-7,
        ds_max=0.005,   # MATLAB: ds=0.005 (fixed step) — ensures peak error < 1%
        max_steps=5000,
        newton_tol=1e-8,
        max_newton_iter=25,
        # Sweep upward from 0.8; no lambda_min so the arc-length can come back
        # through the fold below 0.8 if needed.  Stop when lambda exceeds 1.6.
        lambda_min=None,
        lambda_max=OMEGA_HIGH + 0.05,  # small margin past the fold
    )
    result = solver.run(residual_fn, Q0, omega_start, opts)
    print(f"  exc_lev={force_amp:.0e}: {result.n_steps} steps, {result.message}")

    solutions = result.solutions  # (n_steps, n_total + 1)
    omegas    = solutions[:, -1]
    # Fundamental harmonic amplitude of DOF 0: sqrt(cos1^2 + sin1^2)
    # MATLAB: a{iex} = sqrt(Q(n+1,:).^2 + Q(2*n+1,:).^2)  (1-based: n+1=DOF0 cos, 2n+1=DOF0 sin)
    cos1 = solutions[:, n_dof * 1 + 0]
    sin1 = solutions[:, n_dof * 2 + 0]
    amp  = np.sqrt(cos1**2 + sin1**2)
    return omegas, amp


# ---------------------------------------------------------------------------
# Run for all excitation levels
# ---------------------------------------------------------------------------
print("Running HB continuation for 4 excitation levels...")
results_om  = []
results_amp = []
for exc in EXC_LEVELS:
    om_arr, amp_arr = _run_frf(exc)
    results_om.append(om_arr)
    results_amp.append(amp_arr)

# ---------------------------------------------------------------------------
# Plot FRF — matches MATLAB style:
#   all curves green, no stable/unstable coloring, xlim=[0.8, 1.6]
#   xlabel='excitation frequency', ylabel='response amplitude |Q_{0,1}|'
#   legend upper left
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for i, (om_arr, amp_arr) in enumerate(zip(results_om, results_amp)):
    ax.plot(om_arr, amp_arr, color="g", linewidth=1.5, label=f"exc lev {i+1}")

ax.set_xlabel("excitation frequency")
ax.set_ylabel("response amplitude |Q_{1,1}|")
ax.set_xlim([1.085, 1.15])   # MATLAB: set(gca,'xlim',[1.085 1.15])
ax.set_ylim([0, 0.35])        # MATLAB: set(gca,'ylim',[0 .35])
ax.legend(loc="upper left")
ax.grid(True, alpha=0.4)

frf_path = OUTPUT_DIR / "frf.png"
fig.tight_layout()
fig.savefig(frf_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  Example 05 — Geometric Nonlinearity Peak Amplitudes")
print("=" * 55)
for i, (om_arr, amp_arr) in enumerate(zip(results_om, results_amp)):
    if len(amp_arr) > 0:
        peak_idx   = int(np.argmax(amp_arr))
        peak_amp   = float(amp_arr[peak_idx])
        peak_omega = float(om_arr[peak_idx])
        print(f"  exc_lev={EXC_LEVELS[i]:.0e}: peak amp = {peak_amp:.6f}  at omega = {peak_omega:.4f} rad/s")
    else:
        print(f"  exc_lev={EXC_LEVELS[i]:.0e}: no data")
print("=" * 55)
