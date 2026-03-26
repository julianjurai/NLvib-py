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
Excitation   : F = 1e-3 at DOF 0, harmonic 1
Frequency    : omega in [0.8, 1.6] rad/s, n_harmonics = 7

Outputs
-------
examples/05_geometric_nonlinearity/output/frf.png
  - FRF showing hardening branch

Printed summary
---------------
- Peak amplitude
- Frequency at peak
- Hardening ratio  omega_peak / omega_linear
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

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

FORCE_AMP    = 1e-3              # MATLAB: exc_lev=[3e-4,5e-4,1e-3,3e-3]; use single level 1e-3
OMEGA_MIN    = 0.8               # MATLAB: Om_e=0.8 (sweeps down from 1.6; Python sweeps up)
OMEGA_MAX    = 1.6               # MATLAB: Om_s=1.6
N_HARMONICS  = 7                 # MATLAB: H=7 (was 3)

system = MechanicalSystem(M_mat, D_mat, K_mat)

# Polynomial nonlinearity for DOF 0:
# fnl0 = 3*om1^2/2*q0^2 + om1^2/2*q1^2 + om2^2*q0*q1
#       + (om1^2+om2^2)/2*q0^3 + (om1^2+om2^2)/2*q0*q1^2
# dof_indices=[0,1] → q_local = [q0, q1]
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
# dof_indices=[1,0] → q_local = [q1, q0]
_exp1 = np.array([[2, 0], [0, 2], [1, 1], [3, 0], [1, 2]], dtype=np.intp)
_coeff1 = np.array([
    3 * om2**2 / 2,
    om2**2 / 2,
    om1**2,
    (om1**2 + om2**2) / 2,
    (om1**2 + om2**2) / 2,
])
system.add_nonlinear_element(polynomial_stiffness(_exp1, _coeff1, np.array([1, 0], dtype=np.intp)))

excitation = {"dof": 0, "amplitude": FORCE_AMP, "harmonic": 1}

# ---------------------------------------------------------------------------
# Linear natural frequency of DOF 0 (for hardening ratio)
# ---------------------------------------------------------------------------
K_dense = system.K.toarray()
M_dense = system.M.toarray()
eigvals, _ = eigh(K_dense, M_dense)
omega_linear = float(np.sqrt(np.abs(eigvals[0])))  # first natural frequency

# ---------------------------------------------------------------------------
# Initial condition: linear solution at omega_start
# ---------------------------------------------------------------------------
omega_start = OMEGA_MIN
n_dof = system.n_dof
n_total = n_dof * (2 * N_HARMONICS + 1)
Q0 = np.zeros(n_total, dtype=np.float64)

# Warm up: refine Q0 at omega_start with a few Newton steps
for _newton in range(30):
    R, J = hb_residual(Q0, omega_start, system, N_HARMONICS, excitation)
    if np.linalg.norm(R) < 1e-10:
        break
    try:
        dQ = np.linalg.solve(J, -R)
    except np.linalg.LinAlgError:
        dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
    Q0 += dQ

# ---------------------------------------------------------------------------
# Arc-length continuation
# ---------------------------------------------------------------------------
def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    return hb_residual(Q, omega, system, N_HARMONICS, excitation)

solver = ContinuationSolver()
opts = ContinuationOptions(
        verbose=True,
    ds_initial=0.01,
    ds_min=1e-6,
    ds_max=0.05,
    max_steps=800,
    newton_tol=1e-8,
    lambda_min=OMEGA_MIN,
    lambda_max=OMEGA_MAX,
)
result = solver.run(residual_fn, Q0, omega_start, opts)

print(f"Continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract omega and amplitude at DOF 0, harmonic 1
# ---------------------------------------------------------------------------
solutions = result.solutions        # shape (n_steps, n_total + 1)
omegas    = solutions[:, -1]        # last column = lambda = omega

# Cosine coefficient of harmonic 1 at DOF 0:  block index = 2*1-1 = 1  → index n_dof*1 + 0
# Sine  coefficient of harmonic 1 at DOF 0:  block index = 2*1   = 2  → index n_dof*2 + 0
cos1_dof0 = solutions[:, n_dof * 1 + 0]
sin1_dof0 = solutions[:, n_dof * 2 + 0]
amp_dof0  = np.sqrt(cos1_dof0**2 + sin1_dof0**2)

stability = result.stability

# ---------------------------------------------------------------------------
# Peak amplitude and hardening ratio
# ---------------------------------------------------------------------------
if len(amp_dof0) > 0:
    peak_idx   = int(np.argmax(amp_dof0))
    peak_amp   = float(amp_dof0[peak_idx])
    peak_omega = float(omegas[peak_idx])
else:
    peak_amp   = float("nan")
    peak_omega = float("nan")

hardening_ratio = peak_omega / omega_linear if omega_linear > 0 else float("nan")

# ---------------------------------------------------------------------------
# Plot FRF
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Separate stable / unstable segments
for i in range(len(omegas) - 1):
    seg_omega = omegas[i:i+2]
    seg_amp   = amp_dof0[i:i+2]
    is_stable = not bool(stability[i])
    color     = "tab:blue" if is_stable else "tab:red"
    ls        = "-" if is_stable else "--"
    ax.plot(seg_omega, seg_amp, color=color, linestyle=ls, linewidth=1.5)

# Legend proxies
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="tab:blue", linestyle="-",  label="stable"),
    Line2D([0], [0], color="tab:red",  linestyle="--", label="unstable"),
]
ax.legend(handles=handles)

ax.set_xlabel(r"Excitation frequency $\Omega$ (rad/s)")
ax.set_ylabel(r"Amplitude $|q_0|$ (harmonic 1)")
ax.set_title("Example 05 — Geometric Nonlinearity (hardening FRF)")
ax.axvline(peak_omega, color="gray", linestyle=":", linewidth=0.8, label=f"peak Ω={peak_omega:.3f}")

frf_path = OUTPUT_DIR / "frf.png"
fig.tight_layout()
fig.savefig(frf_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  Example 05 — Geometric Nonlinearity Summary")
print("=" * 50)
print(f"  Linear natural frequency (mode 1) : {omega_linear:.4f} rad/s")
print(f"  Peak amplitude (DOF 0, harmonic 1): {peak_amp:.6f}")
print(f"  Frequency at peak                 : {peak_omega:.4f} rad/s")
print(f"  Hardening ratio  omega_peak/omega1: {hardening_ratio:.4f}")
print("=" * 50)
