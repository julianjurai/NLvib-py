"""
Example 06 — Multi-DOF Multi-Nonlinearity FRF.

4-DOF chain of oscillators with two different nonlinear elements:
  - cubic_spring(k3=0.5)           at DOF 0
  - tanh_dry_friction(f0=0.2, c=8) at DOF 2

Demonstrates FRF for all DOFs and Newton convergence history.

System parameters
-----------------
masses      = [1.0, 1.5, 1.0, 0.8]
stiffnesses = [1.0, 0.8, 1.2, 0.6, 1.0]
dampings    = [0.02] * 5
Excitation  : F = 0.1 at DOF 0, harmonic 1
Frequency   : omega in [0.3, 1.5] rad/s, n_harmonics = 3

Outputs
-------
examples/06_multi_dof_multi_nl/output/frf_all_dofs.png
  - FRF for all 4 DOFs (4 subplots)
examples/06_multi_dof_multi_nl/output/convergence.png
  - Step-size history as a proxy for Newton convergence

Printed summary
---------------
- Peak amplitude for each DOF
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.nonlinearities.elements import cubic_spring, tanh_dry_friction
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------
MASSES      = [1.0, 1.5, 1.0, 0.8]
STIFFNESSES = [1.0, 0.8, 1.2, 0.6, 1.0]
DAMPINGS    = [0.02, 0.02, 0.02, 0.02, 0.02]
FORCE_AMP   = 0.1
OMEGA_MIN   = 0.3
OMEGA_MAX   = 1.5
N_HARMONICS = 3

system = ChainOfOscillators(
    masses=MASSES,
    stiffnesses=STIFFNESSES,
    dampings=DAMPINGS,
)
system.add_nonlinear_element(cubic_spring(k3=0.5, dof_index=0))
system.add_nonlinear_element(tanh_dry_friction(f0=0.2, c=8.0, dof_index=2))

excitation = {"dof": 0, "amplitude": FORCE_AMP, "harmonic": 1}

# ---------------------------------------------------------------------------
# Initial solution at omega_start
# ---------------------------------------------------------------------------
n_dof   = system.n_dof
omega_start = OMEGA_MIN
n_total = n_dof * (2 * N_HARMONICS + 1)
Q0      = np.zeros(n_total, dtype=np.float64)

for _newton in range(40):
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
    ds_initial=0.01,
    ds_min=1e-6,
    ds_max=0.05,
    max_steps=1000,
    newton_tol=1e-8,
    lambda_min=OMEGA_MIN,
    lambda_max=OMEGA_MAX,
)
result = solver.run(residual_fn, Q0, omega_start, opts)

print(f"Continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract omega and amplitudes for each DOF (harmonic 1)
# ---------------------------------------------------------------------------
solutions = result.solutions  # (n_steps, n_total + 1)
omegas    = solutions[:, -1]
stability = result.stability

# Harmonic 1 amplitude for each DOF:
# cos1 block starts at n_dof * 1, sin1 block at n_dof * 2
amplitudes = np.zeros((n_dof, len(omegas)), dtype=np.float64)
for dof in range(n_dof):
    cos1 = solutions[:, n_dof * 1 + dof]
    sin1 = solutions[:, n_dof * 2 + dof]
    amplitudes[dof, :] = np.sqrt(cos1**2 + sin1**2)

# ---------------------------------------------------------------------------
# Plot: FRF for all DOFs
# ---------------------------------------------------------------------------
fig_frf, axes_frf = plt.subplots(n_dof, 1, figsize=(8, 10), sharex=True)

for dof in range(n_dof):
    ax = axes_frf[dof]
    amp = amplitudes[dof, :]
    for i in range(len(omegas) - 1):
        is_stable = not bool(stability[i])
        color = "tab:blue" if is_stable else "tab:red"
        ls    = "-" if is_stable else "--"
        ax.plot(omegas[i:i+2], amp[i:i+2], color=color, linestyle=ls, linewidth=1.2)
    ax.set_ylabel(f"$|q_{dof}|$")
    ax.set_title(f"DOF {dof}")

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="tab:blue", linestyle="-",  label="stable"),
    Line2D([0], [0], color="tab:red",  linestyle="--", label="unstable"),
]
axes_frf[0].legend(handles=handles, loc="upper left")
axes_frf[-1].set_xlabel(r"Excitation frequency $\Omega$ (rad/s)")
fig_frf.suptitle("Example 06 — Multi-DOF Multi-NL FRF (all DOFs)")
fig_frf.tight_layout()
frf_path = OUTPUT_DIR / "frf_all_dofs.png"
fig_frf.savefig(frf_path, dpi=150)
plt.close(fig_frf)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Plot: convergence proxy (ds_history)
# ---------------------------------------------------------------------------
fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ds_history = result.ds_history[1:]  # skip the initial 0.0 entry
step_indices = np.arange(1, len(ds_history) + 1)
ax_conv.semilogy(step_indices, ds_history, color="tab:green", linewidth=1.2)
ax_conv.set_xlabel("Continuation step")
ax_conv.set_ylabel("Arc-length step size ds")
ax_conv.set_title("Example 06 — Newton Convergence Proxy (ds history)")
ax_conv.grid(True, which="both", alpha=0.4)
fig_conv.tight_layout()
conv_path = OUTPUT_DIR / "convergence.png"
fig_conv.savefig(conv_path, dpi=150)
plt.close(fig_conv)
print(f"Plot saved: {conv_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  Example 06 — Multi-DOF Multi-NL Peak Amplitudes")
print("=" * 55)
for dof in range(n_dof):
    amp = amplitudes[dof, :]
    if len(amp) > 0:
        peak_idx   = int(np.argmax(amp))
        peak_amp   = float(amp[peak_idx])
        peak_omega = float(omegas[peak_idx])
        print(f"  DOF {dof}: peak amplitude = {peak_amp:.6f}  at omega = {peak_omega:.4f} rad/s")
    else:
        print(f"  DOF {dof}: no data")
print("=" * 55)
