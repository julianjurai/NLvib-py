"""
Example 08 — FE Euler-Bernoulli Beam with Cubic Spring, NMA Backbone.

Same beam as Example 07 but with a cubic spring at the midpoint and NO
linear damping.  Uses `hb_residual_nma` with arc-length continuation to
trace the nonlinear backbone curve (frequency vs. modal amplitude).

System parameters
-----------------
n_elements = 10
L = 1.0 m
E = 2.1e11 Pa
I_area = 1e-8 m^4
rho = 7800 kg/m^3
A = 1e-4 m^2
bc = "clamped-free"
Damping = 0 (no linear damping)

Nonlinearity: cubic_spring(k3=1e12) at midpoint node 5, dof_type="w"
NMA backbone via hb_residual_nma, n_harmonics = 3

Outputs
-------
examples/08_beam_cubic_spring_nma/output/backbone.png
  - Backbone curve: frequency vs. tip amplitude
examples/08_beam_cubic_spring_nma/output/mode_shape.png
  - Mode shape at a mid-amplitude point on the backbone

Printed summary
---------------
- Backbone frequency range
- Backbone amplitude range

Implementation notes
--------------------
Two-level reduction strategy to make the NMA tractable:

1. Galerkin modal reduction: the full 10-element beam (n_dof=20) is projected
   onto its mass-normalised first mode shape φ₁, giving an equivalent SDOF
   system with m=1, k=ω₁², k₃_modal = k₃·φ₁[mid]⁴.

2. Continuation with omega as the parameter (lambda = omega): the NMA
   residual hb_residual_nma([Q; omega]) returns n_total+1 = 8 equations
   for the augmented state.  We drop the phase-constraint row (it is
   automatically satisfied by the phase normalisation) and treat Q (length
   n_total=7) as the state vector `x`, with omega as lambda.  This maps
   naturally onto the continuation solver's `(x, lambda)` interface.

   With n_total=7 and a SDOF NMA call taking ~0.024s, each continuation step
   takes ~0.3s, making a 100-step backbone trace feasible.

The mode shape at each backbone point is recovered via q = η·φ₁ where η
is the modal amplitude from the NMA computation.
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

from nlvib.systems.fe_beam import FE_EulerBernoulliBeam
from nlvib.systems.oscillators import SingleMassOscillator
from nlvib.nonlinearities.elements import cubic_spring
from nlvib.solvers.harmonic_balance import hb_residual_nma
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Full beam parameters
# ---------------------------------------------------------------------------
N_ELEMENTS  = 19               # MATLAB: 10 (n_nodes=20 → n_elements=19)
L_BEAM      = 0.7              # MATLAB: 1.0 (len=0.7)
E_MOD       = 2.05e11          # MATLAB: 2.1e11 (E=2.05e11)
I_AREA      = 3.201e-9         # MATLAB: 1e-8 (I=0.014*0.014^3/12≈3.201e-9)
RHO         = 7800.0
A_SECT      = 1.96e-4          # MATLAB: 1e-4 (A=0.014^2=1.96e-4)
BC          = "clamped-free"

K3_CUBIC    = 6e9              # MATLAB: 1e12 (knl=6e9)
CUBIC_NODE  = 19               # MATLAB: 5 (free end node = N_ELEMENTS = 19)

N_HARMONICS = 5                # MATLAB: 3 (H=5)

# ---------------------------------------------------------------------------
# Build the full beam and extract the first mass-normalised mode shape
# ---------------------------------------------------------------------------
beam_full = FE_EulerBernoulliBeam(
    n_elements=N_ELEMENTS,
    L=L_BEAM,
    E=E_MOD,
    I_area=I_AREA,
    rho=RHO,
    A=A_SECT,
    bc=BC,
)

K_dense = beam_full.K.toarray()
M_dense = beam_full.M.toarray()
eigvals, eigvecs = eigh(K_dense, M_dense)

omega1_linear = float(np.sqrt(np.abs(eigvals[0])))
phi1_raw      = eigvecs[:, 0].real
mass_norm     = float(np.sqrt(phi1_raw @ M_dense @ phi1_raw))
phi1          = phi1_raw / mass_norm   # mass-normalised mode shape

midpoint_dof  = beam_full.find_dof(CUBIC_NODE, "w")
phi1_mid      = float(phi1[midpoint_dof])
tip_dof       = beam_full.find_dof(N_ELEMENTS, "w")
phi1_tip      = float(phi1[tip_dof])

print(f"Full beam: n_dof={beam_full.n_dof}, omega1={omega1_linear:.2f} rad/s")
print(f"  phi1 at midpoint (node {CUBIC_NODE}): {phi1_mid:.6f}")
print(f"  phi1 at tip (node {N_ELEMENTS}):      {phi1_tip:.6f}")

# Projected cubic stiffness
k3_modal = K3_CUBIC * (phi1_mid ** 4)
print(f"  k3_modal (projected) = {k3_modal:.4e}")

# ---------------------------------------------------------------------------
# Build SDOF modal surrogate (no damping for NMA)
# ---------------------------------------------------------------------------
sdof = SingleMassOscillator(m=1.0, d=0.0, k=omega1_linear**2)
sdof.add_nonlinear_element(cubic_spring(k3=k3_modal, dof_index=0))

n_dof_sdof = sdof.n_dof   # = 1
n_total    = n_dof_sdof * (2 * N_HARMONICS + 1)   # = 7 for H=3
print(f"\nSDOF NMA: n_total={n_total}, state dim={n_total+1}")

# ---------------------------------------------------------------------------
# NMA residual with omega as the continuation parameter
#
# hb_residual_nma takes z = [Q; omega] of length n_total+1 and returns
# R of length n_total+1 (physical equations + phase constraint).
#
# We use lambda = omega (the continuation parameter) and x = Q (length n_total).
# The phase constraint row (index n_total in R) is automatically satisfied by
# the NMA phase normalisation; dropping it reduces R to length n_total = 7,
# matching x.  The Jacobian w.r.t. Q is the (n_total × n_total) upper-left block.
# ---------------------------------------------------------------------------

def nma_residual_omega_param(
    Q: np.ndarray, omega: float
) -> tuple[np.ndarray, np.ndarray]:
    """NMA residual with omega as the continuation parameter.

    Parameters
    ----------
    Q : ndarray, shape (n_total,)
        Fourier coefficient vector.
    omega : float
        Excitation / natural frequency (continuation parameter).

    Returns
    -------
    R : ndarray, shape (n_total,)
        Physical NMA residual (phase constraint row dropped).
    J : ndarray, shape (n_total, n_total)
        Jacobian dR/dQ.
    """
    z = np.append(Q, omega)
    R_full, J_full = hb_residual_nma(z, sdof, N_HARMONICS)

    # Drop phase constraint row (index n_total)
    R = np.delete(R_full, n_total)        # shape (n_total,)
    J = J_full[:n_total, :n_total]        # shape (n_total, n_total)

    return R, J


# ---------------------------------------------------------------------------
# Initial point: near-linear regime (very small Q_s1)
# ---------------------------------------------------------------------------
INITIAL_MODAL_AMP = 1e-8   # modal amplitude — well within linear regime

Q0 = np.zeros(n_total, dtype=np.float64)
Q0[n_dof_sdof * 2] = INITIAL_MODAL_AMP  # Q_s1 (sine h=1)
omega0 = omega1_linear

# Refine with Newton
for _newton in range(30):
    R_c, J_c = nma_residual_omega_param(Q0, omega0)
    if np.linalg.norm(R_c) < 1e-10:
        break
    try:
        dQ = np.linalg.solve(J_c, -R_c)
    except np.linalg.LinAlgError:
        dQ = np.linalg.lstsq(J_c, -R_c, rcond=None)[0]
    Q0 += dQ

R_init, _ = nma_residual_omega_param(Q0, omega0)
print(f"Refined initial residual: {np.linalg.norm(R_init):.3e}")
print(f"Initial Q_s1 (modal amp): {Q0[n_dof_sdof * 2]:.3e}")

# ---------------------------------------------------------------------------
# Arc-length continuation along the backbone
# (omega runs from omega1_linear upward — hardening backbone)
# ---------------------------------------------------------------------------
OMEGA_MAX_BACKBONE = omega1_linear * 2.0   # trace to 2x linear frequency

solver = ContinuationSolver()
opts = ContinuationOptions(
        verbose=True,
    ds_initial=0.01,
    ds_min=1e-8,
    ds_max=2.0,
    max_steps=200,
    newton_tol=1e-8,
    max_newton_iter=20,
    lambda_min=omega1_linear * 0.9,
    lambda_max=OMEGA_MAX_BACKBONE,
)

result = solver.run(nma_residual_omega_param, Q0, omega0, opts)

print(f"\nNMA continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract backbone
# ---------------------------------------------------------------------------
solutions   = result.solutions   # (n_steps, n_total + 1)
# solutions[:, n_total] = lambda = omega
omega_backbone = solutions[:, n_total]

# Fundamental harmonic amplitude (harmonic 1)
cos1_modal = solutions[:, n_dof_sdof * 1]   # Q_c1 (≈0 by phase constraint)
sin1_modal = solutions[:, n_dof_sdof * 2]   # Q_s1
amp_modal  = np.sqrt(cos1_modal**2 + sin1_modal**2)   # modal coordinate amplitude

# Physical tip amplitude
amp_tip = amp_modal * abs(phi1_tip)

valid_mask = (omega_backbone > 0.5 * omega1_linear) & np.isfinite(omega_backbone)
omega_bb = omega_backbone[valid_mask]
amp_bb   = amp_tip[valid_mask]

if len(omega_bb) > 1:
    freq_min = float(np.min(omega_bb))
    freq_max = float(np.max(omega_bb))
    amp_min  = float(np.min(amp_bb))
    amp_max  = float(np.max(amp_bb))
else:
    freq_min = freq_max = float("nan")
    amp_min  = amp_max  = float("nan")

# ---------------------------------------------------------------------------
# Plot: backbone curve (frequency vs. tip amplitude)
# ---------------------------------------------------------------------------
fig_bb, ax_bb = plt.subplots(figsize=(8, 5))
ax_bb.plot(amp_bb, omega_bb, color="tab:blue", linewidth=2.0,
           label=f"NMA backbone (H={N_HARMONICS}, modal reduction)")
ax_bb.axhline(omega1_linear, color="gray", linestyle="--", linewidth=0.8,
              label=f"linear ω₁ = {omega1_linear:.1f} rad/s")
ax_bb.set_xlabel(r"Tip amplitude $|w_{\rm tip}|$ (m)")
ax_bb.set_ylabel(r"Natural frequency $\omega_n$ (rad/s)")
ax_bb.set_title("Example 08 — Beam Cubic Spring NMA Backbone Curve")
ax_bb.legend()
ax_bb.grid(True, alpha=0.3)

bb_path = OUTPUT_DIR / "backbone.png"
fig_bb.tight_layout()
fig_bb.savefig(bb_path, dpi=150)
plt.close(fig_bb)
print(f"\nPlot saved: {bb_path}")

# ---------------------------------------------------------------------------
# Mode shape at mid-amplitude backbone point
# Recovery: q_physical = eta * phi1  (single-mode assumption)
# ---------------------------------------------------------------------------
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) >= 3:
    mid_sol_idx = valid_indices[len(valid_indices) // 2]
elif len(valid_indices) > 0:
    mid_sol_idx = valid_indices[0]
else:
    mid_sol_idx = 0

eta_mid   = float(amp_modal[mid_sol_idx])   # modal amplitude (η̂)
omega_mid = float(omega_backbone[mid_sol_idx])
q_physical = eta_mid * phi1   # physical displacement, length n_dof=20

node_positions = [0.0]   # clamped root
mode_disp      = [0.0]
for node_i in range(1, N_ELEMENTS + 1):
    try:
        r_dof = beam_full.find_dof(node_i, "w")
        node_positions.append(node_i * L_BEAM / N_ELEMENTS)
        mode_disp.append(float(q_physical[r_dof]))
    except ValueError:
        pass

node_positions_arr = np.array(node_positions)
mode_disp_arr      = np.array(mode_disp)

fig_mode, ax_mode = plt.subplots(figsize=(8, 4))
ax_mode.plot(node_positions_arr, np.zeros_like(node_positions_arr), "k--",
             linewidth=0.6, label="undeformed")
ax_mode.plot(node_positions_arr, mode_disp_arr, "o-", color="tab:green",
             linewidth=1.5, label=f"mode shape at ω={omega_mid:.1f} rad/s")
ax_mode.fill_between(node_positions_arr, mode_disp_arr, alpha=0.2, color="tab:green")
ax_mode.set_xlabel("Position along beam (m)")
ax_mode.set_ylabel("Transverse displacement (m)")
ax_mode.set_title("Example 08 — NMA Mode Shape at Mid-Amplitude Backbone Point")
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
print("\n" + "=" * 65)
print("  Example 08 — Beam Cubic Spring NMA Summary")
print("=" * 65)
print(f"  Linear 1st natural frequency  : {omega1_linear:.2f} rad/s")
print(f"  Backbone frequency range      : [{freq_min:.2f}, {freq_max:.2f}] rad/s")
print(f"  Backbone tip amplitude range  : [{amp_min:.3e}, {amp_max:.3e}] m")
print(f"  k3_modal (Galerkin projected) : {k3_modal:.4e}")
print(f"  Method: single-mode Galerkin reduction + omega-continuation, H={N_HARMONICS}")
print("=" * 65)
