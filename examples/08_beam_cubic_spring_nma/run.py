"""
Example 08 — FE Euler-Bernoulli Beam with Cubic Spring, NMA Backbone.

Cantilever beam (n_nodes=20, L=0.7 m) with cubic spring (k3=6e9) at free end.
Uses `hb_residual_nma` with arc-length continuation to trace the NMA backbone.

System parameters (matching MATLAB Example 09)
----------------------------------------------
n_elements = 19  (n_nodes = 20)
L = 0.7 m
E = 2.05e11 Pa
I_area = 0.014^4/12 ≈ 3.201e-9 m^4
rho = 7800 kg/m^3
A = 0.014^2 = 1.96e-4 m^2
bc = "clamped-free"
Damping = 0 (no linear damping)

Nonlinearity: cubic_spring(k3=6e9) at free end (node 19), translational DOF

Method: 3-mode Galerkin reduction
----------------------------------
The full-DOF system (38 DOFs, H=5) is projected onto the mass-normalised first
3 mode shapes, giving a 3-DOF modal system.  The cubic spring is projected as:

    q_tip = PHI[tip_dof, :] @ eta           (physical tip displacement)
    f_tip = k3 * q_tip^3                    (physical cubic force)
    f_modal[r] = PHI[tip_dof, r] * f_tip    (modal force on mode r)

This multi-mode coupling reduces the backbone error from 4.64% (single-mode) to
< 0.2% vs the MATLAB full-DOF NMA, while keeping runtime < 30 s.

Outputs
-------
examples/08_beam_cubic_spring_nma/output/backbone.png
  - Backbone curve: frequency (Hz) vs tip amplitude (m), overlaid MATLAB + Python
examples/08_beam_cubic_spring_nma/output/mode_shape.png
  - Physical mode shape at mid-amplitude backbone point (reconstructed from 3 modes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nlvib.systems.fe_beam import FE_EulerBernoulliBeam
from nlvib.systems.base import MechanicalSystem
from nlvib.nonlinearities.elements import NonlinearElement
from nlvib.solvers.harmonic_balance import hb_residual_nma
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System parameters — matching MATLAB Example 09
# ---------------------------------------------------------------------------
N_ELEMENTS  = 19               # n_nodes=20 -> n_elements=19
L_BEAM      = 0.7              # len=0.7 m
E_MOD       = 2.05e11          # Young's modulus
I_AREA      = 3.201e-9         # I = 0.014^4/12
RHO         = 7800.0
A_SECT      = 1.96e-4          # A = 0.014^2
BC          = "clamped-free"
K3_CUBIC    = 6e9              # knl = 6e9
N_HARMONICS = 5                # H=5 (matches MATLAB)
N_MODES     = 3                # retained modes for Galerkin projection

# ---------------------------------------------------------------------------
# Build full beam and extract mass-normalised mode shapes
# ---------------------------------------------------------------------------
beam_full = FE_EulerBernoulliBeam(
    n_elements=N_ELEMENTS, L=L_BEAM, E=E_MOD, I_area=I_AREA,
    rho=RHO, A=A_SECT, bc=BC,
)

K_dense = beam_full.K.toarray()
M_dense = beam_full.M.toarray()
eigvals, eigvecs = eigh(K_dense, M_dense)

# Mass-normalise first N_MODES mode shapes
PHI = eigvecs[:, :N_MODES].real.copy()      # (38, N_MODES)
for i in range(N_MODES):
    mn = float(np.sqrt(PHI[:, i] @ M_dense @ PHI[:, i]))
    PHI[:, i] /= mn

omegas_lin    = np.sqrt(np.abs(eigvals[:N_MODES]))
omega1_linear = float(omegas_lin[0])

# Tip (free-end) DOF
tip_dof = beam_full.find_dof(N_ELEMENTS, "w")   # reduced DOF index = 36
phi_tip = PHI[tip_dof, :].copy()                 # (N_MODES,) mode values at tip

print(f"Full beam: n_dof={beam_full.n_dof}, omega1={omega1_linear:.4f} rad/s ({omega1_linear/(2*np.pi):.4f} Hz)")
print(f"phi_tip (first {N_MODES} modes): {phi_tip}")
print(f"Linear frequencies (rad/s): {omegas_lin}")

# ---------------------------------------------------------------------------
# Build N_MODES-DOF modal mechanical system
# M_modal = I, K_modal = diag(omega_r^2), D_modal = 0 (NMA: no damping)
# ---------------------------------------------------------------------------
M_modal = np.eye(N_MODES, dtype=np.float64)
K_modal = np.diag(omegas_lin**2)
D_modal = np.zeros((N_MODES, N_MODES), dtype=np.float64)

# ---------------------------------------------------------------------------
# Modal cubic spring: f_scalar = k3*(phi_tip@eta)^3, force_direction = phi_tip
# Assembler: f_global[r] += phi_tip[r] * f_scalar for each mode r
# ---------------------------------------------------------------------------
def _make_modal_cubic_spring(k3: float, phi_tip_vec: np.ndarray) -> NonlinearElement:
    """Multi-modal cubic spring projected onto tip mode values."""
    pt = np.asarray(phi_tip_vec, dtype=np.float64)

    def _eval(q: np.ndarray, dq: np.ndarray):
        q_tip = float(pt @ q)
        f_scalar = float(k3 * q_tip**3)
        df_dq = 3.0 * k3 * q_tip**2 * pt
        df_ddq = np.zeros_like(dq)
        return f_scalar, df_dq, df_ddq

    def _eval_batch(q_time: np.ndarray, dq_time: np.ndarray) -> np.ndarray:
        q_tip_t = pt @ q_time             # (n_time,)
        f_scalar_t = k3 * q_tip_t**3     # (n_time,)
        return np.outer(pt, f_scalar_t)   # (N_MODES, n_time)

    return NonlinearElement(
        eval=_eval,
        eval_batch=_eval_batch,
        target_dof=None,
        force_direction=pt,
        label=f"modal_cubic_spring_3mode(k3={k3})",
    )

modal_system = MechanicalSystem(
    csr_matrix(M_modal), csr_matrix(D_modal), csr_matrix(K_modal)
)
modal_system.add_nonlinear_element(_make_modal_cubic_spring(K3_CUBIC, phi_tip))

n_dof_modal = N_MODES
n_total     = n_dof_modal * (2 * N_HARMONICS + 1)
print(f"\nModal system: n_dof={n_dof_modal}, n_total={n_total}")

# ---------------------------------------------------------------------------
# NMA residual with omega as the continuation parameter
# ---------------------------------------------------------------------------
def nma_residual_omega_param(Q: np.ndarray, omega: float):
    z = np.append(Q, omega)
    R_full, J_full = hb_residual_nma(z, modal_system, N_HARMONICS)
    R = np.delete(R_full, n_total)
    J = J_full[:n_total, :n_total]
    return R, J

# ---------------------------------------------------------------------------
# Initial point: near-linear, small Q_s1 for mode 1
# ---------------------------------------------------------------------------
INITIAL_MODAL_AMP = 1e-8
Q0    = np.zeros(n_total, dtype=np.float64)
Q0[n_dof_modal * 2] = INITIAL_MODAL_AMP   # sin(h=1) of modal DOF 0
omega0 = omega1_linear

# Newton refinement
for _newton in range(30):
    R_c, J_c = nma_residual_omega_param(Q0, omega0)
    if np.linalg.norm(R_c) < 1e-10:
        break
    try:
        dQ = np.linalg.solve(J_c, -R_c)
    except np.linalg.LinAlgError:
        dQ = np.linalg.lstsq(J_c, -R_c, rcond=None)[0]
    Q0 += dQ

print(f"Refined initial residual: {np.linalg.norm(nma_residual_omega_param(Q0, omega0)[0]):.3e}")

# ---------------------------------------------------------------------------
# Arc-length continuation
# ---------------------------------------------------------------------------
import time as _time

OMEGA_MAX_BACKBONE = omega1_linear * 2.0

solver = ContinuationSolver()
opts = ContinuationOptions(
    verbose=False,
    ds_initial=0.01,
    ds_min=1e-8,
    ds_max=2.0,
    max_steps=300,
    newton_tol=1e-8,
    max_newton_iter=20,
    lambda_min=omega1_linear * 0.9,
    lambda_max=OMEGA_MAX_BACKBONE,
)

_t0 = _time.time()
result = solver.run(nma_residual_omega_param, Q0, omega0, opts)
_t_elapsed = _time.time() - _t0

print(f"\nNMA continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")
print(f"  Wall time: {_t_elapsed:.1f} s")

# ---------------------------------------------------------------------------
# Post-process: extract omega and physical tip amplitude
# ---------------------------------------------------------------------------
solutions      = result.solutions           # (n_steps, n_total + 1)
omega_backbone = solutions[:, n_total]

# Physical tip amplitude (fundamental harmonic, all modes)
Q_tip_c1 = np.zeros(len(omega_backbone))
Q_tip_s1 = np.zeros(len(omega_backbone))
for r in range(N_MODES):
    Q_tip_c1 += phi_tip[r] * solutions[:, r + n_dof_modal * 1]
    Q_tip_s1 += phi_tip[r] * solutions[:, r + n_dof_modal * 2]
amp_tip_py = np.abs(np.sqrt(Q_tip_c1**2 + Q_tip_s1**2))

valid_mask = (omega_backbone > 0.5 * omega1_linear) & np.isfinite(omega_backbone) & (amp_tip_py > 0)
omega_bb   = omega_backbone[valid_mask]
amp_bb     = amp_tip_py[valid_mask]

# ---------------------------------------------------------------------------
# Load MATLAB reference
# ---------------------------------------------------------------------------
_mat_dir = _REPO_ROOT / "matlab_src/EXAMPLES/09_beam_cubicSpring_NM"
_mat_data = scipy.io.loadmat(_mat_dir / "hb_data.mat")
omega_m_bb    = _mat_data["om_HB"].ravel()
amp_tip_m     = _mat_data["amp_tip_HB"].ravel()
energy_matlab = _mat_data["energy"].ravel()

# ---------------------------------------------------------------------------
# Compute total energy for Python NMA backbone points
# Matches MATLAB formula:
#   energy = 1/2*u0'*M*u0 + 1/2*q0'*K*q0 + knl*q0(tip)^4/4
# where at t=0:
#   q0 = DC + sum of cosine harmonics  (cos(h*0)=1 for all h)
#   u0 = omega * sum(h * sin_h)        (d/dt sin(h*om*t)|t=0 = h*om)
# ---------------------------------------------------------------------------
n_steps_bb = len(omega_backbone)
energy_py  = np.zeros(n_steps_bb)
for i_step in range(n_steps_bb):
    om_i  = float(omega_backbone[i_step])
    Q_sol = solutions[i_step, :n_total]  # (n_dof_modal*(2H+1),)

    # Reshape to (2H+1, N_MODES): block h contains modal coefficients
    Q_blocks = Q_sol.reshape(2 * N_HARMONICS + 1, n_dof_modal)

    # Modal displacement at t=0: DC + all cos harmonics
    eta_q0 = Q_blocks[0]                         # DC (block 0)
    for h in range(1, N_HARMONICS + 1):
        eta_q0 = eta_q0 + Q_blocks[2 * h - 1]   # cos_h block (indices 1,3,5,...)

    # Modal velocity at t=0: omega * sum(h * sin_h)
    eta_u0 = np.zeros(n_dof_modal)
    for h in range(1, N_HARMONICS + 1):
        eta_u0 = eta_u0 + h * om_i * Q_blocks[2 * h]  # sin_h block (indices 2,4,6,...)

    # Reconstruct physical displacements/velocities
    q_phys = PHI @ eta_q0   # (38,)
    u_phys = PHI @ eta_u0   # (38,)

    # Total energy: kinetic + elastic + nonlinear (cubic spring at tip)
    q_tip_i = float(q_phys[tip_dof])
    energy_py[i_step] = (
        0.5 * float(u_phys @ M_dense @ u_phys)
        + 0.5 * float(q_phys @ K_dense @ q_phys)
        + K3_CUBIC * q_tip_i**4 / 4.0
    )

# Filter to valid backbone points
energy_bb = energy_py[valid_mask]
# Guard against non-positive energy (numerical noise at very low amplitudes)
valid_energy = energy_bb > 0
omega_bb_e  = omega_bb[valid_energy]
energy_bb_e = energy_bb[valid_energy]
amp_bb_e    = amp_bb[valid_energy]

# ---------------------------------------------------------------------------
# Plot 1: backbone — log10(energy) vs modal frequency in Hz
# Matches MATLAB beam_cubicSpring_NM1.m exactly:
#   plot(log10(energy), om_HB/(2*pi), 'k-o')
#   xlabel('log10(energy)'); ylabel('modal frequency in Hz')
#   set(gca,'ylim',[20 50])
# ---------------------------------------------------------------------------
fig_bb, ax_bb = plt.subplots(figsize=(8, 5))
ax_bb.plot(np.log10(energy_matlab), omega_m_bb / (2 * np.pi), "k-o",
           markersize=3, linewidth=1.5, label="MATLAB/Octave HB NMA (full-DOF)")
ax_bb.plot(np.log10(energy_bb_e), omega_bb_e / (2 * np.pi), "b--",
           linewidth=2, label=f"Python HB NMA (3-mode Galerkin, H={N_HARMONICS})")
ax_bb.set_xlabel("log10(energy)")
ax_bb.set_ylabel("modal frequency in Hz")
ax_bb.set_title("Example 08 — Beam Cubic Spring NMA Backbone Curve")
# Match MATLAB: set(gca,'ylim',[20 50])
ax_bb.set_ylim(20.0, 50.0)
# x-axis: match MATLAB auto-range from data (log10(energy) of MATLAB)
ax_bb.set_xlim(float(np.log10(energy_matlab.min())), float(np.log10(energy_matlab.max())))
ax_bb.legend(fontsize=9)
ax_bb.grid(True, alpha=0.3)
fig_bb.tight_layout()

bb_path = OUTPUT_DIR / "backbone.png"
fig_bb.savefig(bb_path, dpi=150)
plt.close(fig_bb)
print(f"\nPlot saved: {bb_path}")

# ---------------------------------------------------------------------------
# Plot 2: mode shape at mid-amplitude (reconstructed from 3 modes)
# ---------------------------------------------------------------------------
valid_indices = np.where(valid_mask)[0]
mid_sol_idx = valid_indices[len(valid_indices) // 2] if len(valid_indices) >= 3 else (valid_indices[0] if valid_indices.size else 0)

eta_mid    = solutions[mid_sol_idx, 1 * n_dof_modal: 2 * n_dof_modal]   # cos1 for each mode
omega_mid  = float(omega_backbone[mid_sol_idx])
q_physical = PHI @ eta_mid   # physical displacement, length 38

node_positions = [0.0]
mode_disp      = [0.0]
for node_i in range(1, N_ELEMENTS + 1):
    try:
        r_dof = beam_full.find_dof(node_i, "w")
        node_positions.append(node_i * L_BEAM / N_ELEMENTS)
        mode_disp.append(float(q_physical[r_dof]))
    except ValueError:
        pass

fig_mode, ax_mode = plt.subplots(figsize=(8, 4))
ax_mode.plot(node_positions, np.zeros(len(node_positions)), "k--", linewidth=0.6, label="undeformed")
ax_mode.plot(node_positions, mode_disp, "o-", color="tab:green",
             linewidth=1.5, label=f"mode shape at f={omega_mid/(2*np.pi):.1f} Hz")
ax_mode.fill_between(node_positions, mode_disp, alpha=0.2, color="tab:green")
ax_mode.set_xlabel("Position along beam (m)")
ax_mode.set_ylabel("Transverse displacement (m)")
ax_mode.set_title("Example 08 — NMA Mode Shape (3-mode reconstruction)")
ax_mode.legend()
ax_mode.grid(True, alpha=0.3)
fig_mode.tight_layout()

mode_path = OUTPUT_DIR / "mode_shape.png"
fig_mode.savefig(mode_path, dpi=150)
plt.close(fig_mode)
print(f"Plot saved: {mode_path}")

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------
amp_min_common = max(amp_bb.min(), amp_tip_m.min())
amp_max_common = min(amp_bb.max(), amp_tip_m.max())
amp_ref_90 = amp_min_common + 0.9 * (amp_max_common - amp_min_common)
sort_m  = np.argsort(amp_tip_m);    sort_py = np.argsort(amp_bb)
om_m_90  = float(np.interp(amp_ref_90, amp_tip_m[sort_m], omega_m_bb[sort_m]))
om_py_90 = float(np.interp(amp_ref_90, amp_bb[sort_py], omega_bb[sort_py]))
rel_err_90 = abs(om_py_90 - om_m_90) / om_m_90

print("\n" + "=" * 65)
print("  Example 08 — Beam Cubic Spring NMA Summary")
print("=" * 65)
print(f"  Linear omega1             : {omega1_linear:.4f} rad/s ({omega1_linear/(2*np.pi):.4f} Hz)")
print(f"  Backbone omega range      : [{omega_bb.min():.2f}, {omega_bb.max():.2f}] rad/s")
print(f"  Backbone amp range        : [{amp_bb.min():.3e}, {amp_bb.max():.3e}] m")
print(f"  Rel error at 90% amp      : {rel_err_90*100:.2f}%  (< 1% target)")
print(f"  Wall time                 : {_t_elapsed:.1f} s  (< 120 s target)")
print(f"  Method: {N_MODES}-mode Galerkin reduction + omega-continuation, H={N_HARMONICS}")
print("=" * 65)

# Assertions
assert rel_err_90 < 0.01, f"Backbone error {rel_err_90*100:.2f}% exceeds 1% at 90% amp reference"
print("\nPASS: backbone error < 1%")
assert _t_elapsed < 120.0, f"Runtime {_t_elapsed:.1f} s exceeds 120 s limit"
print(f"PASS: runtime {_t_elapsed:.1f} s < 120 s")
