# NLvib Tasks — PM State File

> **PM Agent**: This is your source of truth. Read this at the start of every session.
> Update status fields in-place. Append to `## Session Log` — never delete log entries.

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `todo` | Not started, dependencies not yet met |
| `ready` | Dependencies met, can be assigned |
| `in_progress` | Actively being worked on |
| `review` | Code complete, awaiting QA + Review sign-off |
| `done` | QA passed, Review passed, merged |
| `blocked` | Waiting on external input (note reason) |

---

## Dependency Tiers

```
Tier 0 (independent)
  T-01  Nonlinear elements
  T-02  Utility functions (FFT transforms, norms, scaling)
  T-03  IO parsers (CalculiX FRD/mesh reader)
  T-04  MATLAB fixture generator spec + reference outputs
  T-27  Visualization module                 [no deps — pure matplotlib wrappers]

Tier 1 (needs Tier 0)
  T-05  Base MechanicalSystem class          [needs: T-01]
  T-06  SingleMassOscillator                 [needs: T-05]
  T-07  ChainOfOscillators                   [needs: T-05]
  T-08  FE EulerBernoulli beam               [needs: T-05, T-03]
  T-09  FE ElasticRod                        [needs: T-05, T-03]
  T-10  System with PolynomialStiffness      [needs: T-05, T-01]
  T-11  CMS model reduction (ROM)            [needs: T-05, T-08, T-09]

Tier 2 (needs Tier 1)
  T-12  Harmonic Balance residual + AFT      [needs: T-05, T-01, T-02]
  T-13  Shooting residual + Newmark          [needs: T-05, T-01]

Tier 3 (needs Tier 2)
  T-14  Arc-length continuation solver       [needs: T-12, T-13]

Tier 4 (needs Tier 3 + T-27)
  T-15  Example: 01 Duffing oscillator       [needs: T-14, T-06, T-27]
  T-16  Example: 02 Two-DOF cubic spring     [needs: T-14, T-07, T-27]
  T-17  Example: 03 Two-DOF unilateral spring[needs: T-14, T-07, T-27]
  T-18  Example: 04 Two-DOF tanh friction    [needs: T-14, T-07, T-27]
  T-19  Example: 05 Geometric nonlinearity   [needs: T-14, T-07, T-27]
  T-20  Example: 06 Multi-DOF multi-NL       [needs: T-14, T-07, T-27]
  T-21  Example: 07 Beam tanh friction       [needs: T-14, T-08, T-27]
  T-22  Example: 08 Beam cubic spring NM     [needs: T-14, T-08, T-27]

Tier 5 (needs Tier 4)
  T-23  Jupyter notebooks (one per example)  [needs: T-15 through T-22]
  T-24  API reference docs (mkdocs)          [needs: all Tier 1-3]
  T-25  CI/CD pipeline (GitHub Actions)      [needs: T-04, test suite]
  T-26  PyPI packaging                       [needs: T-24, T-25]
```

---

## Task Details

### T-01 — Nonlinear Elements
- **Status**: `ready`
- **Module**: `src/nlvib/nonlinearities/elements.py`
- **Deliver**:
  - `cubic_spring(k3, dof_index)` → NonlinearElement
  - `quadratic_damper(c2, dof_index)` → NonlinearElement
  - `tanh_dry_friction(f0, c, dof_index)` → NonlinearElement
  - `unilateral_spring(k, gap, dof_index)` → NonlinearElement
  - `polynomial_stiffness(exponents, coefficients, dof_indices)` → NonlinearElement
- **Equation refs**: Krack & Gross (2019) Appendix C, Table C.1
- **Tests**: `tests/unit/test_elements.py` — verify force values and Jacobians via finite difference
- **Acceptance**: Force values match MATLAB `HB_residual.m` nonlinearity evaluation; Jacobians match FD to 1e-8
- **Notes**: Each element returns (f, df_dq, df_ddq) — shape conventions must match HB and Shooting residual expectations

---

### T-02 — Utility Functions
- **Status**: `ready`
- **Module**: `src/nlvib/utils/transforms.py`, `src/nlvib/utils/linalg.py`
- **Deliver**:
  - `time_to_freq(q_time, n_harmonics)` — real-valued cosine-sine to complex Fourier
  - `freq_to_time(Q_freq, n_time_samples)` — inverse
  - `aft_transform(Q_freq, force_fn, n_time)` — full AFT pipeline
  - `dynamic_scaling(x, x_ref)` — NLvib's variable scaling for mixed-scale problems
  - `arc_length(x_prev, x_curr)` — arc length between two continuation points
- **Equation refs**: Krack & Gross (2019) §2.3 (AFT), §4.2 (scaling)
- **Tests**: `tests/unit/test_transforms.py` — round-trip FFT, AFT on known analytic function
- **Acceptance**: Round-trip error < 1e-12; AFT matches analytic Fourier coefficient for sin/cos inputs

---

### T-03 — IO Parsers
- **Status**: `ready`
- **Module**: `src/nlvib/io/calculix.py`
- **Deliver**:
  - `read_mesh(path)` → nodes, elements, element_type
  - `read_sparse_matrix(path)` → scipy.sparse.csr_matrix
  - `write_frd(path, nodes, time_series)` — animation export
- **Tests**: `tests/unit/test_io.py` with synthetic mesh files
- **Acceptance**: Round-trip write/read produces identical matrices

---

### T-04 — MATLAB Fixture Generator Spec
- **Status**: `ready`
- **Module**: `tests/fixtures/`
- **Deliver**:
  - `tests/fixtures/README.md` — documents what each fixture contains, how it was generated, MATLAB version used
  - `tests/fixtures/<example_name>.npz` — one file per example (12 total)
  - Each .npz contains: `omega`, `amplitude`, `phase`, `stability`, `tolerance`
  - Fixture generation script: `tools/generate_fixtures.m` (MATLAB script for reproducibility)
- **Notes**: Fixtures are the ground truth for all validation tests. Generated once from MATLAB, committed to repo. If MATLAB is unavailable, document analytical solutions for T-15 (Duffing) as backup.
- **Acceptance**: All fixture files present and documented

---

### T-27 — Visualization Module
- **Status**: `ready`
- **Module**: `src/nlvib/visualization/`
- **Deliver**:
  - `plot_frf(result, dof=0, harmonic=1)` → `Figure`
    Frequency response: amplitude vs Ω. Stable branches solid, unstable dashed. Matches MATLAB FRF plots.
  - `plot_backbone(result)` → `Figure`
    Backbone curve for NMA: frequency vs modal amplitude.
  - `plot_time_series(t, q, dof=0)` → `Figure`
    Steady-state time domain displacement and velocity.
  - `plot_phase_portrait(t, q, dq, dof=0)` → `Figure`
    Phase portrait q̇ vs q.
  - `plot_floquet(multipliers)` → `Figure`
    Floquet multipliers on the complex plane with unit circle. Stability indicator.
  - `plot_mode_shape(nodes, displacement, title="")` → `Figure`
    Spatial mode shape for FE beam/rod models.
  - `plot_harmonic_content(Q_harmonics, omega)` → `Figure`
    Bar chart of harmonic amplitudes Q_1, Q_3, Q_5 ...
  - `plot_convergence(residuals)` → `Figure`
    Residual norm vs continuation step or Newton iteration.
- **Design rules**:
  - All functions return `matplotlib.figure.Figure` — no `plt.show()`, no global state
  - Accept optional `ax=` parameter to plot into an existing axes
  - Accept optional `backend="matplotlib"|"plotly"` (plotly is optional dep)
  - Stable/unstable branch coloring driven by a `stability` boolean array on the result
- **Visual fixtures**: MATLAB-generated PNG snapshots stored in `tests/fixtures/plots/` for human comparison (not automated assertion)
- **Tests**: `tests/unit/test_visualization.py` — smoke tests that figures are created without error; check axis labels, legend entries
- **Acceptance**: All 8 plot functions implemented; smoke tests pass; visual match to MATLAB PNGs confirmed by human review

---

### T-05 — Base MechanicalSystem Class
- **Status**: `todo`
- **Deps**: T-01
- **Module**: `src/nlvib/systems/base.py`
- **Deliver**:
  - `NonlinearElement` dataclass
  - `MechanicalSystem` base class with M, D, K storage and `eval_nonlinear_forces()`
- **Tests**: `tests/unit/test_systems.py`
- **Acceptance**: mypy strict pass; eval_nonlinear_forces assembles correctly from multiple elements

---

### T-06 — SingleMassOscillator
- **Status**: `todo`
- **Deps**: T-05
- **Module**: `src/nlvib/systems/oscillators.py`
- **Deliver**: `SingleMassOscillator(m, d, k)` class
- **Tests**: Unit + verify M/D/K shapes and values
- **Acceptance**: Reproduces Duffing example system setup

---

### T-07 — ChainOfOscillators
- **Status**: `todo`
- **Deps**: T-05
- **Module**: `src/nlvib/systems/oscillators.py`
- **Deliver**: `ChainOfOscillators(masses, stiffnesses, dampings)` with tridiagonal matrix builder
- **Tests**: 2-DOF and 5-DOF cases, compare K matrix analytically
- **Acceptance**: Matches MATLAB ChainOfOscillators.m output matrices

---

### T-08 — FE EulerBernoulli Beam
- **Status**: `todo`
- **Deps**: T-05, T-03
- **Module**: `src/nlvib/systems/fe_beam.py`
- **Deliver**:
  - `FE_EulerBernoulliBeam(n_elements, L, E, I, rho, A, bc)`
  - `add_forcing(node, dof, amplitude)`
  - `add_nonlinear_attachment(node, element)`
  - `find_coordinate(node, dof)` → global DOF index
- **Equation refs**: Standard Euler-Bernoulli FEM; Krack & Gross (2019) §5.x
- **Tests**: Free vibration eigenfrequencies vs. analytical beam formula
- **Acceptance**: First 3 eigenfrequencies within 1% of analytical for n≥10 elements

---

### T-09 — FE ElasticRod
- **Status**: `todo`
- **Deps**: T-05, T-03
- **Module**: `src/nlvib/systems/fe_rod.py`
- **Deliver**: `FE_ElasticRod(n_elements, L, E, A, rho, bc)`
- **Tests**: Eigenfrequencies vs. analytical rod formula
- **Acceptance**: First 3 eigenfrequencies within 1% of analytical for n≥5 elements

---

### T-10 — System with Polynomial Stiffness
- **Status**: `todo`
- **Deps**: T-05, T-01
- **Module**: `src/nlvib/systems/polynomial.py`
- **Deliver**: `System_with_PolynomialStiffness(M, D, K, exponents, coefficients)`
- **Tests**: Force evaluation vs. analytic polynomial
- **Acceptance**: Matches MATLAB System_with_PolynomialStiffnessNonlinearity.m

---

### T-11 — CMS Model Reduction
- **Status**: `todo`
- **Deps**: T-05, T-08, T-09
- **Module**: `src/nlvib/systems/cms.py`
- **Deliver**: Craig-Bampton and Rubin reduction variants
- **Notes**: Most complex system module; may need Assume sub-agent for reduction basis selection
- **Acceptance**: Reduced model eigenfrequencies match full model within 1% for retained modes

---

### T-12 — Harmonic Balance Residual + AFT
- **Status**: `todo`
- **Deps**: T-05, T-01, T-02
- **Module**: `src/nlvib/solvers/harmonic_balance.py`
- **Deliver**:
  - `hb_residual(Q, omega, system, n_harmonics, excitation)` → (R, J)
  - `hb_residual_nma(Q, omega, system, n_harmonics)` → (R, J) for nonlinear modal analysis
  - AFT loop using `utils.transforms.aft_transform`
- **Equation refs**: Krack & Gross (2019) §2, §3; Appendix C
- **Tests**: Duffing oscillator residual at known solution point; Jacobian vs. FD
- **Acceptance**: Residual norm < 1e-10 at MATLAB solution points; Jacobian error < 1e-6 vs FD

---

### T-13 — Shooting Residual + Newmark
- **Status**: `todo`
- **Deps**: T-05, T-01
- **Module**: `src/nlvib/solvers/shooting.py`
- **Deliver**:
  - `newmark_step(y, f, M, D, K, dt, beta, gamma)` — single time step
  - `shooting_residual(y0, omega, system, n_periods, n_steps)` → (R, J)
- **Equation refs**: Krack & Gross (2019) §3.2; Newmark average constant acceleration (β=1/4, γ=1/2)
- **Tests**: Compare time integration against scipy.integrate.solve_ivp for linear system
- **Acceptance**: Period-averaged energy matches HB result within 1% on Duffing example

---

### T-14 — Arc-Length Continuation Solver
- **Status**: `todo`
- **Deps**: T-12, T-13
- **Module**: `src/nlvib/continuation/solver.py`
- **Deliver**:
  - `ContinuationSolver` class with `run(residual_fn, x0, lambda0, options)` method
  - Tangent predictor
  - Arc-length parametrisation
  - Adaptive step size (double if <5 iters, halve if >9 iters)
  - Termination criteria: parameter endpoint, user callback, reversal bounds
  - Returns: `ContinuationResult` dataclass with solution branch
- **Equation refs**: Krack & Gross (2019) §4
- **Notes**: Most algorithmically complex module. Dev agent must use Assume sub-agents to confirm: predictor scheme, step-size logic, parametrisation choice before implementation.
- **Tests**: Trace a circle in 2D (known analytic); trace Duffing FRF branch
- **Acceptance**: Duffing FRF matches MATLAB fixture within 1e-6

---

### T-15 through T-22 — Examples
- **Status**: `todo` (all)
- **Deps**: See dependency tier above
- **Module**: `examples/<name>/run.py`
- **Deliver**: Runnable script that:
  1. Runs the full continuation/analysis
  2. Produces all required plots via `nlvib.visualization`
  3. Saves plots to `examples/<name>/output/` as PNG
  4. Prints a summary table of key results (peak amplitude, resonance frequency)

**Required plots per example:**

| Example | Plots required |
|---------|---------------|
| 01 Duffing | FRF (HB + shooting overlay), harmonic content, time series at peak |
| 02 Two-DOF cubic | FRF (both DOFs), harmonic content |
| 03 Two-DOF unilateral | FRF, phase portrait (impact dynamics) |
| 04 Two-DOF tanh friction NM | Backbone curve, FRF |
| 05 Geometric nonlinearity | FRF (hardening/softening branch) |
| 06 Multi-DOF multi-NL | FRF all DOFs, convergence plot |
| 07 Beam tanh friction | FRF, mode shape at resonance |
| 08 Beam cubic spring NM | Backbone curve, mode shape |

- **Acceptance**: All plots saved without error; numerical results match MATLAB fixtures ≤ 1e-6; visual match to MATLAB PNGs confirmed by human review

---

### T-23 — Jupyter Notebooks
- **Status**: `todo`
- **Deps**: T-15 through T-22
- **Module**: `notebooks/`
- **Deliver**: One notebook per example, runs clean top-to-bottom
- **Acceptance**: `jupyter nbconvert --execute` exits 0

---

### T-24 — API Reference Docs
- **Status**: `todo`
- **Deps**: All Tier 1-3 done
- **Module**: `docs/`
- **Deliver**: mkdocs site with mkdocstrings, auto-built from docstrings
- **Acceptance**: Site builds without warnings

---

### T-25 — CI/CD Pipeline
- **Status**: `ready` (can be built anytime)
- **Module**: `.github/workflows/`
- **Deliver**:
  - `ci.yml` — runs pytest, mypy, ruff on push/PR
  - `validate.yml` — runs validation suite against fixtures (separate, slower)
- **Acceptance**: Both workflows pass on clean clone

---

### T-26 — PyPI Packaging
- **Status**: `todo`
- **Deps**: T-24, T-25
- **Deliver**: `pyproject.toml` complete, `twine upload` workflow, version bump script
- **Acceptance**: `pip install nlvib` installs and `import nlvib` works

---

## Current Sprint

> PM agent fills this in at session start.

- **Session date**: —
- **Focus**: —
- **Assigned tasks**: —
- **Blocked tasks**: —

---

## Session Log

> Append entries below. Never delete.

### Session 0 — Project initialization
- Repo initialized
- `PROJECT_GOALS.md`, `AGENTS.md`, `TASKS.md` written
- All T-00 through T-04 set to `ready`; remainder `todo`
- Skeleton `src/nlvib/` structure created (will be cleaned up before Tier 0 work begins)
- **Next**: PM to assign T-01, T-02, T-03, T-04, T-25, T-27 in parallel (all Tier 0 / independent)

### Session 1 — Spec updates
- Added G10 (Visualization) to PROJECT_GOALS.md — full inventory of all MATLAB plot types
- Added T-27 (Visualization module) to TASKS.md — `src/nlvib/visualization/`, 8 plot functions
- Added T-27 as dependency for all Tier 4 example tasks (T-15 through T-22)
- Expanded example task specs to list required plots per example
- Added OpenAI integration to AGENTS.md (o3 assumption testing, GPT-4o cross-validation)
- Added MATLAB source download tooling: `tools/fetch_matlab_source.sh`, `tools/generate_fixtures.py`
- Added `tools/openai_validator.py`
- Claude session startup instructions added to README.md
- **Next**: PM to assign T-01, T-02, T-03, T-04, T-25, T-27 in parallel (all Tier 0 / independent)
