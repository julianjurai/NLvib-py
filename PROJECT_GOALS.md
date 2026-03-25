# NLvib Python Port — Project Goals

## Attribution

This project is a Python port of **NLvib**, a MATLAB toolbox for nonlinear vibration analysis.

- **Original authors**: Prof. Dr.-Ing. Malte Krack & Dr.-Ing. Johann Groß
- **Institution**: Institute for Aircraft Propulsion (ILA), University of Stuttgart
- **Original source**: https://www.ila.uni-stuttgart.de/nlvib/ | https://github.com/maltekrack/NLvib
- **Reference textbook**: Krack M. & Gross J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*. Springer. ISBN 978-3-030-14022-9.
- **License**: GPL-3.0 (inherited from original)

---

## Goals

### G1 — Numerical Correctness
- All results must match MATLAB reference outputs within a defined tolerance (default ≤ 1e-6 relative error on frequency response amplitudes and eigenfrequencies).
- MATLAB reference outputs for all 12 canonical examples are committed as test fixtures.
- Floquet stability analysis is included (not just frequency response).

### G2 — Full Agentic Workflow
- A persistent PM agent owns `TASKS.md` and controls what is in-progress at any time.
- Dev agents implement modules; QA agents validate; Review agents enforce quality.
- Work can be stopped and resumed at any point via the PM state file.
- Sub-agents are spun up on demand to test numerical assumptions before committing to an approach.

### G3 — Pythonic API
- Not a literal MATLAB-to-Python translation. The API is designed for Python idioms: dataclasses, type hints, composable builders, `__repr__`.
- Users do not need to read MATLAB source to use this library.
- Full `mypy --strict` coverage on all public interfaces.
- Each public function/class docstring cross-references the equation number in Krack & Gross (2019).

### G4 — Comprehensive Test Suite
- Three tiers: unit → integration → validation.
- Unit: individual force functions, matrix builders, FFT transforms.
- Integration: solver + continuation end-to-end on simple systems.
- Validation: full 12-example suite compared against MATLAB fixtures.
- CI runs all tests on every commit (GitHub Actions).

### G5 — Demo Notebooks
- Every canonical example has a Jupyter notebook that runs top-to-bottom without errors.
- Notebooks are also available as plain Python scripts (converted via jupytext or equivalent).
- Notebooks include inline theory explanations cross-referencing Krack & Gross (2019).
- Binder/Colab launch badges in README.

### G6 — Documentation
- Auto-generated API reference from docstrings (mkdocs + mkdocstrings).
- A "Differences from MATLAB" page documenting intentional API changes.
- Theory ↔ code cross-reference index.

### G7 — Performance
- NumPy vectorisation throughout; no Python loops in hot paths.
- JAX backend is an optional install for autodiff Jacobians.
- scipy.sparse used from day one for all system matrices.
- Profiling benchmarks committed alongside each major module.

### G8 — Incremental Delivery
- Modules are delivered and validated independently, in dependency order.
- Nothing is merged unless it has passing tests, type annotations, and at least one MATLAB-validated fixture.
- See `TASKS.md` for the dependency graph and current state.

### G9 — Packaging
- `pip install nlvib` as stretch goal (PyPI publish).
- `pyproject.toml`-based build (PEP 517/518).
- semver versioning from v0.1.0.

---

## Success Criteria (Definition of Done for v1.0)

- [ ] All 12 MATLAB examples reproduced in Python with ≤ 1e-6 relative error
- [ ] Full test suite passing in CI (unit + integration + validation)
- [ ] All 12 Jupyter notebooks run clean
- [ ] API reference site deployed
- [ ] `mypy --strict` passes on public API
- [ ] README, attribution, and license in place
- [ ] CHANGELOG maintained
