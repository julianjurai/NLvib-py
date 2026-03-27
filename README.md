# NLvib — Python

A Python port of **NLvib**, the MATLAB toolbox for nonlinear vibration analysis developed at the University of Stuttgart.

Implements harmonic balance, shooting method, and arc-length continuation for analyzing systems with nonlinear vibrations.

## Attribution

Original MATLAB toolbox by **Prof. Dr.-Ing. Malte Krack** and **Dr.-Ing. Johann Groß**,
Institute for Aircraft Propulsion (ILA), University of Stuttgart.

- Website: https://www.ila.uni-stuttgart.de/nlvib/
- Source: https://github.com/maltekrack/NLvib
- Reference: Krack M. & Gross J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*. Springer.

License: MIT. See [LICENSE](LICENSE) for details.

---

## What this port includes

| Module | Status |
|--------|--------|
| Nonlinear elements (cubic spring, friction, unilateral, polynomial) | In progress |
| Mechanical system classes (1-DOF, chain, FE beam, FE rod) | Planned |
| Harmonic balance residual + AFT | Planned |
| Shooting residual + Newmark integrator | Planned |
| Arc-length continuation solver | Planned |
| CMS model reduction | Planned |
| All 12 canonical examples as Python scripts | Planned |
| Jupyter notebooks for each example | Planned |

See `TASKS.md` for the full dependency graph and current state.

---

## Setup

### Requirements

- Python ≥ 3.11
- numpy ≥ 1.26
- scipy ≥ 1.12
- matplotlib ≥ 3.8

### Install (development)

```bash
# Clone this repo
git clone <this-repo>
cd nonlinear_vibration_analysis_toolbox

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest
```

### Run a specific example

```bash
python examples/01_duffing/run.py
```

---

## Generating MATLAB reference outputs (for validation)

This toolbox validates its results against the original MATLAB implementation.
To generate the reference fixtures locally:

### Step 1 — Download the MATLAB source

```bash
bash tools/fetch_matlab_source.sh
```

This clones the original NLvib MATLAB repo (NLvib-Basic branch) into `tools/NLvib_matlab/`.
The MATLAB source is not bundled here — it is fetched on demand.

For the NLvib-PEACE extended branch:

```bash
bash tools/fetch_matlab_source.sh peace
```

### Step 2 — Run the MATLAB examples and save fixtures

Requires **MATLAB** or **GNU Octave** on your PATH.

```bash
# Auto-detects matlab or octave
python tools/generate_fixtures.py

# Specify engine explicitly
python tools/generate_fixtures.py --engine octave

# Generate one example only
python tools/generate_fixtures.py --example 01_Duffing
```

Fixtures are saved to `tests/fixtures/*.npz` and used by the validation test suite.

### Step 3 — Run validation tests

```bash
pytest tests/validation/
```

See `tests/fixtures/README.md` for fixture format and tolerance documentation.

---

## Notebooks

Interactive Jupyter notebooks are in `notebooks/`. Each notebook corresponds to one canonical example.

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## Comparison Notebooks

`notebooks/comparison/` contains eight notebooks that run the Python HB continuation and the
MATLAB/Octave reference side-by-side, then overlay both curves on one figure to quantify
numerical agreement.

| Example | MATLAB source | Peak error |
|---------|--------------|------------|
| 01 Duffing | `EXAMPLES/01_Duffing` | 0.0007% |
| 02 Two-DOF cubic | `EXAMPLES/02_twoDOFoscillator_cubicSpring` | 0.01% |
| 03 Two-DOF unilateral | `EXAMPLES/03_twoDOFoscillator_unilateralSpring` | 0.08% |
| 04 Two-DOF tanh NMA | `EXAMPLES/05_twoDOFoscillator_tanhDryFriction_NM` | 0.09% |
| 05 Geometric nonlinearity | `EXAMPLES/06_twoSprings_geometricNonlinearity` | <1% |
| 06 Multi-DOF multi-NL | `EXAMPLES/07_multiDOFoscillator_multipleNonlinearities` | <5% (Jenkins) |
| 07 Beam tanh friction | `EXAMPLES/08_beam_tanhDryFriction` | 0.29% |
| 08 Beam cubic NMA | `EXAMPLES/09_beam_cubicSpring_NM` | <5% (3-mode) |

See [`notebooks/comparison/CONTEXT.md`](notebooks/comparison/CONTEXT.md) for the full technical
context, notebook template, and per-example notes.

---

## Citing this work

If you use this toolbox in published work, please cite both the original Krack & Gross (2019)
textbook and this Python port.

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff). GitHub will
surface a "Cite this repository" button in the sidebar automatically.

Manual BibTeX entry for the Python port:

```bibtex
@software{nlvib_python,
  title  = {{NLvib Python}: A Python port of the {NLvib} {MATLAB} toolbox},
  author = {NLvib Python Contributors},
  year   = {2026},
  url    = {https://github.com/julianjurai/nonlinear_vibration_analysis_toolbox},
}
```

For the underlying algorithms, cite the textbook:

```bibtex
@book{krack2019harmonic,
  author    = {Krack, M. and Gro{\ss}, J.},
  title     = {Harmonic Balance for Nonlinear Vibration Problems},
  year      = {2019},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-14023-6},
}
```

---

## Project structure

```
src/nlvib/
├── nonlinearities/    # Force element definitions
├── systems/           # Mechanical system classes
├── solvers/           # HB residual, shooting residual
├── continuation/      # Arc-length continuation
├── io/                # CalculiX mesh/matrix IO
└── utils/             # FFT transforms, scaling, linear algebra

tests/
├── unit/              # Per-function tests
├── integration/       # End-to-end solver tests
└── validation/        # Comparison against MATLAB fixtures

examples/              # Runnable Python scripts
notebooks/             # Jupyter notebooks
tools/                 # MATLAB source fetcher, fixture generator
docs/                  # API reference (auto-generated)
```

---

---

## Starting a Development Session (Claude Code)

This project uses a structured multi-agent workflow. Use these prompts at the start of each session.

### 1. Orient the PM agent

Paste this at the start of every session:

```
Read TASKS.md and AGENTS.md. You are the PM agent. Report:
1. Current task status table (done / in_progress / ready / todo / blocked)
2. What was in progress last session and its state
3. What is unblocked and ready to assign this session
Do not start any work yet — wait for confirmation.
```

### 2. Assign work

After reviewing the status, tell Claude what to focus on:

```
PM: assign T-01 and T-02 in parallel. Start with T-01.
```

Or let the PM decide:

```
PM: assign the highest-priority unblocked tasks for this session.
```

### 3. Run an assumption sub-agent before complex implementation

When a Dev agent hits an uncertain algorithmic choice:

```
Before implementing T-12, run an assumption check:
python tools/openai_validator.py assume \
  "In NLvib's HB method, the AFT transform uses H harmonics.
   What numpy.fft convention matches the MATLAB fft/ifft convention used in HB_residual.m?"
```

### 4. Run the QA checklist on a completed task

```
QA: run checklist on T-01.
Check: pytest tests/unit/test_elements.py, mypy, ruff, docstring completeness,
and run openai_validator jacobian check on all new nonlinear elements.
```

### 5. Stop cleanly

```
PM: I am ending this session. Write the current state of all in-progress tasks
to TASKS.md session log and confirm what is safe to resume next time.
```

### Key files to read at session start

| File | Purpose |
|------|---------|
| `TASKS.md` | Current task state, dependency graph, session log |
| `AGENTS.md` | Agent roles, protocols, OpenAI integration, file ownership |
| `PROJECT_GOALS.md` | Locked goals and definition of done |

---

## Development

See `AGENTS.md` for the multi-agent development framework and `TASKS.md` for current task state and dependency graph.

```bash
# Lint
ruff check src/

# Type check
mypy src/nlvib/

# Format
ruff format src/
```
