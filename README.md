# NLvib-py

A Python port of **NLvib**, the MATLAB toolbox for nonlinear vibration analysis developed at the University of Stuttgart (Krack & Groß, 2019).

Implements harmonic balance (HB), shooting method, and arc-length continuation for analyzing systems with nonlinear steady-state vibrations.

---

## Features

| Module | Description |
|--------|-------------|
| `nlvib.nonlinearities` | Cubic spring, unilateral spring, tanh/elastic dry friction, polynomial stiffness |
| `nlvib.systems` | 1-DOF, chain of oscillators, FE beam, FE rod |
| `nlvib.solvers` | Harmonic balance residual + AFT, shooting residual + Newmark integrator |
| `nlvib.continuation` | Pseudo-arc-length continuation with fold detection and adaptive step control |
| `nlvib.io` | CalculiX mesh/matrix IO |
| Examples | 8 canonical examples as runnable Python scripts |
| Notebooks | Jupyter demo notebooks + MATLAB/Python comparison notebooks |

---

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/julianjurai/NLvib-py.git
cd NLvib-py
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest
```

### Run an example

```bash
python examples/01_Duffing/run.py
```

---

## Examples

| # | System | Nonlinearity | Method |
|---|--------|-------------|--------|
| 01 | Duffing (SDOF) | Cubic spring | HB + Shooting |
| 02 | 2-DOF chain | Cubic spring | HB |
| 03 | 2-DOF chain | Unilateral spring | HB + Shooting |
| 04 | 2-DOF chain | Tanh dry friction | NMA backbone |
| 05 | Single DOF geometric | Polynomial stiffness | HB + NMA |
| 06 | 3-DOF chain | Elastic dry friction (Jenkins) | HB |
| 07 | FE beam | Tanh dry friction at tip | HB |
| 08 | FE beam | Cubic spring at tip | NMA backbone |

---

## Validation against MATLAB

All 8 examples are validated against Octave-executed MATLAB reference outputs.
Comparison notebooks in `examples/comparison/` run both solvers side-by-side and overlay results.

| Example | Peak amplitude error | Peak frequency error |
|---------|---------------------|---------------------|
| 01 Duffing | 0.0007% | 0.010% |
| 02 Two-DOF cubic | 0.01% | 0.14% |
| 03 Two-DOF unilateral | 0.027% | 0.047% |
| 04 Two-DOF tanh NMA | <0.001% | <0.37% |
| 05 Geometric nonlinearity | <2% (3 of 4 levels) | <1% |
| 06 Multi-DOF Jenkins | <3.2% | <3.8% |
| 07 Beam tanh friction | 0.29% | 0.00% |
| 08 Beam cubic NMA | 0.14% (90th-pct) | 0.019% (ω₁) |

See [`examples/comparison/CONTEXT.md`](examples/comparison/CONTEXT.md) for methodology, known limitations, and per-example technical notes.

---

## Project structure

```
src/nlvib/              # Python package
├── nonlinearities/     # Force element definitions
├── systems/            # Mechanical system classes
├── solvers/            # HB residual, shooting residual
├── continuation/       # Arc-length continuation
├── io/                 # CalculiX mesh/matrix IO
└── utils/              # FFT transforms, scaling, linear algebra

examples/               # All Python examples and notebooks
├── demo/               # Tutorial Jupyter notebooks
├── comparison/         # MATLAB vs Python comparison notebooks
└── 01_Duffing/, ...    # Runnable Python scripts (8 examples)

matlab_src/             # Original MATLAB source (reference)
├── DOC/                # MATLAB documentation
├── EXAMPLES/           # Original MATLAB examples
└── SRC/                # Original MATLAB source code

tests/
├── unit/               # Per-function unit tests
├── integration/        # End-to-end solver tests
└── validation/         # Comparison against MATLAB fixtures

agents/                 # Agentic development framework
├── AGENTS.md           # Agent roles and protocols
├── PM.md               # PM agent guide
└── TASKS.md            # Task tracking

docs/                   # Documentation
├── user-guide/         # Conceptual guides
├── examples/           # Example documentation
└── api/                # API reference
```

---

## Attribution

Original MATLAB toolbox by **Prof. Dr.-Ing. Malte Krack** and **Dr.-Ing. Johann Groß**,
Institute for Aircraft Propulsion (ILA), University of Stuttgart.

- Website: https://www.ila.uni-stuttgart.de/nlvib/
- Source: https://github.com/maltekrack/NLvib
- Reference: Krack M. & Groß J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*. Springer. [doi:10.1007/978-3-030-14023-6](https://doi.org/10.1007/978-3-030-14023-6)

The original MATLAB commit history is preserved in the [`NLvib-Basic`](../../tree/NLvib-Basic) branch.

---

## Citing this work

A machine-readable citation is in [`CITATION.cff`](CITATION.cff). GitHub surfaces a "Cite this repository" button in the sidebar automatically.

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

For this Python port:

```bibtex
@software{nlvib_python,
  title  = {{NLvib-py}: A Python port of the {NLvib} {MATLAB} toolbox},
  year   = {2026},
  url    = {https://github.com/julianjurai/NLvib-py},
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
