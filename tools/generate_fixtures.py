"""Fixture loader and inspector for NLvib validation fixtures.

This script serves two purposes:

1. **Library** — provides :func:`load_fixture` for use in tests and notebooks.
2. **CLI tool** — ``python tools/generate_fixtures.py --list`` prints a summary
   table of all fixtures found under ``tests/fixtures/``.

The companion MATLAB script ``tools/generate_fixtures.m`` generates the ``.npz``
files from the original NLvib MATLAB source.  This Python script does not invoke
MATLAB; it only loads and inspects the already-generated fixture files.

Usage::

    # List all fixtures
    python tools/generate_fixtures.py --list

    # Inspect a specific fixture
    python tools/generate_fixtures.py --inspect 01_Duffing

    # Load in Python code
    from tools.generate_fixtures import load_fixture
    fx = load_fixture("01_Duffing")
    print(fx["omega"].shape)

Schema
------
Every fixture is a dict with at minimum:

    omega      : ndarray, shape (n_points,)  — frequency (rad/s)
    amplitude  : ndarray, shape (n_points,)  — response amplitude at DOF 0
    phase      : ndarray, shape (n_points,)  — phase of fundamental harmonic (rad)
    stability  : ndarray, shape (n_points,), dtype bool
    tolerance  : float  — validation tolerance (1e-6 default, 1e-4 for non-smooth)

Additional arrays (omega_shoot, amplitude_dof2, Q_harmonics, …) may be present
depending on the example; see ``tests/fixtures/README.md`` for the full catalogue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"

# Canonical list of all expected MATLAB-generated fixtures (name → description).
_EXPECTED_FIXTURES: dict[str, str] = {
    "01_Duffing": "1-DOF Duffing oscillator — HB + Shooting overlay",
    "02_twoDOFoscillator_cubicSpring": "2-DOF chain with cubic spring — HB",
    "03_twoDOFoscillator_unilateralSpring": "2-DOF chain with unilateral spring — HB",
    "04_twoDOFoscillator_cubicSpring_NM": "2-DOF cubic spring — NMA backbone",
    "05_twoDOFoscillator_tanhDryFriction_NM": "2-DOF tanh friction — NMA backbone",
    "06_twoDOFoscillator_tanhDryFriction_FRF": "2-DOF tanh friction — HB FRF",
    "07_geometricNonlinearity": "Multi-DOF geometric nonlinearity — HB",
    "08_multiDOF_multiNL": "Multi-DOF with multiple NL elements — HB",
    "09_beam_tanhFriction": "FE beam + tanh friction — HB",
    "10_beam_cubicSpring_NM": "FE beam + cubic spring — NMA backbone",
    "analytical/duffing_backbone": "Duffing backbone (analytical, no MATLAB)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_fixture(name: str, fixtures_dir: Path | None = None) -> dict[str, Any]:
    """Load a fixture ``.npz`` file and return its contents as a plain dict.

    Parameters
    ----------
    name:
        Fixture name without the ``.npz`` extension, e.g. ``"01_Duffing"`` or
        ``"analytical/duffing_backbone"``.
    fixtures_dir:
        Override the default ``tests/fixtures/`` directory.  Useful in tests
        that need to point at a temporary directory.

    Returns
    -------
    dict[str, Any]
        Keys are the array names stored in the ``.npz`` file.  Every fixture
        guarantees ``omega``, ``amplitude``, ``phase``, ``stability``, and
        ``tolerance``.  Additional arrays may be present (see module docstring).

    Raises
    ------
    FileNotFoundError
        If the requested fixture file does not exist.
    """
    base_dir = fixtures_dir if fixtures_dir is not None else _FIXTURES_DIR
    path = base_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture not found: {path}\n"
            "Generate it with:  matlab -batch \"run('tools/generate_fixtures.m')\"\n"
            "or for the analytical fixture:  python tools/generate_fixtures.py --gen-analytical"
        )
    with np.load(str(path), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def list_fixtures(fixtures_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return metadata for every ``.npz`` file found in *fixtures_dir*.

    Each entry is a dict with keys:

    - ``name``        — fixture name relative to fixtures_dir (no extension)
    - ``path``        — absolute Path to the file
    - ``present``     — bool
    - ``n_points``    — int or None  (length of ``omega`` array if present)
    - ``arrays``      — list[str]  (array names in the file)
    - ``description`` — str  (from ``_EXPECTED_FIXTURES`` catalogue, or "unknown")
    """
    base_dir = fixtures_dir if fixtures_dir is not None else _FIXTURES_DIR
    results: list[dict[str, Any]] = []

    # Walk expected list first so missing files are included in the output.
    seen: set[str] = set()
    for name, description in _EXPECTED_FIXTURES.items():
        path = base_dir / f"{name}.npz"
        present = path.exists()
        entry: dict[str, Any] = {
            "name": name,
            "path": path,
            "present": present,
            "n_points": None,
            "arrays": [],
            "description": description,
        }
        if present:
            with np.load(str(path), allow_pickle=False) as npz:
                entry["arrays"] = list(npz.files)
                if "omega" in npz.files:
                    entry["n_points"] = int(npz["omega"].shape[0])
        results.append(entry)
        seen.add(name)

    # Also surface any unexpected .npz files already in the directory.
    for npz_path in sorted(base_dir.rglob("*.npz")):
        rel = npz_path.relative_to(base_dir)
        name = str(rel.with_suffix(""))
        if name in seen:
            continue
        with np.load(str(npz_path), allow_pickle=False) as npz:
            arrays = list(npz.files)
            n_points = int(npz["omega"].shape[0]) if "omega" in npz.files else None
        results.append(
            {
                "name": name,
                "path": npz_path,
                "present": True,
                "n_points": n_points,
                "arrays": arrays,
                "description": "unknown",
            }
        )

    return results


def print_summary_table(fixtures_dir: Path | None = None) -> None:
    """Print a human-readable summary table of all fixtures to stdout."""
    rows = list_fixtures(fixtures_dir)
    col_name = max(len(r["name"]) for r in rows) + 2
    col_desc = 50
    header = (
        f"{'Fixture':<{col_name}} {'Present':<8} {'n_points':<10} "
        f"{'Arrays':<8} {'Description':<{col_desc}}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for r in rows:
        status = "YES" if r["present"] else "---"
        n_pts = str(r["n_points"]) if r["n_points"] is not None else "—"
        n_arr = str(len(r["arrays"])) if r["arrays"] else "—"
        desc = r["description"][:col_desc]
        print(
            f"{r['name']:<{col_name}} {status:<8} {n_pts:<10} {n_arr:<8} {desc:<{col_desc}}"
        )
    print(separator)
    present = sum(1 for r in rows if r["present"])
    print(f"\n{present}/{len(rows)} fixtures present in {_FIXTURES_DIR}\n")


def inspect_fixture(name: str, fixtures_dir: Path | None = None) -> None:
    """Print detailed information about a single fixture to stdout."""
    try:
        data = load_fixture(name, fixtures_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFixture: {name}")
    print(f"File:    {(_FIXTURES_DIR if fixtures_dir is None else fixtures_dir) / (name + '.npz')}")
    print(f"\n{'Array':<25} {'dtype':<10} {'shape':<20} {'min':>12} {'max':>12}")
    print("-" * 82)
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.ndim == 0:
            # Scalar
            print(f"  {key:<23} {str(arr.dtype):<10} {'scalar':<20} {float(arr):>12.4g}")
        else:
            vmin = float(np.min(arr)) if np.issubdtype(arr.dtype, np.number) else "—"
            vmax = float(np.max(arr)) if np.issubdtype(arr.dtype, np.number) else "—"
            min_str = f"{vmin:>12.4g}" if isinstance(vmin, float) else f"{'—':>12}"
            max_str = f"{vmax:>12.4g}" if isinstance(vmax, float) else f"{'—':>12}"
            print(f"  {key:<23} {str(arr.dtype):<10} {str(arr.shape):<20} {min_str} {max_str}")
    print()


def generate_analytical_duffing_backbone(
    fixtures_dir: Path | None = None,
    n_points: int = 200,
    omega0: float = 1.0,
    k3: float = 1.0,
    a_min: float = 0.01,
    a_max: float = 2.0,
) -> Path:
    """Generate the analytical Duffing backbone curve and save as ``.npz``.

    The Duffing backbone relates modal amplitude *A* to natural frequency via:

        ω²(A) = ω₀² + (3/4) k₃ A²

    Reference: Krack & Gross (2019), §2.2, eq. (2.33) (single-harmonic
    approximation for conservative Duffing oscillator).

    Parameters
    ----------
    fixtures_dir:
        Directory under which ``analytical/`` will be created.  Defaults to
        ``tests/fixtures/``.
    n_points:
        Number of amplitude samples.  Defaults to 200.
    omega0:
        Linear natural frequency (rad/s).  Default 1.0.
    k3:
        Cubic stiffness coefficient.  Default 1.0.
    a_min:
        Minimum amplitude.  Default 0.01.
    a_max:
        Maximum amplitude.  Default 2.0.

    Returns
    -------
    Path
        Absolute path to the written ``.npz`` file.
    """
    base_dir = fixtures_dir if fixtures_dir is not None else _FIXTURES_DIR
    out_dir = base_dir / "analytical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "duffing_backbone.npz"

    amplitude = np.linspace(a_min, a_max, n_points)
    omega = np.sqrt(omega0**2 + 0.75 * k3 * amplitude**2)

    np.savez(str(out_path), amplitude=amplitude, omega=omega)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="Print a summary table of all fixtures.",
    )
    group.add_argument(
        "--inspect",
        metavar="NAME",
        help="Inspect a single fixture (e.g. '01_Duffing').",
    )
    group.add_argument(
        "--gen-analytical",
        action="store_true",
        help="Generate (or regenerate) analytical/duffing_backbone.npz.",
    )
    parser.add_argument(
        "--fixtures-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help="Override default tests/fixtures/ directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    fdir: Path | None = args.fixtures_dir

    if args.list:
        print_summary_table(fdir)
    elif args.inspect:
        inspect_fixture(args.inspect, fdir)
    elif args.gen_analytical:
        path = generate_analytical_duffing_backbone(fdir)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
