"""Generate MATLAB reference fixtures for validation.

Runs the original NLvib MATLAB examples via subprocess (MATLAB or Octave)
and saves outputs as .npz files in tests/fixtures/.

Prerequisites:
  - MATLAB or Octave must be on PATH, OR
  - Run tools/fetch_matlab_source.sh first to download MATLAB source

Usage:
    python tools/generate_fixtures.py [--engine matlab|octave] [--example 01_Duffing]

The generated fixtures are the ground truth for all Python validation tests.
Each fixture stores: omega, amplitude, phase, stability arrays + metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MATLAB_SRC = REPO_ROOT / "tools" / "NLvib_matlab"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# Maps example folder name → expected output variables exported by the wrapper script
EXAMPLES: dict[str, dict] = {
    "01_Duffing": {
        "script": "Duffing.m",
        "exports": ["Om_HB", "Q_HB", "Om_shoot", "Q_shoot"],
    },
    "02_twoDOFoscillator_cubicSpring": {
        "script": "twoDOFoscillator_cubicSpring.m",
        "exports": ["Om", "Q"],
    },
    "03_twoDOFoscillator_unilateralSpring": {
        "script": "twoDOFoscillator_unilateralSpring.m",
        "exports": ["Om", "Q"],
    },
    "04_twoDOFoscillator_cubicSpring_NM": {
        "script": "twoDOFoscillator_cubicSpring_NM.m",
        "exports": ["Om", "Q"],
    },
    "05_twoDOFoscillator_tanhDryFriction_NM": {
        "script": "twoDOFoscillator_tanhDryFriction_NM.m",
        "exports": ["Om", "Q"],
    },
}


def find_engine() -> str:
    """Auto-detect MATLAB or Octave on PATH."""
    for engine in ("matlab", "octave", "octave-cli"):
        if shutil.which(engine):
            return engine
    raise RuntimeError(
        "Neither MATLAB nor Octave found on PATH.\n"
        "Install one of them or provide --engine with the full path."
    )


def run_example(example: str, engine: str) -> Path:
    """Run a single MATLAB example and return path to generated fixture."""
    src_dir = MATLAB_SRC / "EXAMPLES" / example
    if not src_dir.exists():
        raise FileNotFoundError(
            f"MATLAB source not found at {src_dir}\n"
            "Run:  bash tools/fetch_matlab_source.sh"
        )

    info = EXAMPLES[example]
    # Wrapper: run script then save variables to a .mat file
    mat_out = FIXTURES_DIR / f"{example}.mat"
    export_vars = ", ".join(f"'{v}'" for v in info["exports"])

    if "matlab" in engine:
        cmd = [
            engine,
            "-batch",
            f"cd('{src_dir}'); run('{info['script']}'); "
            f"save('{mat_out}', {export_vars});",
        ]
    else:  # octave
        cmd = [
            engine,
            "--no-gui",
            "--eval",
            f"cd('{src_dir}'); run('{info['script']}'); "
            f"save('{mat_out}', {export_vars});",
        ]

    print(f"Running {example} via {engine} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"MATLAB/Octave run failed for {example}")

    # Convert .mat → .npz using scipy
    try:
        import numpy as np
        import scipy.io as sio

        mat = sio.loadmat(str(mat_out))
        arrays = {k: mat[k] for k in info["exports"] if k in mat}
        npz_out = FIXTURES_DIR / f"{example}.npz"
        np.savez(str(npz_out), **arrays)
        mat_out.unlink()  # remove intermediate .mat
        print(f"  Saved fixture: {npz_out}")
        return npz_out
    except ImportError:
        print("  scipy/numpy not available — .mat file left in place")
        return mat_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=None, help="matlab|octave|path-to-binary")
    parser.add_argument("--example", default=None, help="Run only this example")
    args = parser.parse_args()

    if not MATLAB_SRC.exists():
        sys.exit(
            "MATLAB source not found.\nRun:  bash tools/fetch_matlab_source.sh"
        )

    engine = args.engine or find_engine()
    targets = [args.example] if args.example else list(EXAMPLES.keys())

    failed = []
    for ex in targets:
        try:
            run_example(ex, engine)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(ex)

    print()
    print(f"Done. {len(targets) - len(failed)}/{len(targets)} fixtures generated.")
    if failed:
        print("Failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
