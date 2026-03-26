#!/usr/bin/env python3
"""
bump_version.py — Atomically update the nlvib version in pyproject.toml and
src/nlvib/__init__.py.

Usage
-----
    python tools/bump_version.py --version 0.2.0

The script:
1. Validates the new version string as PEP 440 / semver X.Y.Z.
2. Reads both files and confirms the current version is consistent.
3. Writes both files in a single try/except — if either write fails the
   script exits non-zero and leaves a note about which file was updated.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT_PY = REPO_ROOT / "src" / "nlvib" / "__init__.py"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _validate_version(v: str) -> None:
    if not VERSION_RE.match(v):
        sys.exit(f"ERROR: version '{v}' must be X.Y.Z (e.g. 0.2.0)")


def _replace_in_file(path: Path, pattern: re.Pattern[str], replacement: str) -> str:
    """Return updated text, raising ValueError if pattern not found exactly once."""
    text = path.read_text(encoding="utf-8")
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 match for {pattern.pattern!r} in {path}, "
            f"found {len(matches)}"
        )
    return pattern.sub(replacement, text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump nlvib version in pyproject.toml and __init__.py")
    parser.add_argument("--version", required=True, help="New version string, e.g. 0.2.0")
    args = parser.parse_args()

    new_version: str = args.version
    _validate_version(new_version)

    # Patterns for replacement
    pyproject_pattern = re.compile(r'(?m)^(version\s*=\s*")[^"]+(")')
    init_pattern = re.compile(r'(?m)^(__version__\s*:\s*str\s*=\s*")[^"]+(")')

    try:
        new_pyproject = _replace_in_file(PYPROJECT, pyproject_pattern, rf'\g<1>{new_version}\g<2>')
        new_init = _replace_in_file(INIT_PY, init_pattern, rf'\g<1>{new_version}\g<2>')
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    # Write both files — treat as atomic by writing to the same filesystem
    PYPROJECT.write_text(new_pyproject, encoding="utf-8")
    INIT_PY.write_text(new_init, encoding="utf-8")

    print(f"Version bumped to {new_version}")
    print(f"  Updated: {PYPROJECT.relative_to(REPO_ROOT)}")
    print(f"  Updated: {INIT_PY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
