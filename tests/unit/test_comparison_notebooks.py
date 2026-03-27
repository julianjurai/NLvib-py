"""
Structural smoke tests for comparison notebooks 03–08.

These tests check that each notebook file exists and contains the expected
``## MATLAB vs Python`` section header (as a markdown cell source string).
No notebook execution is performed; the tests are lightweight JSON-parse checks.

Reference: notebooks/comparison/0{3-8}_*.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPARISON_DIR = (
    Path(__file__).parents[2] / "notebooks" / "comparison"
)

_EXPECTED_SECTION = "## MATLAB vs Python"


def _notebook_has_section(nb_path: Path, section: str) -> bool:
    """Return True if any cell in *nb_path* contains *section* in its source.

    Parameters
    ----------
    nb_path:
        Absolute path to a Jupyter notebook (.ipynb) file.
    section:
        Markdown heading string to search for (exact substring match).

    Returns
    -------
    bool
    """
    with nb_path.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return any(section in "".join(cell["source"]) for cell in nb.get("cells", []))


def _find_notebook(prefix: str) -> Path:
    """Return the path to the first notebook matching *prefix* (e.g. ``'03_'``).

    Parameters
    ----------
    prefix:
        Filename prefix, e.g. ``'03_'`` or ``'07_'``.

    Raises
    ------
    FileNotFoundError
        If no matching notebook is found in ``_COMPARISON_DIR``.
    """
    matches = sorted(_COMPARISON_DIR.glob(f"{prefix}*.ipynb"))
    if not matches:
        raise FileNotFoundError(
            f"No comparison notebook matching '{prefix}*.ipynb' found in {_COMPARISON_DIR}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Tests: notebooks 03–08
# ---------------------------------------------------------------------------


class TestComparisonNotebooks:
    """Lightweight structural checks for comparison notebooks 03–08.

    Each test:
    1. Asserts the notebook file exists on disk.
    2. Parses the notebook JSON (no kernel execution).
    3. Asserts that at least one cell contains the heading
       ``## MATLAB vs Python``, which demarcates the side-by-side comparison
       section added in the T-29–T-36 comparison-notebook build pass.
    """

    def test_notebook_03_has_matlab_vs_python_section(self) -> None:
        """Notebook 03 (two-DOF unilateral spring) must have MATLAB vs Python section."""
        nb_path = _find_notebook("03_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )

    def test_notebook_04_has_matlab_vs_python_section(self) -> None:
        """Notebook 04 (two-DOF tanh friction) must have MATLAB vs Python section."""
        nb_path = _find_notebook("04_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )

    def test_notebook_05_has_matlab_vs_python_section(self) -> None:
        """Notebook 05 (geometric nonlinearity) must have MATLAB vs Python section."""
        nb_path = _find_notebook("05_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )

    def test_notebook_06_has_matlab_vs_python_section(self) -> None:
        """Notebook 06 (multi-DOF, multi-NL) must have MATLAB vs Python section."""
        nb_path = _find_notebook("06_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )

    def test_notebook_07_has_matlab_vs_python_section(self) -> None:
        """Notebook 07 (FE beam + tanh friction) must have MATLAB vs Python section."""
        nb_path = _find_notebook("07_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )

    def test_notebook_08_has_matlab_vs_python_section(self) -> None:
        """Notebook 08 (FE beam + cubic spring NMA) must have MATLAB vs Python section."""
        nb_path = _find_notebook("08_")
        assert nb_path.exists(), f"Notebook not found: {nb_path}"
        assert _notebook_has_section(nb_path, _EXPECTED_SECTION), (
            f"'{_EXPECTED_SECTION}' section missing in {nb_path.name}"
        )
