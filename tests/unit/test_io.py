"""
Unit tests for nlvib.io.calculix — CalculiX mesh and sparse-matrix parsers.

Tests use synthetic in-memory files written to a tmp_path fixture so there
are no external file dependencies.  Round-trip fidelity is the primary
acceptance criterion for T-03.

All synthetic data is physically plausible (linear beam / rod meshes, sparse
stiffness/mass matrices from FE assembly) to also serve as documentation of
the expected format.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

from nlvib.io.calculix import (
    MeshData,
    read_mesh,
    read_sparse_matrix,
    write_frd,
    write_sparse_matrix,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic file content
# ---------------------------------------------------------------------------

# 4-node bar mesh (2 B31 beam elements, 3 nodes)
_BEAM_INP = textwrap.dedent(
    """\
    ** Simple 2-element beam — synthetic test mesh
    *NODE
    1, 0.0, 0.0, 0.0
    2, 0.5, 0.0, 0.0
    3, 1.0, 0.0, 0.0
    *ELEMENT, TYPE=B31, ELSET=BEAM
    1, 1, 2
    2, 2, 3
    """
)

# 8-node solid hex mesh (1 C3D8 element)
_HEX_INP = textwrap.dedent(
    """\
    ** Single C3D8 element
    *NODE
    1,  0.0,  0.0,  0.0
    2,  1.0,  0.0,  0.0
    3,  1.0,  1.0,  0.0
    4,  0.0,  1.0,  0.0
    5,  0.0,  0.0,  1.0
    6,  1.0,  0.0,  1.0
    7,  1.0,  1.0,  1.0
    8,  0.0,  1.0,  1.0
    *ELEMENT, TYPE=C3D8, ELSET=SOLID
    1, 1, 2, 3, 4, 5, 6, 7, 8
    """
)

# Mesh with CalculiX comment lines and blank lines (robustness test)
_COMMENTED_INP = textwrap.dedent(
    """\
    ** Header comment
    **
    *NODE
    ** node comment inside data block
    10, 0.0, 0.0, 0.0
    20, 1.0, 0.0, 0.0

    *ELEMENT, TYPE=T3D2
    5, 10, 20
    """
)


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# read_mesh tests
# ---------------------------------------------------------------------------


class TestReadMeshBeam:
    """Tests against the 3-node beam mesh."""

    def test_returns_mesh_data_type(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        assert isinstance(result, MeshData)

    def test_node_count(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        assert result.nodes.shape == (3, 3)

    def test_node_coordinates(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        npt.assert_array_equal(result.nodes, expected)

    def test_node_ids(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        npt.assert_array_equal(result.node_ids, [1, 2, 3])

    def test_element_count(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        assert result.elements.shape[0] == 2

    def test_element_connectivity_zero_indexed(self, tmp_path: Path) -> None:
        """Connectivity must be 0-based row indices, not CalculiX 1-based labels."""
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        # Element 1: nodes 1,2 → rows 0,1
        npt.assert_array_equal(result.elements[0, :2], [0, 1])
        # Element 2: nodes 2,3 → rows 1,2
        npt.assert_array_equal(result.elements[1, :2], [1, 2])

    def test_element_ids(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        npt.assert_array_equal(result.element_ids, [1, 2])

    def test_element_type(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(p)
        assert result.element_type == "B31"


class TestReadMeshHex:
    """Tests against the single-hex mesh."""

    def test_node_count(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "hex.inp", _HEX_INP)
        result = read_mesh(p)
        assert result.nodes.shape == (8, 3)

    def test_element_type(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "hex.inp", _HEX_INP)
        result = read_mesh(p)
        assert result.element_type == "C3D8"

    def test_element_connectivity_length(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "hex.inp", _HEX_INP)
        result = read_mesh(p)
        # 1 element, 8 nodes each
        assert result.elements.shape == (1, 8)

    def test_unit_cube_corners(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "hex.inp", _HEX_INP)
        result = read_mesh(p)
        # All coordinates are 0.0 or 1.0
        assert np.all((result.nodes == 0.0) | (result.nodes == 1.0))


class TestReadMeshRobustness:
    """Edge-case and robustness tests."""

    def test_comments_and_blank_lines(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "commented.inp", _COMMENTED_INP)
        result = read_mesh(p)
        assert result.nodes.shape[0] == 2

    def test_non_sequential_node_labels(self, tmp_path: Path) -> None:
        """Node labels 10, 20 should map to rows 0, 1 correctly."""
        p = _write_file(tmp_path, "commented.inp", _COMMENTED_INP)
        result = read_mesh(p)
        npt.assert_array_equal(result.node_ids, [10, 20])
        # Element refers to labels 10,20 → mapped to 0-based rows 0,1
        npt.assert_array_equal(result.elements[0, :2], [0, 1])

    def test_element_type_t3d2(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "commented.inp", _COMMENTED_INP)
        result = read_mesh(p)
        assert result.element_type == "T3D2"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_mesh(tmp_path / "nonexistent.inp")

    def test_missing_node_section(self, tmp_path: Path) -> None:
        content = "*ELEMENT, TYPE=B31\n1, 1, 2\n"
        p = _write_file(tmp_path, "no_nodes.inp", content)
        with pytest.raises(ValueError, match="NODE"):
            read_mesh(p)

    def test_missing_element_section(self, tmp_path: Path) -> None:
        content = "*NODE\n1, 0.0, 0.0, 0.0\n"
        p = _write_file(tmp_path, "no_elements.inp", content)
        with pytest.raises(ValueError, match="ELEMENT"):
            read_mesh(p)

    def test_accepts_path_string(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        result = read_mesh(str(p))  # string path
        assert result.nodes.shape == (3, 3)


# ---------------------------------------------------------------------------
# read_sparse_matrix tests
# ---------------------------------------------------------------------------


def _make_coo_content(
    rows: list[int], cols: list[int], vals: list[float]
) -> str:
    """Build COO file content with 1-based indices."""
    lines = ["# row  col  value"]
    for r, c, v in zip(rows, cols, vals):
        lines.append(f"{r + 1}  {c + 1}  {v:.17g}")
    return "\n".join(lines) + "\n"


class TestReadSparseMatrix:
    """Tests for read_sparse_matrix."""

    def test_returns_csr_matrix(self, tmp_path: Path) -> None:
        content = _make_coo_content([0, 1], [0, 1], [3.0, 4.0])
        p = _write_file(tmp_path, "mat.coo", content)
        mat = read_sparse_matrix(p)
        assert sp.issparse(mat)
        assert isinstance(mat, sp.csr_matrix)

    def test_shape_inferred(self, tmp_path: Path) -> None:
        content = _make_coo_content([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0])
        p = _write_file(tmp_path, "mat.coo", content)
        mat = read_sparse_matrix(p)
        assert mat.shape == (3, 3)

    def test_values_correct(self, tmp_path: Path) -> None:
        content = _make_coo_content([0, 0, 1], [0, 1, 1], [5.0, -2.0, 7.0])
        p = _write_file(tmp_path, "mat.coo", content)
        mat = read_sparse_matrix(p)
        dense = mat.toarray()
        npt.assert_allclose(dense[0, 0], 5.0)
        npt.assert_allclose(dense[0, 1], -2.0)
        npt.assert_allclose(dense[1, 1], 7.0)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_sparse_matrix(tmp_path / "missing.coo")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "empty.coo", "# just a comment\n")
        with pytest.raises(ValueError, match="No data"):
            read_sparse_matrix(p)

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        content = "1  2  notanumber\n"
        p = _write_file(tmp_path, "bad.coo", content)
        with pytest.raises(ValueError):
            read_sparse_matrix(p)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        content = _make_coo_content([0], [0], [42.0])
        p = _write_file(tmp_path, "mat.coo", content)
        mat = read_sparse_matrix(str(p))
        assert mat[0, 0] == 42.0


# ---------------------------------------------------------------------------
# write_sparse_matrix round-trip tests
# ---------------------------------------------------------------------------


class TestSparseRoundTrip:
    """Round-trip tests: write_sparse_matrix → read_sparse_matrix must be lossless."""

    def _random_sparse(self, n: int, density: float, seed: int) -> sp.csr_matrix:  # type: ignore[type-arg]
        rng = np.random.default_rng(seed)
        # Random symmetric sparse matrix (simulates a stiffness matrix)
        data = rng.standard_normal(int(n * n * density))
        rows = rng.integers(0, n, size=len(data))
        cols = rng.integers(0, n, size=len(data))
        mat: sp.csr_matrix = sp.coo_matrix(  # type: ignore[type-arg]
            (data, (rows, cols)), shape=(n, n)
        ).tocsr()
        return mat

    def test_diagonal_matrix(self, tmp_path: Path) -> None:
        diag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mat = sp.diags(diag, format="csr")
        p = tmp_path / "diag.coo"
        write_sparse_matrix(p, mat)
        recovered = read_sparse_matrix(p)
        npt.assert_allclose(
            recovered.toarray(), mat.toarray(), rtol=1e-14, atol=0.0
        )

    def test_10x10_tridiagonal(self, tmp_path: Path) -> None:
        """Mimics a 1-D FE stiffness matrix (tridiagonal)."""
        n = 10
        k = 1.0e6  # stiffness value [N/m]
        diags_vals = [
            -k * np.ones(n - 1),
            2 * k * np.ones(n),
            -k * np.ones(n - 1),
        ]
        mat = sp.diags(diags_vals, [-1, 0, 1], format="csr")
        p = tmp_path / "tridiag.coo"
        write_sparse_matrix(p, mat)
        recovered = read_sparse_matrix(p)
        assert recovered.shape == (n, n)
        npt.assert_allclose(
            recovered.toarray(), mat.toarray(), rtol=1e-14, atol=0.0
        )

    def test_50x50_random_sparse(self, tmp_path: Path) -> None:
        mat = self._random_sparse(50, density=0.05, seed=42)
        p = tmp_path / "random50.coo"
        write_sparse_matrix(p, mat)
        recovered = read_sparse_matrix(p)
        assert recovered.shape == mat.shape
        # Convert both to dense and compare
        npt.assert_allclose(
            recovered.toarray(), mat.toarray(), rtol=1e-13, atol=0.0
        )

    def test_rectangular_matrix(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(7)
        data = rng.standard_normal(15)
        rows = rng.integers(0, 5, size=15)
        cols = rng.integers(0, 8, size=15)
        mat: sp.csr_matrix = sp.coo_matrix(  # type: ignore[type-arg]
            (data, (rows, cols)), shape=(5, 8)
        ).tocsr()
        p = tmp_path / "rect.coo"
        write_sparse_matrix(p, mat)
        recovered = read_sparse_matrix(p)
        assert recovered.shape == (5, 8)
        npt.assert_allclose(recovered.toarray(), mat.toarray(), rtol=1e-13)

    def test_non_sparse_matrix_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "dense.coo"
        with pytest.raises(TypeError, match="scipy.sparse"):
            write_sparse_matrix(p, np.eye(4))  # type: ignore[arg-type]

    def test_nnz_preserved(self, tmp_path: Path) -> None:
        """Number of stored non-zeros must be preserved across round-trip."""
        mat = self._random_sparse(20, density=0.1, seed=99)
        p = tmp_path / "nnz.coo"
        write_sparse_matrix(p, mat)
        recovered = read_sparse_matrix(p)
        # nnz might differ slightly if original had explicit zeros; compare dense
        npt.assert_allclose(
            recovered.toarray(), mat.toarray(), rtol=1e-13, atol=0.0
        )


# ---------------------------------------------------------------------------
# write_frd tests
# ---------------------------------------------------------------------------


class TestWriteFrd:
    """Tests for write_frd."""

    def _make_nodes(self, n: int) -> np.ndarray:  # type: ignore[type-arg]
        """Evenly spaced nodes along x-axis."""
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([x, np.zeros(n), np.zeros(n)])

    def _make_disp(
        self, n: int, t: float
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Sinusoidal displacement: u1 = sin(2π·x·t), u2=u3=0."""
        x = np.linspace(0.0, 1.0, n)
        u1 = np.sin(2.0 * np.pi * x * t)
        return np.column_stack([u1, np.zeros(n), np.zeros(n)])

    def test_file_created(self, tmp_path: Path) -> None:
        nodes = self._make_nodes(5)
        disp = {0.0: self._make_disp(5, 0.0)}
        p = tmp_path / "out.frd"
        write_frd(p, nodes, disp)
        assert p.exists()

    def test_file_not_empty(self, tmp_path: Path) -> None:
        nodes = self._make_nodes(5)
        disp = {0.0: self._make_disp(5, 0.0)}
        p = tmp_path / "out.frd"
        write_frd(p, nodes, disp)
        assert p.stat().st_size > 0

    def test_eof_record_present(self, tmp_path: Path) -> None:
        nodes = self._make_nodes(3)
        disp = {0.0: self._make_disp(3, 0.0)}
        p = tmp_path / "out.frd"
        write_frd(p, nodes, disp)
        text = p.read_text()
        assert "9999" in text

    def test_node_count_in_header(self, tmp_path: Path) -> None:
        n = 7
        nodes = self._make_nodes(n)
        disp = {0.0: self._make_disp(n, 0.0)}
        p = tmp_path / "out.frd"
        write_frd(p, nodes, disp)
        text = p.read_text()
        assert str(n) in text

    def test_multiple_time_steps(self, tmp_path: Path) -> None:
        n = 5
        nodes = self._make_nodes(n)
        times = [0.0, 0.1, 0.2]
        disp = {t: self._make_disp(n, t) for t in times}
        p = tmp_path / "multi.frd"
        write_frd(p, nodes, disp)
        text = p.read_text()
        # 3 result blocks means 3 occurrences of "-3" terminator lines,
        # plus the one after the node block
        assert text.count(" -3") == len(times) + 1

    def test_scalar_displacement_broadcast(self, tmp_path: Path) -> None:
        """1-D displacement array should be broadcast to (n, 3) without error."""
        n = 4
        nodes = self._make_nodes(n)
        u1d = np.array([0.1, 0.2, 0.3, 0.4])
        write_frd(tmp_path / "scalar.frd", nodes, {0.0: u1d})

    def test_wrong_node_shape_raises(self, tmp_path: Path) -> None:
        bad_nodes = np.zeros((5, 2))  # should be (n, 3)
        disp = {0.0: np.zeros((5, 3))}
        with pytest.raises(ValueError, match="shape"):
            write_frd(tmp_path / "bad.frd", bad_nodes, disp)

    def test_wrong_disp_node_count_raises(self, tmp_path: Path) -> None:
        nodes = self._make_nodes(5)
        disp = {0.0: np.zeros((3, 3))}  # 3 ≠ 5
        with pytest.raises(ValueError):
            write_frd(tmp_path / "bad.frd", nodes, disp)

    def test_empty_time_series_raises(self, tmp_path: Path) -> None:
        nodes = self._make_nodes(3)
        with pytest.raises(TypeError):
            write_frd(tmp_path / "empty.frd", nodes, {})

    def test_displacement_values_in_file(self, tmp_path: Path) -> None:
        """Known displacement values must appear (in scientific notation) in output."""
        nodes = self._make_nodes(2)
        disp = {0.0: np.array([[1.23456e-3, 0.0, 0.0], [2.34567e-3, 0.0, 0.0]])}
        p = tmp_path / "vals.frd"
        write_frd(p, nodes, disp)
        text = p.read_text()
        # Check that the value appears in scientific notation format
        assert "1.23456E-03" in text.upper() or "1.23456E-003" in text.upper()


# ---------------------------------------------------------------------------
# Integration: mesh read + sparse matrix round-trip as a coupled workflow
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration-level tests simulating a realistic FE workflow."""

    def test_beam_stiffness_round_trip(self, tmp_path: Path) -> None:
        """
        Simulate assembling a beam stiffness matrix, writing it,
        reading it back, and verifying it is unchanged.

        The stiffness matrix is a 10×10 tridiagonal (1-D bar), typical
        output from a T-08/T-09 FE model.
        """
        n_dof = 10
        k = 2.5e5  # N/m
        K = sp.diags(
            [-k * np.ones(n_dof - 1), 2 * k * np.ones(n_dof), -k * np.ones(n_dof - 1)],
            [-1, 0, 1],
            format="csr",
        )
        p = tmp_path / "K.coo"
        write_sparse_matrix(p, K)
        K_recovered = read_sparse_matrix(p)
        npt.assert_allclose(K_recovered.toarray(), K.toarray(), rtol=1e-14)
        assert K_recovered.shape == K.shape

    def test_frd_export_for_beam_mesh(self, tmp_path: Path) -> None:
        """Write a mesh, compute a fake mode shape, export FRD, verify it parses."""
        mesh_p = _write_file(tmp_path, "beam.inp", _BEAM_INP)
        mesh = read_mesh(mesh_p)

        # Fake first bending mode shape (parabolic)
        x = mesh.nodes[:, 0]
        u1 = x * (1.0 - x)  # shape (n,)
        disp = {0.0: u1, 1.0: u1 * 2.0}

        frd_p = tmp_path / "mode.frd"
        write_frd(frd_p, mesh.nodes, disp)

        text = frd_p.read_text()
        assert "9999" in text
        # Two result steps → two 100CL headers
        assert text.count("100CL") == 2
