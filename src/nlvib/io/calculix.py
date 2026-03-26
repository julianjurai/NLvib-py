"""
CalculiX IO parsers for the NLvib Python port.

Supports reading CalculiX mesh files (.inp), sparse matrix files (custom
COO text format), and writing CalculiX FRD result files (.frd) for
post-processing and animation export.

CalculiX FRD format reference:
    Dhondt, G. (2004). *The Finite Element Method for Three-dimensional
    Thermomechanical Applications*. Wiley. Appendix B (FRD file specification).
    Also: CalculiX CrunchiX User's Manual, Chapter 7.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# FRD format record-type identifiers
_FRD_HEADER_KEY = "1C"
_FRD_NODE_KEY = "2C"
_FRD_ELEMENT_KEY = "3C"
_FRD_RESULT_HEADER = "100CL"
_FRD_RESULT_NODE = "-1"
_FRD_RESULT_EOF = "9999"

# COO sparse matrix file column indices
_COO_ROW_COL = 0
_COO_COL_COL = 1
_COO_VAL_COL = 2

# Minimum dimension for which scipy.sparse is required
_SPARSE_MIN_DIM = 10

# FRD field widths (fixed-format columns)
_FRD_NODE_LABEL_WIDTH = 10
_FRD_COORD_WIDTH = 12
_FRD_DISP_WIDTH = 12


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class MeshData(NamedTuple):
    """Container for CalculiX mesh data.

    Attributes
    ----------
    nodes:
        Float array of shape ``(n_nodes, 3)`` with node coordinates
        ``[x, y, z]`` in the order they appear in the file.
    node_ids:
        Integer array of shape ``(n_nodes,)`` — original CalculiX node labels
        (1-based).  Use this to map element connectivity back to row indices.
    elements:
        Integer array of shape ``(n_elements, max_nodes_per_element)`` with
        0-based indices into *nodes* / *node_ids*.  Rows are padded with -1
        for element types that have fewer nodes than the maximum.
    element_ids:
        Integer array of shape ``(n_elements,)`` — original CalculiX element
        labels (1-based).
    element_type:
        String identifier of the element type encountered (e.g. ``"C3D8"``
        for a linear hexahedron, ``"B31"`` for a 2-node Bernoulli beam).
        If multiple element types exist the dominant type is returned.
    """

    nodes: npt.NDArray[np.float64]
    node_ids: npt.NDArray[np.int64]
    elements: npt.NDArray[np.int64]
    element_ids: npt.NDArray[np.int64]
    element_type: str


# ---------------------------------------------------------------------------
# Public API — read_mesh
# ---------------------------------------------------------------------------


def read_mesh(path: str | Path) -> MeshData:
    """Parse a CalculiX input file (``.inp``) and extract mesh data.

    Reads ``*NODE`` and ``*ELEMENT`` sections from a CalculiX ``.inp``
    keyword-based input deck.  Only the first ``*ELEMENT`` block is parsed;
    if multiple element types are present the caller should split the file
    first.

    Parameters
    ----------
    path:
        Path to the ``.inp`` file.

    Returns
    -------
    MeshData
        Named tuple with fields ``nodes``, ``node_ids``, ``elements``,
        ``element_ids``, ``element_type``.  See :class:`MeshData` for
        shapes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains no ``*NODE`` or ``*ELEMENT`` section.

    Notes
    -----
    Node coordinates are stored as ``float64``.  Element connectivity is
    converted from 1-based CalculiX labels to 0-based row indices into the
    ``nodes`` array via ``node_ids``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    node_ids_list: list[int] = []
    coords_list: list[list[float]] = []
    element_ids_list: list[int] = []
    conn_list: list[list[int]] = []
    element_type = ""

    mode = "none"  # "nodes" | "elements" | "none"

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("**"):
            # blank line or CalculiX comment — reset mode if not inside data
            continue

        if line.upper().startswith("*NODE"):
            mode = "nodes"
            continue

        if line.upper().startswith("*ELEMENT"):
            mode = "elements"
            element_type = _parse_element_type(line)
            continue

        # Any other keyword line ends the current data block
        if line.startswith("*"):
            mode = "none"
            continue

        if mode == "nodes":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            node_ids_list.append(int(parts[0]))
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

        elif mode == "elements":
            # CalculiX may split long element lines with continuation
            parts = [p.strip() for p in line.split(",")]
            numeric = []
            for p in parts:
                if p:
                    numeric.append(int(p))
            if not numeric:
                continue
            # First token is the element label
            element_ids_list.append(numeric[0])
            conn_list.append(numeric[1:])

    if not node_ids_list:
        raise ValueError(f"No *NODE section found in {path}")
    if not element_ids_list:
        raise ValueError(f"No *ELEMENT section found in {path}")

    # Build arrays
    nodes_arr = np.array(coords_list, dtype=np.float64)
    node_ids_arr = np.array(node_ids_list, dtype=np.int64)

    # Map CalculiX 1-based labels → 0-based row indices
    label_to_row: dict[int, int] = {label: idx for idx, label in enumerate(node_ids_list)}

    max_conn = max(len(c) for c in conn_list)
    n_el = len(element_ids_list)
    elements_arr = np.full((n_el, max_conn), -1, dtype=np.int64)
    for i, conn in enumerate(conn_list):
        mapped = [label_to_row[lbl] for lbl in conn]
        elements_arr[i, : len(mapped)] = mapped

    element_ids_arr = np.array(element_ids_list, dtype=np.int64)

    return MeshData(
        nodes=nodes_arr,
        node_ids=node_ids_arr,
        elements=elements_arr,
        element_ids=element_ids_arr,
        element_type=element_type,
    )


def _parse_element_type(line: str) -> str:
    """Extract element type keyword from a ``*ELEMENT`` header line.

    Parameters
    ----------
    line:
        Raw ``*ELEMENT`` header line, e.g.
        ``*ELEMENT, TYPE=C3D8, ELSET=BODY``.

    Returns
    -------
    str
        Upper-case element type string, e.g. ``"C3D8"``.  Returns ``""``
        if no ``TYPE=`` token is found.
    """
    match = re.search(r"TYPE\s*=\s*(\w+)", line, re.IGNORECASE)
    return match.group(1).upper() if match else ""


# ---------------------------------------------------------------------------
# Public API — read_sparse_matrix
# ---------------------------------------------------------------------------


def read_sparse_matrix(path: str | Path) -> sp.csr_matrix:
    """Read a sparse matrix stored in COO (coordinate) text format.

    The file must contain one non-zero entry per line with three
    whitespace-separated values::

        <row>  <col>  <value>

    Row and column indices are **1-based** (CalculiX convention) and are
    converted to 0-based internally.  The matrix shape is inferred from the
    maximum row and column indices in the file.

    Parameters
    ----------
    path:
        Path to the COO text file.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix in Compressed Sparse Row format.  Always
        ``dtype=float64``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty or contains unparseable lines.

    Notes
    -----
    ``scipy.sparse`` is always used here regardless of matrix size because
    this function is specifically for sparse-format inputs.  For matrices
    with dimension ≥ :data:`_SPARSE_MIN_DIM` the caller should never convert
    to dense.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sparse matrix file not found: {path}")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Expected 3 columns (row col val) on line {lineno} of {path}; "
                    f"got: {line!r}"
                )
            try:
                rows.append(int(parts[_COO_ROW_COL]) - 1)  # 1-based → 0-based
                cols.append(int(parts[_COO_COL_COL]) - 1)
                vals.append(float(parts[_COO_VAL_COL]))
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse line {lineno} of {path}: {line!r}"
                ) from exc

    if not rows:
        raise ValueError(f"No data found in sparse matrix file: {path}")

    n_rows = max(rows) + 1
    n_cols = max(cols) + 1

    rows_arr = np.array(rows, dtype=np.int32)
    cols_arr = np.array(cols, dtype=np.int32)
    vals_arr = np.array(vals, dtype=np.float64)

    mat: sp.csr_matrix = sp.coo_matrix(
        (vals_arr, (rows_arr, cols_arr)), shape=(n_rows, n_cols)
    ).tocsr()
    return mat


# ---------------------------------------------------------------------------
# Public API — write_frd
# ---------------------------------------------------------------------------


def write_frd(
    path: str | Path,
    nodes: npt.NDArray[np.float64],
    time_series: dict[float, npt.NDArray[np.float64]],
    *,
    title: str = "NLvib result",
) -> None:
    """Write nodal displacement results to a CalculiX FRD file.

    Produces a text file in CalculiX FRD format (version ``CALCULIX 2.x``).
    The file can be opened with CGX (CalculiX GraphiX) for post-processing
    and animation.

    FRD format reference: CalculiX CrunchiX User's Manual, Chapter 7 —
    Output file formats.

    Parameters
    ----------
    path:
        Destination ``.frd`` file path.  Parent directories must exist.
    nodes:
        Float array of shape ``(n_nodes, 3)`` — node coordinates
        ``[x, y, z]``.  Node labels are assigned as 1-based integers
        ``1 … n_nodes``.
    time_series:
        Mapping ``{time_value: displacement_array}`` where each
        *displacement_array* has shape ``(n_nodes, 3)`` (or ``(n_nodes,)``
        for scalar DOF results, which are broadcast to 3 components with
        zeros in y/z).  Time values define the animation frames.
    title:
        Optional one-line title string embedded in the FRD header (max 66
        characters; truncated if longer).

    Raises
    ------
    ValueError
        If *nodes* is not 2-D with 3 columns, or if any displacement array
        has the wrong number of nodes.
    TypeError
        If *time_series* is empty.

    Notes
    -----
    FRD is a fixed-column-width ASCII format.  Field widths follow the
    specification:

    - Node label: 10 characters, right-justified.
    - Coordinate / displacement values: 12 characters each (``%12.5E``).

    The result block written is ``DISP`` (displacement), which CGX
    interprets as vector field ``U1, U2, U3``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            f"nodes must have shape (n_nodes, 3); got {nodes.shape}"
        )
    n_nodes = nodes.shape[0]

    if not time_series:
        raise TypeError("time_series must not be empty")

    path = Path(path)

    with path.open("w", encoding="utf-8") as fh:

        # ----------------------------------------------------------------
        # File header block (record type 1C)
        # ----------------------------------------------------------------
        title_trunc = title[:66]
        fh.write(f"    1C{title_trunc:<66s}                 0\n")

        # ----------------------------------------------------------------
        # Node block (record type 2C)
        # ----------------------------------------------------------------
        fh.write(f"    2C                                                                  {n_nodes:>8d}\n")
        for i in range(n_nodes):
            label = i + 1  # 1-based
            x, y, z = nodes[i, 0], nodes[i, 1], nodes[i, 2]
            fh.write(
                f" -1{label:{_FRD_NODE_LABEL_WIDTH}d}"
                f"{x:{_FRD_COORD_WIDTH}.5E}"
                f"{y:{_FRD_COORD_WIDTH}.5E}"
                f"{z:{_FRD_COORD_WIDTH}.5E}\n"
            )
        fh.write(" -3\n")

        # ----------------------------------------------------------------
        # Result blocks — one per time step
        # ----------------------------------------------------------------
        for step_idx, (t_val, disp) in enumerate(sorted(time_series.items()), start=1):
            disp_arr = np.asarray(disp, dtype=np.float64)
            if disp_arr.ndim == 1:
                if disp_arr.shape[0] != n_nodes:
                    raise ValueError(
                        f"Displacement array at t={t_val} has {disp_arr.shape[0]} "
                        f"entries but nodes has {n_nodes} rows."
                    )
                # Broadcast scalar DOF → (n_nodes, 3) with zeros in y/z
                disp_arr = np.column_stack(
                    [disp_arr, np.zeros(n_nodes), np.zeros(n_nodes)]
                )
            elif disp_arr.ndim == 2:
                if disp_arr.shape != (n_nodes, 3):
                    raise ValueError(
                        f"Displacement array at t={t_val} must have shape "
                        f"({n_nodes}, 3); got {disp_arr.shape}."
                    )
            else:
                raise ValueError(
                    f"Displacement array at t={t_val} must be 1-D or 2-D; "
                    f"got ndim={disp_arr.ndim}."
                )

            # Result header (100CL record)
            fh.write(
                f" 100CL{step_idx:>5d}"
                f"{t_val:12.5E}"
                f"{'DISP':>20s}"
                f"{'DISP':>8s}"
                f"{'4':>2s}\n"
            )
            # Component labels: U1, U2, U3 (3 displacement components)
            fh.write(" -4  DISP        4    1\n")
            fh.write(" -5  D1          1    2    1    0\n")
            fh.write(" -5  D2          1    2    2    0\n")
            fh.write(" -5  D3          1    2    3    0\n")
            fh.write(" -5  ALL         1    2    0    0    1ALL\n")

            # Nodal result values (-1 records)
            for i in range(n_nodes):
                label = i + 1
                u1, u2, u3 = disp_arr[i, 0], disp_arr[i, 1], disp_arr[i, 2]
                fh.write(
                    f" -1{label:{_FRD_NODE_LABEL_WIDTH}d}"
                    f"{u1:{_FRD_DISP_WIDTH}.5E}"
                    f"{u2:{_FRD_DISP_WIDTH}.5E}"
                    f"{u3:{_FRD_DISP_WIDTH}.5E}\n"
                )
            fh.write(" -3\n")

        # ----------------------------------------------------------------
        # End-of-file record
        # ----------------------------------------------------------------
        fh.write(f"  {_FRD_RESULT_EOF}\n")


# ---------------------------------------------------------------------------
# Public API — write_sparse_matrix  (complement for round-trip tests)
# ---------------------------------------------------------------------------


def write_sparse_matrix(
    path: str | Path,
    matrix: sp.spmatrix,
) -> None:
    """Write a sparse matrix to a COO text file (1-based indices).

    Produces the format expected by :func:`read_sparse_matrix`::

        <row>  <col>  <value>

    Row and column indices are written as **1-based** integers (CalculiX
    convention).

    Parameters
    ----------
    path:
        Destination file path.
    matrix:
        Any ``scipy.sparse`` matrix.  Converted to COO format internally.
        Zero entries are not written.

    Raises
    ------
    TypeError
        If *matrix* is not a ``scipy.sparse`` matrix.
    """
    if not sp.issparse(matrix):
        raise TypeError(
            f"matrix must be a scipy.sparse matrix; got {type(matrix)}"
        )

    path = Path(path)
    coo = matrix.tocoo()

    with path.open("w", encoding="utf-8") as fh:
        fh.write("# row  col  value  (1-based indices)\n")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            fh.write(f"{int(r) + 1}  {int(c) + 1}  {v:.17g}\n")
