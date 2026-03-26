"""
Visualization functions for NLvib nonlinear vibration analysis results.

All functions return a ``matplotlib.figure.Figure`` and never call
``plt.show()`` or mutate global state.  An optional ``ax`` parameter lets
callers supply an existing axes, and an optional ``backend`` parameter
switches between ``"matplotlib"`` (default) and ``"plotly"`` (optional dep).

Design reference: NLvib MATLAB toolbox plotting conventions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# ---------------------------------------------------------------------------
# Public type protocol for continuation results
# ---------------------------------------------------------------------------


@runtime_checkable
class ContinuationResult(Protocol):
    """Minimal protocol for objects produced by the continuation solver.

    Any object that exposes ``omega``, ``amplitude``, and ``stability``
    attributes satisfies this protocol — no inheritance required.

    Attributes
    ----------
    omega:
        1-D array of angular frequencies (rad/s), shape ``(n_points,)``.
    amplitude:
        Array of amplitudes.  For FRF use shape ``(n_dof, n_harmonics, n)``
        or a 1-D array of length ``n`` for scalar problems.
    stability:
        1-D boolean array, shape ``(n_points,)``.  ``True`` → stable branch.
    """

    omega: npt.NDArray[np.float64]
    amplitude: npt.NDArray[np.float64]
    stability: npt.NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STABLE_STYLE: dict[str, object] = {"linestyle": "-", "label": "stable"}
_UNSTABLE_STYLE: dict[str, object] = {"linestyle": "--", "label": "unstable"}


def _make_fig_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return a (figure, axes) pair.

    If *ax* is provided the figure is obtained from it; otherwise a new
    figure with a single subplot is created.
    """
    if ax is not None:
        raw_fig = ax.get_figure()
        if raw_fig is None:
            raise ValueError("The supplied axes is not attached to any Figure.")
        if not isinstance(raw_fig, Figure):
            raise TypeError("The axes is attached to a SubFigure, not a top-level Figure.")
        return raw_fig, ax
    new_fig, new_ax = plt.subplots()
    assert isinstance(new_fig, Figure)
    return new_fig, new_ax


def _split_by_stability(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    stability: npt.NDArray[np.bool_],
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]]:
    """Split *x* and *y* into contiguous segments by stability flag.

    Returns a list of ``(x_seg, y_seg, is_stable)`` triples suitable for
    line-by-line plotting with different styles.
    """
    if x.shape != y.shape or x.shape != stability.shape:
        raise ValueError("x, y, and stability must share the same shape.")

    segments: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]] = []
    if len(x) == 0:
        return segments

    current_stable = bool(stability[0])
    start = 0
    for i in range(1, len(x)):
        if bool(stability[i]) != current_stable:
            segments.append((x[start:i], y[start:i], current_stable))
            start = i
            current_stable = bool(stability[i])
    segments.append((x[start:], y[start:], current_stable))
    return segments


# ---------------------------------------------------------------------------
# 1. Frequency Response Function
# ---------------------------------------------------------------------------


def plot_frf(
    result: ContinuationResult,
    dof: int = 0,
    harmonic: int = 1,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot the frequency response function (FRF) amplitude vs. Ω.

    Stable branches are drawn with a solid line; unstable branches with a
    dashed line.  Colours follow the default matplotlib cycle.

    Parameters
    ----------
    result:
        Continuation result exposing ``omega`` (shape ``(n,)``),
        ``amplitude`` (shape ``(n_dof, n_harmonics, n)`` or ``(n,)`` for
        scalar problems), and ``stability`` (shape ``(n,)``).
    dof:
        Zero-based DOF index to extract from ``result.amplitude``.
    harmonic:
        Harmonic number (1-based) to extract.
    ax:
        Optional existing axes to draw into.
    backend:
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_frf_plotly(result, dof=dof, harmonic=harmonic)

    fig, axes = _make_fig_ax(ax)

    omega = np.asarray(result.omega, dtype=np.float64)
    amp_raw = np.asarray(result.amplitude, dtype=np.float64)
    stability = np.asarray(result.stability, dtype=np.bool_)

    # Support both scalar (1-D) and multi-DOF/multi-harmonic (3-D) arrays.
    if amp_raw.ndim == 1:
        amp = amp_raw
    elif amp_raw.ndim == 2:
        amp = amp_raw[dof, :]
    else:
        amp = amp_raw[dof, harmonic - 1, :]

    stable_legend_added = False
    unstable_legend_added = False
    for x_seg, y_seg, is_stable in _split_by_stability(omega, amp, stability):
        label: str | None = None
        if is_stable and not stable_legend_added:
            label = "stable"
            stable_legend_added = True
        elif not is_stable and not unstable_legend_added:
            label = "unstable"
            unstable_legend_added = True
        style = _STABLE_STYLE if is_stable else _UNSTABLE_STYLE
        axes.plot(
            x_seg,
            y_seg,
            linestyle=str(style["linestyle"]),
            label=label,
        )

    axes.set_xlabel(r"Excitation frequency $\Omega$ (rad/s)")
    axes.set_ylabel(f"Amplitude |q_{dof}| (harmonic {harmonic})")
    axes.set_title("Frequency Response Function")
    if stable_legend_added or unstable_legend_added:
        axes.legend()

    return fig


def _plot_frf_plotly(
    result: ContinuationResult,
    dof: int = 0,
    harmonic: int = 1,
) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    omega = np.asarray(result.omega, dtype=np.float64)
    amp_raw = np.asarray(result.amplitude, dtype=np.float64)
    stability = np.asarray(result.stability, dtype=np.bool_)

    if amp_raw.ndim == 1:
        amp = amp_raw
    elif amp_raw.ndim == 2:
        amp = amp_raw[dof, :]
    else:
        amp = amp_raw[dof, harmonic - 1, :]

    plotly_fig = go.Figure()
    for x_seg, y_seg, is_stable in _split_by_stability(omega, amp, stability):
        plotly_fig.add_trace(
            go.Scatter(
                x=x_seg.tolist(),
                y=y_seg.tolist(),
                mode="lines",
                line={"dash": "solid" if is_stable else "dash"},
                name="stable" if is_stable else "unstable",
            )
        )
    plotly_fig.update_layout(
        xaxis_title="Ω (rad/s)",
        yaxis_title=f"Amplitude |q_{dof}| (harmonic {harmonic})",
        title="Frequency Response Function",
    )

    # Wrap plotly figure in a matplotlib Figure for API uniformity.
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 2. Backbone Curve
# ---------------------------------------------------------------------------


def plot_backbone(
    result: ContinuationResult,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot the backbone curve: modal frequency vs. modal amplitude.

    For nonlinear modal analysis (NMA) results the continuation parameter is
    the modal amplitude and ``result.omega`` holds the instantaneous
    frequency.

    Parameters
    ----------
    result:
        NMA continuation result.  ``result.amplitude`` is treated as a
        scalar modal amplitude (1-D array of length ``n``).
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_backbone_plotly(result)

    fig, axes = _make_fig_ax(ax)

    omega = np.asarray(result.omega, dtype=np.float64)
    amp_raw = np.asarray(result.amplitude, dtype=np.float64)
    modal_amp = amp_raw.ravel() if amp_raw.ndim > 1 else amp_raw

    axes.plot(modal_amp, omega, color="tab:blue", linewidth=1.5)
    axes.set_xlabel("Modal amplitude")
    axes.set_ylabel(r"Natural frequency $\omega_n$ (rad/s)")
    axes.set_title("Backbone Curve")

    return fig


def _plot_backbone_plotly(result: ContinuationResult) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    omega = np.asarray(result.omega, dtype=np.float64)
    amp_raw = np.asarray(result.amplitude, dtype=np.float64)
    modal_amp = amp_raw.ravel() if amp_raw.ndim > 1 else amp_raw

    plotly_fig = go.Figure(
        go.Scatter(x=modal_amp.tolist(), y=omega.tolist(), mode="lines")
    )
    plotly_fig.update_layout(
        xaxis_title="Modal amplitude",
        yaxis_title="ωₙ (rad/s)",
        title="Backbone Curve",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 3. Time Series
# ---------------------------------------------------------------------------


def plot_time_series(
    t: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    dq: npt.NDArray[np.float64] | None = None,
    dof: int = 0,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot steady-state time-domain displacement (and optionally velocity).

    Parameters
    ----------
    t:
        Time vector, shape ``(n_samples,)``.
    q:
        Displacement array.  Shape ``(n_samples,)`` for scalar DOF or
        ``(n_dof, n_samples)`` for multi-DOF.
    dq:
        Velocity array, same shape as *q*.  If supplied a second subplot is
        added below the displacement trace.
    dof:
        Zero-based DOF index (used when *q* is 2-D).
    ax:
        Optional axes to draw the displacement trace into.  If *dq* is also
        supplied, this axes receives displacement and an additional axes for
        velocity is created in the same figure.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_time_series_plotly(t, q, dq=dq, dof=dof)

    t_arr = np.asarray(t, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    q_dof = q_arr[dof, :] if q_arr.ndim == 2 else q_arr

    if dq is not None:
        dq_arr = np.asarray(dq, dtype=np.float64)
        dq_dof = dq_arr[dof, :] if dq_arr.ndim == 2 else dq_arr

        if ax is not None:
            # Caller-supplied ax → use it for displacement; add velocity axes.
            disp_ax = ax
            raw_fig = disp_ax.get_figure()
            if raw_fig is None:
                raise ValueError("The supplied axes is not attached to any Figure.")
            if not isinstance(raw_fig, Figure):
                raise TypeError(
                    "The axes is attached to a SubFigure, not a top-level Figure."
                )
            fig: Figure = raw_fig
            pos = disp_ax.get_position()
            vel_ax: Axes = fig.add_axes(
                (
                    pos.x0,
                    pos.y0 - pos.height * 0.6,
                    pos.width,
                    pos.height * 0.5,
                )
            )
        else:
            sub_fig, (disp_ax, vel_ax) = plt.subplots(2, 1, sharex=True)
            assert isinstance(sub_fig, Figure)
            fig = sub_fig

        disp_ax.plot(t_arr, q_dof, color="tab:blue")
        disp_ax.set_ylabel(f"Displacement $q_{{{dof}}}$")
        disp_ax.set_title("Time Series")

        vel_ax.plot(t_arr, dq_dof, color="tab:orange")
        vel_ax.set_xlabel("Time (s)")
        vel_ax.set_ylabel(r"Velocity $\dot{q}_{" + str(dof) + r"}$")
    else:
        fig, axes = _make_fig_ax(ax)
        axes.plot(t_arr, q_dof, color="tab:blue")
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(f"Displacement $q_{{{dof}}}$")
        axes.set_title("Time Series")

    return fig


def _plot_time_series_plotly(
    t: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    dq: npt.NDArray[np.float64] | None = None,
    dof: int = 0,
) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    t_arr = np.asarray(t, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    q_dof = q_arr[dof, :] if q_arr.ndim == 2 else q_arr

    if dq is not None:
        dq_arr = np.asarray(dq, dtype=np.float64)
        dq_dof = dq_arr[dof, :] if dq_arr.ndim == 2 else dq_arr
        plotly_fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        plotly_fig.add_trace(
            go.Scatter(x=t_arr.tolist(), y=q_dof.tolist(), name=f"q_{dof}"), row=1, col=1
        )
        plotly_fig.add_trace(
            go.Scatter(x=t_arr.tolist(), y=dq_dof.tolist(), name=f"dq_{dof}"), row=2, col=1
        )
        plotly_fig.update_yaxes(title_text=f"q_{dof}", row=1, col=1)
        plotly_fig.update_yaxes(title_text=f"dq_{dof}", row=2, col=1)
        plotly_fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    else:
        plotly_fig = go.Figure(
            go.Scatter(x=t_arr.tolist(), y=q_dof.tolist(), name=f"q_{dof}")
        )
        plotly_fig.update_layout(xaxis_title="Time (s)", yaxis_title=f"q_{dof}")
    plotly_fig.update_layout(title="Time Series")
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 4. Phase Portrait
# ---------------------------------------------------------------------------


def plot_phase_portrait(
    t: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    dq: npt.NDArray[np.float64],
    dof: int = 0,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    r"""Plot the phase portrait: velocity q̇ vs. displacement q.

    Parameters
    ----------
    t:
        Time vector (unused in the plot but kept for API symmetry with
        :func:`plot_time_series`).
    q:
        Displacement array, shape ``(n_samples,)`` or ``(n_dof, n_samples)``.
    dq:
        Velocity array, same shape as *q*.
    dof:
        Zero-based DOF index.
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_phase_portrait_plotly(t, q, dq, dof=dof)

    q_arr = np.asarray(q, dtype=np.float64)
    dq_arr = np.asarray(dq, dtype=np.float64)
    q_dof = q_arr[dof, :] if q_arr.ndim == 2 else q_arr
    dq_dof = dq_arr[dof, :] if dq_arr.ndim == 2 else dq_arr

    fig, axes = _make_fig_ax(ax)
    axes.plot(q_dof, dq_dof, color="tab:purple", linewidth=1.0)
    axes.set_xlabel(f"$q_{{{dof}}}$")
    axes.set_ylabel(r"$\dot{q}_{" + str(dof) + r"}$")
    axes.set_title("Phase Portrait")

    return fig


def _plot_phase_portrait_plotly(
    t: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    dq: npt.NDArray[np.float64],
    dof: int = 0,
) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    q_arr = np.asarray(q, dtype=np.float64)
    dq_arr = np.asarray(dq, dtype=np.float64)
    q_dof = q_arr[dof, :] if q_arr.ndim == 2 else q_arr
    dq_dof = dq_arr[dof, :] if dq_arr.ndim == 2 else dq_arr

    plotly_fig = go.Figure(
        go.Scatter(x=q_dof.tolist(), y=dq_dof.tolist(), mode="lines")
    )
    plotly_fig.update_layout(
        xaxis_title=f"q_{dof}",
        yaxis_title=f"dq_{dof}",
        title="Phase Portrait",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 5. Floquet Multipliers
# ---------------------------------------------------------------------------


def plot_floquet(
    multipliers: npt.NDArray[np.complex128],
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot Floquet multipliers on the complex plane with the unit circle.

    Multipliers inside the unit circle indicate a stable periodic orbit.
    Multipliers outside the unit circle indicate instability.

    Parameters
    ----------
    multipliers:
        Complex array of Floquet multipliers, shape ``(n,)``.
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_floquet_plotly(multipliers)

    fig, axes = _make_fig_ax(ax)

    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    axes.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, label="unit circle")

    mults = np.asarray(multipliers, dtype=np.complex128)
    inside = np.abs(mults) <= 1.0
    if np.any(inside):
        axes.scatter(
            mults[inside].real,
            mults[inside].imag,
            color="tab:blue",
            marker="o",
            label="stable",
            zorder=3,
        )
    if np.any(~inside):
        axes.scatter(
            mults[~inside].real,
            mults[~inside].imag,
            color="tab:red",
            marker="x",
            label="unstable",
            zorder=3,
        )

    axes.axhline(0, color="gray", linewidth=0.5)
    axes.axvline(0, color="gray", linewidth=0.5)
    axes.set_aspect("equal")
    axes.set_xlabel(r"$\operatorname{Re}(\mu)$")
    axes.set_ylabel(r"$\operatorname{Im}(\mu)$")
    axes.set_title("Floquet Multipliers")
    axes.legend()

    return fig


def _plot_floquet_plotly(multipliers: npt.NDArray[np.complex128]) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    mults = np.asarray(multipliers, dtype=np.complex128)
    inside = np.abs(mults) <= 1.0

    traces = [
        go.Scatter(
            x=np.cos(theta).tolist(),
            y=np.sin(theta).tolist(),
            mode="lines",
            line={"dash": "dash", "color": "black"},
            name="unit circle",
        )
    ]
    if np.any(inside):
        traces.append(
            go.Scatter(
                x=mults[inside].real.tolist(),
                y=mults[inside].imag.tolist(),
                mode="markers",
                marker={"symbol": "circle"},
                name="stable",
            )
        )
    if np.any(~inside):
        traces.append(
            go.Scatter(
                x=mults[~inside].real.tolist(),
                y=mults[~inside].imag.tolist(),
                mode="markers",
                marker={"symbol": "x"},
                name="unstable",
            )
        )

    plotly_fig = go.Figure(traces)
    plotly_fig.update_layout(
        xaxis_title="Re(μ)",
        yaxis_title="Im(μ)",
        title="Floquet Multipliers",
        yaxis_scaleanchor="x",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 6. Mode Shape
# ---------------------------------------------------------------------------


def plot_mode_shape(
    nodes: npt.NDArray[np.float64],
    displacement: npt.NDArray[np.float64],
    title: str = "",
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot spatial mode shape for FE beam or rod models.

    Parameters
    ----------
    nodes:
        Nodal coordinates along the beam/rod axis, shape ``(n_nodes,)``.
        Typically the x-coordinates of each node.
    displacement:
        Modal displacement at each node, shape ``(n_nodes,)``.  For 2-D beam
        problems this is the transverse component.
    title:
        Optional title string (e.g., ``"Mode 1 — ω = 42.3 rad/s"``).
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_mode_shape_plotly(nodes, displacement, title=title)

    nodes_arr = np.asarray(nodes, dtype=np.float64)
    disp_arr = np.asarray(displacement, dtype=np.float64)

    fig, axes = _make_fig_ax(ax)
    axes.plot(nodes_arr, np.zeros_like(nodes_arr), "k--", linewidth=0.6, label="undeformed")
    axes.plot(nodes_arr, disp_arr, color="tab:green", linewidth=1.5, label="mode shape")
    axes.fill_between(nodes_arr, disp_arr, alpha=0.2, color="tab:green")

    axes.set_xlabel("Node position")
    axes.set_ylabel("Displacement")
    axes.set_title(title if title else "Mode Shape")
    axes.legend()

    return fig


def _plot_mode_shape_plotly(
    nodes: npt.NDArray[np.float64],
    displacement: npt.NDArray[np.float64],
    title: str = "",
) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    nodes_arr = np.asarray(nodes, dtype=np.float64)
    disp_arr = np.asarray(displacement, dtype=np.float64)

    plotly_fig = go.Figure(
        [
            go.Scatter(
                x=nodes_arr.tolist(),
                y=np.zeros(len(nodes_arr)).tolist(),
                mode="lines",
                line={"dash": "dash", "color": "black"},
                name="undeformed",
            ),
            go.Scatter(
                x=nodes_arr.tolist(),
                y=disp_arr.tolist(),
                mode="lines",
                name="mode shape",
                fill="tozeroy",
            ),
        ]
    )
    plotly_fig.update_layout(
        xaxis_title="Node position",
        yaxis_title="Displacement",
        title=title if title else "Mode Shape",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 7. Harmonic Content
# ---------------------------------------------------------------------------


def plot_harmonic_content(
    Q_harmonics: npt.NDArray[np.float64],
    omega: float,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Bar chart of harmonic amplitudes Q₁, Q₃, Q₅, …

    Parameters
    ----------
    Q_harmonics:
        Array of harmonic amplitudes, shape ``(n_harmonics,)``.
        Index 0 → 1st harmonic (fundamental), index 1 → 2nd harmonic, etc.
    omega:
        Fundamental angular frequency (rad/s).  Used only in the x-axis label.
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_harmonic_content_plotly(Q_harmonics, omega)

    Q_raw = np.asarray(Q_harmonics)
    Q_abs = np.abs(Q_raw).astype(np.float64)
    n = len(Q_abs)
    harmonic_indices = np.arange(1, n + 1)

    fig, axes = _make_fig_ax(ax)
    axes.bar(harmonic_indices, Q_abs, color="tab:cyan", edgecolor="black", linewidth=0.7)
    axes.set_xticks(harmonic_indices)
    axes.set_xticklabels([f"H{k}" for k in harmonic_indices])
    axes.set_xlabel(f"Harmonic (Ω = {omega:.4g} rad/s)")
    axes.set_ylabel("Amplitude")
    axes.set_title("Harmonic Content")

    return fig


def _plot_harmonic_content_plotly(
    Q_harmonics: npt.NDArray[np.float64], omega: float
) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    Q_raw = np.asarray(Q_harmonics)
    Q_abs = np.abs(Q_raw).astype(np.float64)
    n = len(Q_abs)
    harmonic_indices = list(range(1, n + 1))
    labels = [f"H{k}" for k in harmonic_indices]

    plotly_fig = go.Figure(go.Bar(x=labels, y=Q_abs.tolist()))
    plotly_fig.update_layout(
        xaxis_title=f"Harmonic (Ω = {omega:.4g} rad/s)",
        yaxis_title="Amplitude",
        title="Harmonic Content",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig


# ---------------------------------------------------------------------------
# 8. Convergence
# ---------------------------------------------------------------------------


def plot_convergence(
    residuals: npt.NDArray[np.float64],
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Plot residual norm vs. continuation step or Newton iteration (semilogy).

    Parameters
    ----------
    residuals:
        1-D array of residual norms, one per step or iteration.
    ax:
        Optional existing axes.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend == "plotly":
        return _plot_convergence_plotly(residuals)

    res = np.asarray(residuals, dtype=np.float64)
    steps = np.arange(1, len(res) + 1)

    fig, axes = _make_fig_ax(ax)
    axes.semilogy(steps, res, color="tab:red", marker="o", markersize=4, linewidth=1.2)
    axes.set_xlabel("Iteration / step")
    axes.set_ylabel("Residual norm")
    axes.set_title("Convergence")
    axes.grid(True, which="both", linestyle="--", linewidth=0.4)

    return fig


def _plot_convergence_plotly(residuals: npt.NDArray[np.float64]) -> Figure:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install it with: pip install plotly"
        ) from exc

    res = np.asarray(residuals, dtype=np.float64)
    steps = list(range(1, len(res) + 1))

    plotly_fig = go.Figure(
        go.Scatter(x=steps, y=res.tolist(), mode="lines+markers")
    )
    plotly_fig.update_layout(
        xaxis_title="Iteration / step",
        yaxis_title="Residual norm",
        yaxis_type="log",
        title="Convergence",
    )
    wrap_fig, wrap_ax = plt.subplots()
    assert isinstance(wrap_fig, Figure)
    wrap_ax.set_visible(False)
    wrap_fig._nlvib_plotly_fig = plotly_fig  # type: ignore[attr-defined]
    return wrap_fig
