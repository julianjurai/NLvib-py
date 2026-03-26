"""
Smoke tests for nlvib.visualization.

All tests use synthetic data — no real solver is required.  Tests verify:
  - Each function returns a matplotlib.figure.Figure
  - Axis labels are set correctly
  - Legend entries are present where expected
  - The ``ax=`` injection API works
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pytest
from dataclasses import dataclass
from matplotlib.figure import Figure

matplotlib.use("Agg")  # non-interactive backend for CI

from nlvib.visualization import (
    plot_backbone,
    plot_convergence,
    plot_floquet,
    plot_frf,
    plot_harmonic_content,
    plot_mode_shape,
    plot_phase_portrait,
    plot_time_series,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeContinuationResult:
    omega: np.ndarray
    amplitude: np.ndarray
    stability: np.ndarray


def _make_frf_result(n: int = 40) -> _FakeContinuationResult:
    """Single-DOF, single-harmonic scalar FRF result."""
    omega = np.linspace(0.8, 1.2, n)
    amplitude = 1.0 / np.abs(1.0 - omega**2 + 0.05j * omega)
    # First half stable, second half unstable
    stability = np.array([True] * (n // 2) + [False] * (n - n // 2))
    return _FakeContinuationResult(omega=omega, amplitude=amplitude, stability=stability)


def _make_frf_result_3d(n: int = 40, n_dof: int = 2, n_harm: int = 3) -> _FakeContinuationResult:
    """Multi-DOF, multi-harmonic FRF result (3-D amplitude)."""
    omega = np.linspace(0.5, 1.5, n)
    rng = np.random.default_rng(0)
    amplitude = rng.random((n_dof, n_harm, n))
    stability = np.ones(n, dtype=bool)
    return _FakeContinuationResult(omega=omega, amplitude=amplitude, stability=stability)


def _make_backbone_result(n: int = 30) -> _FakeContinuationResult:
    modal_amp = np.linspace(0.0, 2.0, n)
    omega = 1.0 + 0.1 * modal_amp**2  # hardening
    stability = np.ones(n, dtype=bool)
    return _FakeContinuationResult(omega=omega, amplitude=modal_amp, stability=stability)


# ---------------------------------------------------------------------------
# 1. plot_frf
# ---------------------------------------------------------------------------

class TestPlotFRF:
    def test_returns_figure(self) -> None:
        result = _make_frf_result()
        fig = plot_frf(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_3d_amplitude(self) -> None:
        result = _make_frf_result_3d()
        fig = plot_frf(result, dof=1, harmonic=2)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_axis_labels(self) -> None:
        result = _make_frf_result()
        fig = plot_frf(result)
        ax = fig.axes[0]
        assert "Ω" in ax.get_xlabel() or "Omega" in ax.get_xlabel() or "frequency" in ax.get_xlabel().lower()
        assert "Amplitude" in ax.get_ylabel() or "amplitude" in ax.get_ylabel()
        plt.close(fig)

    def test_legend_stable_unstable(self) -> None:
        result = _make_frf_result()
        fig = plot_frf(result)
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "stable" in labels
        assert "unstable" in labels
        plt.close(fig)

    def test_all_stable_no_unstable_legend(self) -> None:
        result = _make_frf_result()
        result.stability = np.ones(len(result.omega), dtype=bool)
        fig = plot_frf(result)
        ax = fig.axes[0]
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()] if legend else []
        assert "unstable" not in labels
        plt.close(fig)

    def test_ax_injection(self) -> None:
        result = _make_frf_result()
        external_fig, external_ax = plt.subplots()
        returned_fig = plot_frf(result, ax=external_ax)
        assert returned_fig is external_fig
        plt.close(external_fig)

    def test_plotly_raises_without_plotly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins
        real_import = builtins.__import__

        def _block_plotly(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("plotly"):
                raise ImportError("no plotly")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _block_plotly)
        result = _make_frf_result()
        with pytest.raises(ImportError, match="plotly"):
            plot_frf(result, backend="plotly")


# ---------------------------------------------------------------------------
# 2. plot_backbone
# ---------------------------------------------------------------------------

class TestPlotBackbone:
    def test_returns_figure(self) -> None:
        result = _make_backbone_result()
        fig = plot_backbone(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_axis_labels(self) -> None:
        result = _make_backbone_result()
        fig = plot_backbone(result)
        ax = fig.axes[0]
        assert "amplitude" in ax.get_xlabel().lower() or "modal" in ax.get_xlabel().lower()
        assert "frequency" in ax.get_ylabel().lower() or "omega" in ax.get_ylabel().lower() or "ω" in ax.get_ylabel()
        plt.close(fig)

    def test_title_set(self) -> None:
        result = _make_backbone_result()
        fig = plot_backbone(result)
        assert "Backbone" in fig.axes[0].get_title() or "backbone" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_ax_injection(self) -> None:
        result = _make_backbone_result()
        ext_fig, ext_ax = plt.subplots()
        returned = plot_backbone(result, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)


# ---------------------------------------------------------------------------
# 3. plot_time_series
# ---------------------------------------------------------------------------

class TestPlotTimeSeries:
    def setup_method(self) -> None:
        self.t = np.linspace(0, 2 * np.pi, 200)
        self.q = np.sin(self.t)
        self.dq = np.cos(self.t)

    def test_returns_figure_displacement_only(self) -> None:
        fig = plot_time_series(self.t, self.q)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_with_velocity(self) -> None:
        fig = plot_time_series(self.t, self.q, dq=self.dq)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_two_subplots_when_dq_given(self) -> None:
        fig = plot_time_series(self.t, self.q, dq=self.dq)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_one_subplot_without_dq(self) -> None:
        fig = plot_time_series(self.t, self.q)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_xlabel_time(self) -> None:
        fig = plot_time_series(self.t, self.q)
        ax = fig.axes[0]
        assert "time" in ax.get_xlabel().lower() or "t" in ax.get_xlabel().lower()
        plt.close(fig)

    def test_title_set(self) -> None:
        fig = plot_time_series(self.t, self.q)
        assert "Time" in fig.axes[0].get_title() or "time" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_multidof_array(self) -> None:
        q2d = np.vstack([self.q, self.q * 0.5])  # shape (2, 200)
        fig = plot_time_series(self.t, q2d, dof=1)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_ax_injection(self) -> None:
        ext_fig, ext_ax = plt.subplots()
        returned = plot_time_series(self.t, self.q, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)


# ---------------------------------------------------------------------------
# 4. plot_phase_portrait
# ---------------------------------------------------------------------------

class TestPlotPhasePortrait:
    def setup_method(self) -> None:
        t = np.linspace(0, 4 * np.pi, 400)
        self.t = t
        self.q = np.sin(t)
        self.dq = np.cos(t)

    def test_returns_figure(self) -> None:
        fig = plot_phase_portrait(self.t, self.q, self.dq)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_axis_labels(self) -> None:
        fig = plot_phase_portrait(self.t, self.q, self.dq)
        ax = fig.axes[0]
        assert "q" in ax.get_xlabel()
        assert "q" in ax.get_ylabel() or "dot" in ax.get_ylabel().lower() or "̇" in ax.get_ylabel()
        plt.close(fig)

    def test_title_set(self) -> None:
        fig = plot_phase_portrait(self.t, self.q, self.dq)
        assert "Phase" in fig.axes[0].get_title() or "phase" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_ax_injection(self) -> None:
        ext_fig, ext_ax = plt.subplots()
        returned = plot_phase_portrait(self.t, self.q, self.dq, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)

    def test_multidof(self) -> None:
        q2d = np.vstack([self.q, self.q * 2])
        dq2d = np.vstack([self.dq, self.dq * 2])
        fig = plot_phase_portrait(self.t, q2d, dq2d, dof=1)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. plot_floquet
# ---------------------------------------------------------------------------

class TestPlotFloquet:
    def _multipliers(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return (rng.random(10) - 0.5 + 1j * (rng.random(10) - 0.5)) * 1.5

    def test_returns_figure(self) -> None:
        fig = plot_floquet(self._multipliers())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_unit_circle_drawn(self) -> None:
        """At least one line with 300 points (the unit circle) should exist."""
        fig = plot_floquet(self._multipliers())
        ax = fig.axes[0]
        line_lengths = [len(line.get_xdata()) for line in ax.get_lines()]
        assert 300 in line_lengths
        plt.close(fig)

    def test_legend_contains_unit_circle(self) -> None:
        fig = plot_floquet(self._multipliers())
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("circle" in lbl.lower() or "unit" in lbl.lower() for lbl in labels)
        plt.close(fig)

    def test_axis_labels(self) -> None:
        fig = plot_floquet(self._multipliers())
        ax = fig.axes[0]
        assert "Re" in ax.get_xlabel() or "re" in ax.get_xlabel().lower()
        assert "Im" in ax.get_ylabel() or "im" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_ax_injection(self) -> None:
        ext_fig, ext_ax = plt.subplots()
        returned = plot_floquet(self._multipliers(), ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)

    def test_all_stable_multipliers(self) -> None:
        mults = np.array([0.5 + 0.3j, -0.2 + 0.1j, 0.0 - 0.8j])
        fig = plot_floquet(mults)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_all_unstable_multipliers(self) -> None:
        mults = np.array([1.5 + 0.3j, -2.0 + 0.1j, 3.0 - 0.8j])
        fig = plot_floquet(mults)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6. plot_mode_shape
# ---------------------------------------------------------------------------

class TestPlotModeShape:
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.linspace(0.0, 1.0, 20)
        displacement = np.sin(np.pi * nodes)
        return nodes, displacement

    def test_returns_figure(self) -> None:
        nodes, disp = self._data()
        fig = plot_mode_shape(nodes, disp)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self) -> None:
        nodes, disp = self._data()
        fig = plot_mode_shape(nodes, disp, title="Mode 1")
        assert "Mode 1" in fig.axes[0].get_title()
        plt.close(fig)

    def test_default_title(self) -> None:
        nodes, disp = self._data()
        fig = plot_mode_shape(nodes, disp)
        assert "Mode" in fig.axes[0].get_title() or "mode" in fig.axes[0].get_title().lower() or "Shape" in fig.axes[0].get_title()
        plt.close(fig)

    def test_legend_present(self) -> None:
        nodes, disp = self._data()
        fig = plot_mode_shape(nodes, disp)
        legend = fig.axes[0].get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("mode" in lbl.lower() or "shape" in lbl.lower() or "undeformed" in lbl.lower() for lbl in labels)
        plt.close(fig)

    def test_ax_injection(self) -> None:
        nodes, disp = self._data()
        ext_fig, ext_ax = plt.subplots()
        returned = plot_mode_shape(nodes, disp, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)

    def test_axis_labels(self) -> None:
        nodes, disp = self._data()
        fig = plot_mode_shape(nodes, disp)
        ax = fig.axes[0]
        assert "node" in ax.get_xlabel().lower() or "position" in ax.get_xlabel().lower()
        assert "displacement" in ax.get_ylabel().lower()
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. plot_harmonic_content
# ---------------------------------------------------------------------------

class TestPlotHarmonicContent:
    def test_returns_figure(self) -> None:
        Q = np.array([1.0, 0.1, 0.02, 0.005])
        fig = plot_harmonic_content(Q, omega=10.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bar_count_matches_harmonics(self) -> None:
        Q = np.array([1.0, 0.1, 0.02])
        fig = plot_harmonic_content(Q, omega=5.0)
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        plt.close(fig)

    def test_axis_labels(self) -> None:
        Q = np.array([1.0, 0.2])
        fig = plot_harmonic_content(Q, omega=3.14)
        ax = fig.axes[0]
        assert "Ω" in ax.get_xlabel() or "omega" in ax.get_xlabel().lower() or "harmonic" in ax.get_xlabel().lower()
        assert "amplitude" in ax.get_ylabel().lower() or "Amplitude" in ax.get_ylabel()
        plt.close(fig)

    def test_title_set(self) -> None:
        Q = np.array([1.0])
        fig = plot_harmonic_content(Q, omega=1.0)
        assert "Harmonic" in fig.axes[0].get_title() or "harmonic" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_ax_injection(self) -> None:
        Q = np.array([1.0, 0.1])
        ext_fig, ext_ax = plt.subplots()
        returned = plot_harmonic_content(Q, omega=1.0, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)

    def test_complex_input_uses_abs(self) -> None:
        Q = np.array([1.0 + 0.5j, 0.1 + 0.0j])
        fig = plot_harmonic_content(Q, omega=1.0)
        ax = fig.axes[0]
        bar_heights = [p.get_height() for p in ax.patches]
        assert bar_heights[0] == pytest.approx(abs(1.0 + 0.5j), rel=1e-6)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8. plot_convergence
# ---------------------------------------------------------------------------

class TestPlotConvergence:
    def test_returns_figure(self) -> None:
        res = np.array([1e0, 1e-2, 1e-5, 1e-9])
        fig = plot_convergence(res)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_semilogy_scale(self) -> None:
        res = np.array([1e0, 1e-3, 1e-6])
        fig = plot_convergence(res)
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_axis_labels(self) -> None:
        res = np.array([1.0, 0.1, 0.01])
        fig = plot_convergence(res)
        ax = fig.axes[0]
        assert "iter" in ax.get_xlabel().lower() or "step" in ax.get_xlabel().lower()
        assert "residual" in ax.get_ylabel().lower() or "norm" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_title_set(self) -> None:
        res = np.array([1.0, 0.01])
        fig = plot_convergence(res)
        assert "Convergence" in fig.axes[0].get_title() or "convergence" in fig.axes[0].get_title().lower()
        plt.close(fig)

    def test_ax_injection(self) -> None:
        res = np.array([1.0, 0.5, 0.1])
        ext_fig, ext_ax = plt.subplots()
        returned = plot_convergence(res, ax=ext_ax)
        assert returned is ext_fig
        plt.close(ext_fig)

    def test_single_point(self) -> None:
        fig = plot_convergence(np.array([1e-5]))
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Integration: inject same figure across multiple calls
# ---------------------------------------------------------------------------

class TestAxInjectionIntegration:
    def test_two_frf_results_on_one_axes(self) -> None:
        """Two FRF results can be overlaid on the same axes."""
        r1 = _make_frf_result()
        r2 = _make_frf_result(n=20)
        r2.stability = np.ones(20, dtype=bool)  # all stable

        fig, ax = plt.subplots()
        plot_frf(r1, ax=ax)
        plot_frf(r2, ax=ax)
        assert len(ax.lines) > 1
        plt.close(fig)
