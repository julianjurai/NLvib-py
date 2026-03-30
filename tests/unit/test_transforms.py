"""
Unit tests for nlvib.utils.transforms and nlvib.utils.linalg.

Acceptance criteria (from TASKS.md T-02):
- Round-trip FFT error < 1e-12
- AFT matches analytic Fourier coefficient for sin/cos inputs to < 1e-12
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlvib.utils.linalg import arc_length, dynamic_scaling
from nlvib.utils.transforms import aft_transform, freq_to_time, time_to_freq

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
N_TIME = 512          # number of time samples (power of 2)
H = 5                 # number of harmonics
ROUND_TRIP_TOL = 1e-12
AFT_TOL = 1e-12


def _make_signal(n_dof: int, n_harmonics: int, n_time: int) -> np.ndarray:
    """Generate a multi-harmonic signal with known coefficients."""
    t = np.linspace(0, 2 * np.pi, n_time, endpoint=False)
    q = np.zeros((n_dof, n_time))
    for dof in range(n_dof):
        for h in range(1, n_harmonics + 1):
            q[dof] += (dof + 1) * np.cos(h * t) + (dof + 2) * np.sin(h * t)
    return q


# ---------------------------------------------------------------------------
# time_to_freq  /  freq_to_time  — round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip: time → freq → time must recover the original to machine precision."""

    def test_single_dof_round_trip(self) -> None:
        """Round-trip through time_to_freq → freq_to_time → time_to_freq.

        The round-trip is exact (up to floating point) for signals that are
        exactly representable by H harmonics (i.e. band-limited to H).
        We construct such a signal explicitly.
        """
        t = np.linspace(0, 2 * np.pi, N_TIME, endpoint=False)
        # Build a signal that lives exactly in the space spanned by H harmonics
        rng = np.random.default_rng(42)
        a = rng.standard_normal(H + 1)  # DC + H cosine amplitudes
        b = rng.standard_normal(H)      # H sine amplitudes
        q_orig = a[0] * np.ones(N_TIME)
        for h in range(1, H + 1):
            q_orig = q_orig + a[h] * np.cos(h * t) + b[h - 1] * np.sin(h * t)

        Q = time_to_freq(q_orig, H)
        q_rec = freq_to_time(Q, N_TIME)
        Q2 = time_to_freq(q_rec, H)
        assert_allclose(Q2, Q, atol=ROUND_TRIP_TOL, rtol=0,
                        err_msg="Single-DOF round-trip failed")

    def test_multi_dof_round_trip(self) -> None:
        n_dof = 4
        q_orig = _make_signal(n_dof, H, N_TIME)
        Q = time_to_freq(q_orig, H)
        q_rec = freq_to_time(Q, N_TIME)
        Q2 = time_to_freq(q_rec, H)
        assert_allclose(Q2, Q, atol=ROUND_TRIP_TOL, rtol=0,
                        err_msg="Multi-DOF round-trip failed")

    def test_pure_cosine_coefficients(self) -> None:
        """cos(h·t) → Q has a_h = 1, all other coefficients 0."""
        for h_test in range(1, H + 1):
            t = np.linspace(0, 2 * np.pi, N_TIME, endpoint=False)
            q = np.cos(h_test * t)
            Q = time_to_freq(q, H)
            # a_h_test should be 1; b_h_test should be 0; a_0 should be 0
            expected = np.zeros(2 * H + 1)
            expected[2 * h_test - 1] = 1.0   # a_{h_test}
            assert_allclose(Q, expected, atol=ROUND_TRIP_TOL, rtol=0,
                            err_msg=f"Pure cosine h={h_test} failed")

    def test_pure_sine_coefficients(self) -> None:
        """sin(h·t) → Q has b_h = 1, all other coefficients 0."""
        for h_test in range(1, H + 1):
            t = np.linspace(0, 2 * np.pi, N_TIME, endpoint=False)
            q = np.sin(h_test * t)
            Q = time_to_freq(q, H)
            expected = np.zeros(2 * H + 1)
            expected[2 * h_test] = 1.0   # b_{h_test}
            assert_allclose(Q, expected, atol=ROUND_TRIP_TOL, rtol=0,
                            err_msg=f"Pure sine h={h_test} failed")

    def test_dc_only(self) -> None:
        """Constant signal → only a_0 is non-zero."""
        val = 3.7
        q = np.full(N_TIME, val)
        Q = time_to_freq(q, H)
        expected = np.zeros(2 * H + 1)
        expected[0] = val
        assert_allclose(Q, expected, atol=ROUND_TRIP_TOL, rtol=0)

    def test_output_shape_1d(self) -> None:
        q = np.ones(N_TIME)
        Q = time_to_freq(q, H)
        assert Q.shape == (2 * H + 1,)
        q_rec = freq_to_time(Q, N_TIME)
        assert q_rec.shape == (N_TIME,)

    def test_output_shape_2d(self) -> None:
        q = np.ones((3, N_TIME))
        Q = time_to_freq(q, H)
        assert Q.shape == (3, 2 * H + 1)
        q_rec = freq_to_time(Q, N_TIME)
        assert q_rec.shape == (3, N_TIME)


# ---------------------------------------------------------------------------
# time_to_freq — error handling
# ---------------------------------------------------------------------------


class TestTimeToFreqErrors:
    def test_n_time_too_small(self) -> None:
        with pytest.raises(ValueError, match="n_time"):
            time_to_freq(np.ones(3), n_harmonics=5)

    def test_freq_to_time_even_coeffs(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            freq_to_time(np.ones(4), n_time_samples=64)


# ---------------------------------------------------------------------------
# AFT transform — analytic verification
# ---------------------------------------------------------------------------


class TestAFTTransform:
    """AFT on analytic nonlinear functions with known Fourier coefficients."""

    def test_linear_force_single_harmonic(self) -> None:
        """f(q) = k·q for pure cosine input → output same harmonic, scaled by k."""
        k = 3.5
        h_test = 2
        # Build coefficient vector for q(t) = cos(h_test * t)
        Q = np.zeros(2 * H + 1)
        Q[2 * h_test - 1] = 1.0   # a_{h_test} = 1

        def linear_force(q: np.ndarray) -> np.ndarray:
            return k * q

        F = aft_transform(Q, linear_force, n_time=N_TIME)

        # Expected: F has a_{h_test} = k, all others 0
        expected = np.zeros(2 * H + 1)
        expected[2 * h_test - 1] = k
        assert_allclose(F, expected, atol=AFT_TOL, rtol=0,
                        err_msg="AFT linear force failed")

    def test_cubic_force_third_harmonic(self) -> None:
        """f(q) = q^3 for q(t) = A·cos(Ω·t) produces harmonics at 1 and 3.

        Analytic result (Krack & Gross 2019, Appendix C):
            q^3 = A^3·cos^3(t) = (3A^3/4)·cos(t) + (A^3/4)·cos(3t)

        So  a_1 = 3A^3/4,  a_3 = A^3/4,  all others 0.
        """
        A = 2.0
        Q = np.zeros(2 * H + 1)
        Q[1] = A   # a_1 = A → q = A cos(t)

        def cubic(q: np.ndarray) -> np.ndarray:
            return q**3

        F = aft_transform(Q, cubic, n_time=N_TIME)

        assert_allclose(F[1], 3 * A**3 / 4, atol=AFT_TOL, rtol=0,
                        err_msg="AFT cubic: a_1 mismatch")
        assert_allclose(F[5], A**3 / 4, atol=AFT_TOL, rtol=0,
                        err_msg="AFT cubic: a_3 mismatch")
        # All other coefficients must be zero
        mask = np.ones(2 * H + 1, dtype=bool)
        mask[1] = False
        mask[5] = False
        assert_allclose(F[mask], 0.0, atol=AFT_TOL, rtol=0,
                        err_msg="AFT cubic: spurious harmonics")

    def test_sine_input_linear_force(self) -> None:
        """f(q) = q for q = B·sin(t) → F has b_1 = B, others 0."""
        B = 1.7
        Q = np.zeros(2 * H + 1)
        Q[2] = B   # b_1 = B → q = B sin(t)

        def identity(q: np.ndarray) -> np.ndarray:
            return q

        F = aft_transform(Q, identity, n_time=N_TIME)
        expected = np.zeros(2 * H + 1)
        expected[2] = B
        assert_allclose(F, expected, atol=AFT_TOL, rtol=0,
                        err_msg="AFT identity-sine failed")

    def test_aft_multi_dof(self) -> None:
        """Multi-DOF: f_i(q) = k_i * q_i, verify each DOF independently."""
        n_dof = 3
        k_vals = np.array([1.0, 2.5, 0.5])
        Q = np.zeros((n_dof, 2 * H + 1))
        for i in range(n_dof):
            Q[i, 1] = 1.0   # a_1 = 1 for each DOF

        def scaled_linear(q: np.ndarray) -> np.ndarray:  # (n_dof, n_time)
            return k_vals[:, np.newaxis] * q

        F = aft_transform(Q, scaled_linear, n_time=N_TIME)
        for i in range(n_dof):
            expected = np.zeros(2 * H + 1)
            expected[1] = k_vals[i]
            assert_allclose(F[i], expected, atol=AFT_TOL, rtol=0,
                            err_msg=f"AFT multi-DOF failed at DOF {i}")

    def test_aft_error_small_n_time(self) -> None:
        Q = np.zeros(2 * H + 1)
        with pytest.raises(ValueError, match="n_time"):
            aft_transform(Q, lambda q: q, n_time=2)

    def test_aft_error_even_coeffs(self) -> None:
        Q = np.zeros(4)  # even → invalid
        with pytest.raises(ValueError, match="odd"):
            aft_transform(Q, lambda q: q, n_time=N_TIME)


# ---------------------------------------------------------------------------
# Canonical MATLAB reference values for transform functions
# ---------------------------------------------------------------------------


class TestCanonicalTransformReferenceValues:
    """Explicit canonical reference-point tests for freq_to_time and RMS formulae.

    All expected values are derived analytically (no Octave required).
    """

    def test_freq_to_time_at_t0_cosine_coeff_1(self) -> None:
        """freq_to_time: if Q has only a_1=1 (pure cos), then q(t=0) = 1.

        MATLAB ref: q(t) = a_1*cos(t), so at t=0 (k=0): q = a_1 = 1.0
        The NLvib convention is Q = [a_0, a_1, b_1, ...], so Q[1]=1 encodes
        a_1=1, giving q(t_0) = a_0 + a_1*cos(0) + b_1*sin(0) = 0 + 1 + 0 = 1.
        """
        # MATLAB ref: q(0) = 1.0
        H = 3
        Q = np.zeros(2 * H + 1)
        Q[1] = 1.0  # a_1 = 1  =>  q(t) = cos(t)
        q_time = freq_to_time(Q, n_time_samples=N_TIME)
        # k=0 corresponds to t=0: cos(0) = 1
        assert q_time[0] == pytest.approx(1.0, abs=1e-12)

    def test_freq_to_time_at_t0_sine_coeff_only(self) -> None:
        """freq_to_time: if Q has only b_1=1 (pure sin), then q(t=0) = 0.

        MATLAB ref: q(t) = b_1*sin(t), so at t=0: q = sin(0) = 0.
        """
        # MATLAB ref: q(0) = 0.0
        H = 3
        Q = np.zeros(2 * H + 1)
        Q[2] = 1.0  # b_1 = 1  =>  q(t) = sin(t)
        q_time = freq_to_time(Q, n_time_samples=N_TIME)
        assert q_time[0] == pytest.approx(0.0, abs=1e-12)

    def test_rms_of_pure_cosine_amplitude_1(self) -> None:
        """RMS of q(t)=cos(t) reconstructed from Q=[0,1,0,...] equals 1/sqrt(2).

        MATLAB ref: RMS of cos(t) over one period = sqrt(1/2pi * integral_0^{2pi} cos^2(t) dt)
                    = sqrt(1/2) = 1/sqrt(2) ≈ 0.7071067811865476
        The discrete RMS from freq_to_time must match this to 1e-10.
        """
        # MATLAB ref: rms = 1/sqrt(2) ≈ 0.7071
        H = 1
        Q = np.zeros(2 * H + 1)
        Q[1] = 1.0  # a_1 = 1  =>  q(t) = cos(t)
        q_time = freq_to_time(Q, n_time_samples=N_TIME)
        rms = float(np.sqrt(np.mean(q_time**2)))
        assert rms == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-10)

    def test_rms_of_pure_sine_amplitude_1(self) -> None:
        """RMS of q(t)=sin(t) reconstructed from Q=[0,0,1,...] equals 1/sqrt(2).

        MATLAB ref: RMS of sin(t) over one period = 1/sqrt(2) ≈ 0.7071
        """
        # MATLAB ref: rms = 1/sqrt(2) ≈ 0.7071
        H = 1
        Q = np.zeros(2 * H + 1)
        Q[2] = 1.0  # b_1 = 1  =>  q(t) = sin(t)
        q_time = freq_to_time(Q, n_time_samples=N_TIME)
        rms = float(np.sqrt(np.mean(q_time**2)))
        assert rms == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-10)

    def test_rms_formula_from_coefficients(self) -> None:
        """Verify RMS computed from coefficients matches discrete RMS from freq_to_time.

        For q(t) = a_0 + sum_h (a_h cos(h*t) + b_h sin(h*t)), the analytic RMS is:
            RMS^2 = a_0^2 + (1/2) * sum_h (a_h^2 + b_h^2)

        MATLAB ref: for a_1=1, all other coeffs=0:
            RMS^2 = 0 + 0.5*(1^2 + 0^2) = 0.5  =>  RMS = 1/sqrt(2)

        This test verifies the formula for a multi-harmonic signal.
        """
        # MATLAB ref: RMS^2 = a_0^2 + 0.5*sum(a_h^2 + b_h^2)
        H = 3
        # Build a signal with known non-zero coefficients
        Q = np.zeros(2 * H + 1)
        Q[0] = 0.5   # a_0 = 0.5  (DC)
        Q[1] = 1.0   # a_1 = 1.0
        Q[2] = 0.3   # b_1 = 0.3
        Q[5] = 0.7   # a_3 = 0.7
        Q[6] = 0.2   # b_3 = 0.2

        # Analytic RMS from coefficients:
        # a_0^2 + 0.5*(a_1^2 + b_1^2) + 0.5*(a_3^2 + b_3^2)
        rms_analytic = float(np.sqrt(
            Q[0] ** 2
            + 0.5 * (Q[1] ** 2 + Q[2] ** 2)
            + 0.5 * (Q[5] ** 2 + Q[6] ** 2)
        ))

        # Discrete RMS from reconstructed time signal
        q_time = freq_to_time(Q, n_time_samples=N_TIME)
        rms_discrete = float(np.sqrt(np.mean(q_time**2)))

        assert rms_discrete == pytest.approx(rms_analytic, abs=1e-10)


# ---------------------------------------------------------------------------
# dynamic_scaling
# ---------------------------------------------------------------------------


class TestDynamicScaling:
    def test_basic(self) -> None:
        x = np.array([2.0, 3.0, 0.0])
        x_ref = np.array([1.0, 2.0, 4.0])
        s = dynamic_scaling(x, x_ref)
        assert_allclose(s, np.array([2.0, 1.5, 0.0]), atol=1e-15)

    def test_zero_ref_uses_eps(self) -> None:
        """Zero reference component → scale by eps, large output."""
        x = np.array([1.0])
        x_ref = np.array([0.0])
        s = dynamic_scaling(x, x_ref)
        eps = np.finfo(float).eps
        assert_allclose(s, np.array([1.0 / eps]), rtol=1e-10)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            dynamic_scaling(np.ones(3), np.ones(4))

    def test_self_scaling_gives_ones(self) -> None:
        """Scaling a vector by itself gives all-ones (abs ≥ eps assumed)."""
        x = np.array([1.0, -2.0, 5.0])
        s = dynamic_scaling(x, x)
        assert_allclose(s, np.array([1.0, -1.0, 1.0]), atol=1e-15)


# ---------------------------------------------------------------------------
# arc_length
# ---------------------------------------------------------------------------


class TestArcLength:
    def test_zero(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert arc_length(x, x) == pytest.approx(0.0)

    def test_known(self) -> None:
        x_prev = np.array([0.0, 0.0])
        x_curr = np.array([3.0, 4.0])
        assert arc_length(x_prev, x_curr) == pytest.approx(5.0, abs=1e-14)

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(7)
        a, b = rng.standard_normal(10), rng.standard_normal(10)
        assert arc_length(a, b) == pytest.approx(arc_length(b, a), rel=1e-14)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            arc_length(np.ones(3), np.ones(4))

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(99)
        for _ in range(10):
            a, b = rng.standard_normal(8), rng.standard_normal(8)
            assert arc_length(a, b) >= 0.0
