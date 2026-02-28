"""Tests for the electromagnetic rebar coiling simulation module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy.lib._em_rebar_coiling import (
    compute_magnetic_field,
    compute_lorentz_force,
    compute_joule_heating,
    simulate_rebar_coiling,
    generate_coil_animation_frames,
)


class TestComputeMagneticField:
    def test_output_shape(self):
        positions = np.zeros((5, 3))
        positions[:, 2] = np.linspace(0.01, 0.05, 5)
        B = compute_magnetic_field(100.0, positions, [0, 0, 0], 0.05)
        assert B.shape == (5, 3)

    def test_axial_field_sign(self):
        # On the axis above the coil centre the field should point in +z
        pos = np.array([[0.0, 0.0, 0.1]])
        B = compute_magnetic_field(1000.0, pos, [0, 0, 0], 0.05)
        assert B[0, 2] > 0, "Axial B-field above coil should be positive"

    def test_zero_current_gives_zero_field(self):
        positions = np.random.default_rng(0).random((10, 3))
        B = compute_magnetic_field(0.0, positions, [0, 0, 0], 0.1)
        assert_allclose(B, 0.0)

    def test_symmetry_about_coil_plane(self):
        # Points symmetric about z=0 should have opposite z-components
        pos_above = np.array([[0.0, 0.0, 0.05]])
        pos_below = np.array([[0.0, 0.0, -0.05]])
        B_above = compute_magnetic_field(500.0, pos_above, [0, 0, 0], 0.1)
        B_below = compute_magnetic_field(500.0, pos_below, [0, 0, 0], 0.1)
        assert_allclose(B_above[0, 2], B_below[0, 2], rtol=1e-10)


class TestComputeLorentzForce:
    def test_output_shape(self):
        N = 8
        J = np.ones((N, 3))
        B = np.ones((N, 3))
        vol = np.ones(N) * 1e-6
        F = compute_lorentz_force(J, B, vol)
        assert F.shape == (N, 3)

    def test_parallel_J_B_gives_zero_force(self):
        # J parallel to B → J × B = 0
        J = np.array([[1.0, 0.0, 0.0]])
        B = np.array([[2.0, 0.0, 0.0]])
        F = compute_lorentz_force(J, B, np.array([1e-6]))
        assert_allclose(F, 0.0, atol=1e-30)

    def test_perpendicular_J_B(self):
        # J = x̂, B = ŷ → J × B = ẑ
        J = np.array([[1.0, 0.0, 0.0]])
        B = np.array([[0.0, 1.0, 0.0]])
        vol = np.array([2.0])
        F = compute_lorentz_force(J, B, vol)
        assert_allclose(F[0], [0.0, 0.0, 2.0])


class TestComputeJouleHeating:
    def test_output_shape(self):
        N = 6
        J = np.ones((N, 3)) * 1e6
        heat = compute_joule_heating(J, 1.7e-7, np.ones(N) * 1e-6, 1e-4)
        assert heat.shape == (N,)

    def test_zero_current_gives_zero_heat(self):
        N = 4
        J = np.zeros((N, 3))
        heat = compute_joule_heating(J, 1.7e-7, np.ones(N) * 1e-6, 1e-3)
        assert_allclose(heat, 0.0)

    def test_positive_heat(self):
        J = np.array([[1e6, 0.0, 0.0]])
        heat = compute_joule_heating(J, 1.7e-7, np.array([1e-6]), 1e-3)
        assert heat[0] > 0, "Joule heating must be positive"


class TestSimulateRebarCoiling:
    def _default_params(self):
        return dict(
            rebar_length=1.0,
            rebar_radius=0.005,
            n_elements=10,
            coil_radius=0.15,
            current=5000.0,
            resistivity=1.7e-7,
            density=7850.0,
            youngs_modulus=2e11,
            n_steps=5,
            dt=1e-5,
        )

    def test_output_shapes(self):
        params = self._default_params()
        pos, temp = simulate_rebar_coiling(**params)
        n_elem = params["n_elements"]
        n_steps = params["n_steps"]
        assert pos.shape == (n_steps + 1, n_elem, 3)
        assert temp.shape == (n_steps + 1, n_elem)

    def test_initial_position_is_straight(self):
        params = self._default_params()
        pos, _ = simulate_rebar_coiling(**params)
        # At step 0 the rebar lies along the x-axis; y and z should be zero
        assert_allclose(pos[0, :, 1], 0.0, atol=1e-15)
        assert_allclose(pos[0, :, 2], 0.0, atol=1e-15)

    def test_temperatures_non_negative(self):
        params = self._default_params()
        _, temp = simulate_rebar_coiling(**params)
        assert np.all(temp >= 0), "Temperatures must be non-negative"

    def test_temperatures_increase_monotonically(self):
        params = self._default_params()
        _, temp = simulate_rebar_coiling(**params)
        # Each element's temperature should be non-decreasing over time
        diffs = np.diff(temp, axis=0)
        assert np.all(diffs >= -1e-30), (
            "Element temperatures should not decrease over time"
        )


class TestGenerateCoilAnimationFrames:
    def _make_sim_data(self, n_steps=10, n_elem=5):
        pos = np.random.default_rng(42).random((n_steps + 1, n_elem, 3))
        temp = np.random.default_rng(42).random((n_steps + 1, n_elem))
        return pos, temp

    def test_default_stride_gives_all_frames(self):
        n_steps = 10
        pos, temp = self._make_sim_data(n_steps=n_steps)
        frames = generate_coil_animation_frames(pos, temp)
        assert len(frames) == n_steps + 1

    def test_stride_reduces_frame_count(self):
        n_steps = 20
        pos, temp = self._make_sim_data(n_steps=n_steps)
        frames = generate_coil_animation_frames(pos, temp, frame_stride=5)
        # steps 0,5,10,15,20 → 5 frames
        assert len(frames) == 5

    def test_frame_keys(self):
        pos, temp = self._make_sim_data(n_steps=4)
        frames = generate_coil_animation_frames(pos, temp)
        for frame in frames:
            assert "positions" in frame
            assert "temperatures" in frame
            assert "step" in frame

    def test_frame_step_values(self):
        n_steps = 6
        pos, temp = self._make_sim_data(n_steps=n_steps)
        frames = generate_coil_animation_frames(pos, temp, frame_stride=2)
        steps = [f["step"] for f in frames]
        assert steps == [0, 2, 4, 6]

    def test_invalid_stride_raises(self):
        pos, temp = self._make_sim_data(n_steps=4)
        with pytest.raises(ValueError, match="frame_stride"):
            generate_coil_animation_frames(pos, temp, frame_stride=0)
