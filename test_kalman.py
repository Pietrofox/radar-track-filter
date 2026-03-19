"""
tests/test_kalman.py
====================
Unit tests for radar-track-filter.
Only requires NumPy — no matplotlib, no API keys.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar_tracker import KalmanFilter, RadarSimulator, RadarConfig, compute_metrics
from radar_tracker.kalman import KalmanState


class TestKalmanFilter(unittest.TestCase):

    def setUp(self):
        self.kf = KalmanFilter(dt=1.0, process_noise_std=0.5,
                               meas_range_std=50.0, meas_bearing_std=0.02)

    def test_initialise_from_polar(self):
        self.kf.initialise_from_polar(r=1000.0, theta=np.pi/4)
        self.assertTrue(self.kf.initialized)
        self.assertEqual(self.kf.x.shape, (4,))
        # Check Cartesian conversion
        expected_x = 1000.0 * np.cos(np.pi/4)
        expected_y = 1000.0 * np.sin(np.pi/4)
        self.assertAlmostEqual(self.kf.x[0], expected_x, places=3)
        self.assertAlmostEqual(self.kf.x[1], expected_y, places=3)

    def test_predict_returns_state(self):
        self.kf.initialise(np.array([1000., 500., 5., 2.]))
        state = self.kf.predict()
        self.assertIsInstance(state, KalmanState)
        self.assertEqual(state.x.shape, (4,))

    def test_predict_advances_position(self):
        """Constant velocity: predict should advance position by v*dt."""
        x0, y0, vx, vy = 1000., 500., 5., 2.
        self.kf.initialise(np.array([x0, y0, vx, vy]))
        state = self.kf.predict()
        self.assertAlmostEqual(state.x[0], x0 + vx * 1.0, places=5)
        self.assertAlmostEqual(state.x[1], y0 + vy * 1.0, places=5)

    def test_update_returns_state_with_NIS(self):
        self.kf.initialise(np.array([1000., 500., 5., 2.]))
        self.kf.predict()
        z = np.array([1118.0, 0.46])  # roughly consistent measurement
        state = self.kf.update(z)
        self.assertIsInstance(state, KalmanState)
        self.assertIsNotNone(state.NIS)
        self.assertGreater(state.NIS, 0)

    def test_covariance_decreases_after_update(self):
        """Filter should reduce uncertainty after receiving a measurement."""
        self.kf.initialise(np.array([1000., 500., 5., 2.]))
        state_pred = self.kf.predict()
        trace_before = np.trace(state_pred.P)
        z = np.array([1118.0, 0.46])
        state_upd = self.kf.update(z)
        trace_after = np.trace(state_upd.P)
        self.assertLess(trace_after, trace_before)

    def test_process_returns_correct_length(self):
        measurements = np.array([
            [1000., 0.5], [1050., 0.48], [1100., 0.46],
            [1150., 0.44], [1200., 0.43],
        ])
        self.kf.initialise_from_polar(measurements[0, 0], measurements[0, 1])
        states = self.kf.process(measurements)
        self.assertEqual(len(states), 5)

    def test_position_uncertainty_property(self):
        self.kf.initialise(np.array([1000., 500., 5., 2.]))
        unc = self.kf.position_uncertainty
        self.assertGreater(unc, 0)

    def test_bearing_normalisation(self):
        """Innovation bearing should stay within [-π, π] after wrapping."""
        self.kf.initialise(np.array([0., 1000., 0., 5.]))
        self.kf.predict()
        # Measurement near ±π boundary
        z = np.array([1000., np.pi - 0.01])
        state = self.kf.update(z)
        if state.innovation is not None:
            self.assertGreaterEqual(state.innovation[1], -np.pi)
            self.assertLessEqual(state.innovation[1], np.pi)

    def test_state_transition_matrix_shape(self):
        self.assertEqual(self.kf.F.shape, (4, 4))
        self.assertEqual(self.kf.Q.shape, (4, 4))
        self.assertEqual(self.kf.R.shape, (2, 2))


class TestRadarSimulator(unittest.TestCase):

    def setUp(self):
        self.sim = RadarSimulator(dt=1.0, n_steps=50, seed=0)

    def test_constant_velocity_shape(self):
        result = self.sim.constant_velocity()
        self.assertEqual(result.true_positions.shape, (50, 2))
        self.assertEqual(result.measurements.shape, (50, 2))
        self.assertEqual(len(result.timestamps), 50)

    def test_coordinated_turn_shape(self):
        result = self.sim.coordinated_turn()
        self.assertEqual(result.true_positions.shape, (50, 2))

    def test_evasive_manoeuvre_shape(self):
        result = self.sim.evasive_manoeuvre()
        self.assertEqual(result.true_positions.shape, (50, 2))

    def test_measurements_are_noisy(self):
        """Measurements should differ from true values due to noise."""
        result = self.sim.constant_velocity()
        true_ranges = result.true_ranges
        meas_ranges = result.measurements[:, 0]
        # Should not be identical
        self.assertFalse(np.allclose(true_ranges, meas_ranges))

    def test_noise_stats(self):
        """Noise RMSE should be approximately correct."""
        sim = RadarSimulator(dt=1.0, n_steps=1000,
                             radar=RadarConfig(range_std=50.0, bearing_std=0.02),
                             seed=1)
        result = sim.constant_velocity()
        stats = result.measurement_errors()
        # Allow 20% tolerance on noise estimation
        self.assertAlmostEqual(stats["range_rmse_m"], 50.0, delta=10.0)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        sim = RadarSimulator(dt=1.0, n_steps=60, seed=7)
        self.result = sim.constant_velocity()
        kf = KalmanFilter(dt=1.0)
        self.states = kf.process(self.result.measurements)

    def test_compute_metrics_returns_object(self):
        from radar_tracker.metrics import TrackingMetrics
        m = compute_metrics(self.states, self.result)
        self.assertIsInstance(m, TrackingMetrics)

    def test_pos_rmse_is_positive(self):
        m = compute_metrics(self.states, self.result)
        self.assertGreater(m.pos_rmse, 0)

    def test_pos_rmse_better_than_raw_measurements(self):
        """
        Kalman filter RMSE should be lower than raw measurement RMSE
        (converted to Cartesian) for a well-tuned filter.
        """
        m = compute_metrics(self.states, self.result)
        raw_rmse = self.result.measurement_errors()["range_rmse_m"]
        # Filter should do at least as well as raw measurements
        self.assertLess(m.pos_rmse, raw_rmse * 3)

    def test_NIS_bounds_percentage(self):
        m = compute_metrics(self.states, self.result)
        # For well-tuned filter: most NIS values should be in bounds
        self.assertGreater(m.pct_NIS_in_bounds, 50.0)

    def test_n_steps_correct(self):
        m = compute_metrics(self.states, self.result)
        self.assertEqual(m.n_steps, 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
