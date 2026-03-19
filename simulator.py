"""
simulator.py
============
Simulates a moving target (ship / vessel) and a ground-based radar.

The simulator generates:
  - Ground truth trajectory (Cartesian)
  - Noisy radar measurements (polar: range + bearing)

Supports several motion profiles:
  - constant_velocity   : straight-line cruise
  - coordinated_turn    : circular arc (constant turn rate)
  - evasive_manoeuvre   : two turns separated by a straight segment

All units:
  - distance : metres
  - speed    : metres / second
  - angles   : radians (internally), degrees (in display helpers)
  - time     : seconds
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class RadarConfig:
    """
    Parameters of the simulated radar sensor.

    range_std    : range measurement noise std dev (metres)
    bearing_std  : bearing measurement noise std dev (radians)
    max_range    : maximum detection range (metres), None = unlimited
    update_rate  : measurements per second (Hz)
    """
    range_std: float = 50.0
    bearing_std: float = 0.02       # ~1.15 degrees
    max_range: float = 30_000.0     # 30 km — typical maritime radar
    update_rate: float = 1.0        # 1 Hz default


@dataclass
class SimulationResult:
    """Output bundle returned by RadarSimulator.run()."""
    true_positions: np.ndarray      # (N, 2) Cartesian [x, y]
    true_velocities: np.ndarray     # (N, 2) [vx, vy]
    measurements: np.ndarray        # (N, 2) polar [r, theta]
    timestamps: np.ndarray          # (N,) seconds
    dt: float
    radar: RadarConfig

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1])

    @property
    def true_ranges(self) -> np.ndarray:
        """True range (metres) at each step."""
        return np.linalg.norm(self.true_positions, axis=1)

    @property
    def true_bearings(self) -> np.ndarray:
        """True bearing (radians) at each step."""
        return np.arctan2(self.true_positions[:, 1], self.true_positions[:, 0])

    def measurement_errors(self) -> dict:
        """Statistics on raw measurement noise."""
        range_err = self.measurements[:, 0] - self.true_ranges
        bearing_err = self.measurements[:, 1] - self.true_bearings
        return {
            "range_rmse_m":   float(np.sqrt(np.mean(range_err**2))),
            "bearing_rmse_rad": float(np.sqrt(np.mean(bearing_err**2))),
            "bearing_rmse_deg": float(np.degrees(np.sqrt(np.mean(bearing_err**2)))),
        }


class RadarSimulator:
    """
    Simulates a maritime radar tracking a manoeuvring vessel.

    The radar is located at the origin (0, 0).
    The target starts at (x0, y0) with velocity (vx0, vy0).

    Parameters
    ----------
    dt : float
        Simulation time step (seconds). Should match KalmanFilter dt.
    n_steps : int
        Total number of time steps to simulate.
    radar : RadarConfig, optional
        Radar sensor parameters. Defaults to typical maritime radar.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dt: float = 1.0,
        n_steps: int = 120,
        radar: RadarConfig | None = None,
        seed: int | None = 42,
    ):
        self.dt = dt
        self.n_steps = n_steps
        self.radar = radar or RadarConfig()
        self.rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────── #
    # Motion profiles                                              #
    # ──────────────────────────────────────────────────────────── #

    def constant_velocity(
        self,
        x0: float = 2000.0,
        y0: float = -5000.0,
        vx: float = 8.0,
        vy: float = 4.0,
    ) -> SimulationResult:
        """
        Straight-line constant-velocity trajectory.
        Typical for a vessel on a steady heading.
        """
        positions = np.zeros((self.n_steps, 2))
        velocities = np.full((self.n_steps, 2), [vx, vy])
        positions[0] = [x0, y0]
        for k in range(1, self.n_steps):
            positions[k] = positions[k-1] + velocities[k-1] * self.dt
        return self._add_noise(positions, velocities)

    def coordinated_turn(
        self,
        x0: float = 0.0,
        y0: float = -8000.0,
        speed: float = 10.0,
        heading0_deg: float = 80.0,
        turn_rate_deg_s: float = 1.5,
    ) -> SimulationResult:
        """
        Constant-rate circular arc (coordinated turn).
        Models a vessel changing course at fixed rate.

        turn_rate_deg_s : positive = left turn, negative = right turn
        """
        omega = np.radians(turn_rate_deg_s)
        heading = np.radians(heading0_deg)

        positions = np.zeros((self.n_steps, 2))
        velocities = np.zeros((self.n_steps, 2))
        positions[0] = [x0, y0]
        velocities[0] = [speed * np.cos(heading), speed * np.sin(heading)]

        for k in range(1, self.n_steps):
            heading += omega * self.dt
            velocities[k] = [speed * np.cos(heading), speed * np.sin(heading)]
            positions[k] = positions[k-1] + velocities[k-1] * self.dt

        return self._add_noise(positions, velocities)

    def evasive_manoeuvre(
        self,
        x0: float = -3000.0,
        y0: float = -6000.0,
        speed: float = 12.0,
    ) -> SimulationResult:
        """
        Straight → sharp turn → straight → turn back.
        Simulates an evasive or course-correction manoeuvre.
        """
        positions = np.zeros((self.n_steps, 2))
        velocities = np.zeros((self.n_steps, 2))

        # Phase boundaries
        t1 = self.n_steps // 3       # turn starts
        t2 = self.n_steps // 2       # turn ends
        t3 = 2 * self.n_steps // 3   # second turn starts

        heading = 0.0   # initially heading East
        turn_rate = np.radians(2.5)  # degrees per second

        positions[0] = [x0, y0]
        velocities[0] = [speed * np.cos(heading), speed * np.sin(heading)]

        for k in range(1, self.n_steps):
            if t1 <= k < t2:
                heading += turn_rate * self.dt          # left turn
            elif t3 <= k:
                heading -= turn_rate * self.dt * 0.8   # gentle right
            velocities[k] = [speed * np.cos(heading), speed * np.sin(heading)]
            positions[k] = positions[k-1] + velocities[k-1] * self.dt

        return self._add_noise(positions, velocities)

    # ──────────────────────────────────────────────────────────── #
    # Internal helpers                                             #
    # ──────────────────────────────────────────────────────────── #

    def _add_noise(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> SimulationResult:
        """Convert true positions to noisy polar measurements."""
        r_true = np.linalg.norm(positions, axis=1)
        theta_true = np.arctan2(positions[:, 1], positions[:, 0])

        r_noisy = r_true + self.rng.normal(0, self.radar.range_std, self.n_steps)
        theta_noisy = theta_true + self.rng.normal(0, self.radar.bearing_std, self.n_steps)

        # Clip range to radar's maximum range
        r_noisy = np.clip(r_noisy, 0, self.radar.max_range)

        measurements = np.column_stack([r_noisy, theta_noisy])
        timestamps = np.arange(self.n_steps) * self.dt

        return SimulationResult(
            true_positions=positions,
            true_velocities=velocities,
            measurements=measurements,
            timestamps=timestamps,
            dt=self.dt,
            radar=self.radar,
        )
