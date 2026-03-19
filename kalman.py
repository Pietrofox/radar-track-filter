"""
kalman.py
=========
Linear Kalman Filter — pure NumPy, zero dependencies.

Tracks a moving object from noisy radar measurements.

State vector:  x = [x, y, vx, vy]  (position + velocity in 2D)
Measurement:   z = [r, theta]        (range + bearing from radar)

The filter alternates two steps each time step:
  1. Predict — extrapolate state forward using the motion model
  2. Update  — correct the prediction with the new radar measurement

References
----------
Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
Brown & Hwang (2012). "Introduction to Random Signals and Applied Kalman Filtering."
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanState:
    """
    Snapshot of the filter state at one time step.

    x      : state estimate vector [x, y, vx, vy]
    P      : state covariance matrix (4×4) — uncertainty
    K      : Kalman gain matrix used in this update step
    innovation : measurement residual z - H*x_pred
    NIS    : Normalised Innovation Squared — consistency metric
    """
    x: np.ndarray
    P: np.ndarray
    K: Optional[np.ndarray] = None
    innovation: Optional[np.ndarray] = None
    NIS: Optional[float] = None

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def heading_deg(self) -> float:
        """Heading in degrees, 0 = East, counter-clockwise."""
        vx, vy = self.velocity
        return float(np.degrees(np.arctan2(vy, vx)))


class KalmanFilter:
    """
    Constant-velocity Kalman Filter for 2D radar tracking.

    State: x = [px, py, vx, vy]
    Motion model: constant velocity with process noise Q.
    Measurement model: polar coordinates (range, bearing) linearised
    around current state estimate via Extended Kalman Filter (EKF)
    Jacobian.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    process_noise_std : float
        Standard deviation of acceleration noise (m/s²).
        Controls how quickly the filter adapts to manoeuvres.
    meas_range_std : float
        Radar range measurement noise (metres).
    meas_bearing_std : float
        Radar bearing measurement noise (radians).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 0.5,
        meas_range_std: float = 50.0,
        meas_bearing_std: float = 0.02,
    ):
        self.dt = dt
        self.n = 4  # state dimension

        # ── State transition matrix F (constant velocity) ──────────
        #   [1  0  dt  0 ]   px' = px + vx*dt
        #   [0  1  0   dt]   py' = py + vy*dt
        #   [0  0  1   0 ]   vx' = vx
        #   [0  0  0   1 ]   vy' = vy
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # ── Process noise covariance Q ──────────────────────────────
        # Discretised continuous white-noise acceleration model.
        q = process_noise_std ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = q * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ])

        # ── Measurement noise covariance R ──────────────────────────
        self.R = np.diag([meas_range_std**2, meas_bearing_std**2])

        # ── Identity matrix ─────────────────────────────────────────
        self.I = np.eye(self.n)

        # ── Filter state (initialised on first measurement) ─────────
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.initialized: bool = False

    # ────────────────────────────────────────────────────────────── #
    # Initialisation                                                  #
    # ────────────────────────────────────────────────────────────── #

    def initialise(
        self,
        x0: np.ndarray,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """
        Seed the filter with an initial state estimate.

        x0 : [px, py, vx, vy]
        P0 : initial covariance (large diagonal if unknown)
        """
        self.x = x0.copy().astype(float)
        if P0 is None:
            # Conservative initialisation: high uncertainty
            P0 = np.diag([500**2, 500**2, 30**2, 30**2])
        self.P = P0.copy().astype(float)
        self.initialized = True

    def initialise_from_polar(
        self,
        r: float,
        theta: float,
        v0: float = 0.0,
    ) -> None:
        """
        Initialise directly from a first radar measurement (r, theta).
        Velocity is assumed zero unless provided.
        """
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        self.initialise(np.array([px, py, v0, v0]))

    # ────────────────────────────────────────────────────────────── #
    # Predict step                                                    #
    # ────────────────────────────────────────────────────────────── #

    def predict(self) -> KalmanState:
        """
        Extrapolate state one time step forward.
        Returns the predicted state (before measurement update).
        """
        if not self.initialized:
            raise RuntimeError("Filter not initialised. Call initialise() first.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return KalmanState(x=self.x.copy(), P=self.P.copy())

    # ────────────────────────────────────────────────────────────── #
    # Update step (Extended Kalman — linearised polar measurement)   #
    # ────────────────────────────────────────────────────────────── #

    def update(self, z: np.ndarray) -> KalmanState:
        """
        Correct the prediction with a new measurement z = [r, theta].

        Uses the EKF linearisation: the measurement function h(x)
        maps Cartesian state to polar coordinates, and its Jacobian H
        is computed analytically at the current predicted state.

        Returns the updated KalmanState including Kalman gain,
        innovation, and NIS for consistency monitoring.
        """
        px, py = self.x[0], self.x[1]
        r_pred = np.sqrt(px**2 + py**2)

        # Avoid division by zero at the origin
        if r_pred < 1e-6:
            r_pred = 1e-6

        # ── Predicted measurement h(x) ──────────────────────────────
        h = np.array([r_pred, np.arctan2(py, px)])

        # ── Jacobian of h at current x ──────────────────────────────
        H = np.array([
            [ px/r_pred,  py/r_pred, 0, 0],
            [-py/r_pred**2, px/r_pred**2, 0, 0],
        ])

        # ── Innovation (measurement residual) ───────────────────────
        innovation = z - h
        # Normalise bearing innovation to [-π, π]
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

        # ── Innovation covariance S ──────────────────────────────────
        S = H @ self.P @ H.T + self.R

        # ── Kalman gain K ────────────────────────────────────────────
        K = self.P @ H.T @ np.linalg.inv(S)

        # ── State update ─────────────────────────────────────────────
        self.x = self.x + K @ innovation

        # ── Covariance update (Joseph form — numerically stable) ─────
        I_KH = self.I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # ── Normalised Innovation Squared (NIS) ──────────────────────
        # NIS ~ Chi²(2): should be < 5.99 (95th percentile) if tuned well
        NIS = float(innovation.T @ np.linalg.inv(S) @ innovation)

        return KalmanState(
            x=self.x.copy(),
            P=self.P.copy(),
            K=K,
            innovation=innovation,
            NIS=NIS,
        )

    # ────────────────────────────────────────────────────────────── #
    # Convenience: process a full measurement sequence               #
    # ────────────────────────────────────────────────────────────── #

    def process(self, measurements: np.ndarray) -> list[KalmanState]:
        """
        Run predict→update for each measurement in `measurements`.

        measurements : (N, 2) array of [r, theta] readings
        Returns a list of N KalmanState objects (post-update).
        """
        if not self.initialized:
            self.initialise_from_polar(measurements[0, 0], measurements[0, 1])

        states = []
        for z in measurements:
            self.predict()
            state = self.update(z)
            states.append(state)
        return states

    # ────────────────────────────────────────────────────────────── #
    # Properties                                                     #
    # ────────────────────────────────────────────────────────────── #

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty (metres), trace of position block."""
        return float(np.sqrt(np.trace(self.P[:2, :2])))

    @property
    def velocity_uncertainty(self) -> float:
        """1-sigma velocity uncertainty (m/s)."""
        return float(np.sqrt(np.trace(self.P[2:, 2:])))
