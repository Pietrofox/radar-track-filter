"""
metrics.py
==========
Evaluation metrics for tracking performance.

Key metrics used in radar tracking literature:

  RMSE       : Root Mean Square Error of position/velocity
  NIS        : Normalised Innovation Squared (filter consistency)
  ANEES      : Average Normalised Estimation Error Squared
  Track loss : fraction of time steps where error > threshold
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from .kalman import KalmanState
from .simulator import SimulationResult


@dataclass
class TrackingMetrics:
    """Aggregated tracking performance metrics."""

    # Position errors
    pos_rmse: float          # metres
    pos_mae: float           # metres
    pos_max_error: float     # metres

    # Velocity errors
    vel_rmse: float          # m/s
    vel_mae: float           # m/s

    # Filter consistency
    mean_NIS: float          # should be ~2.0 for 2-DOF measurements
    pct_NIS_in_bounds: float # % of NIS within 95% Chi²(2) interval [0.05, 7.38]

    # Track quality
    track_loss_pct: float    # % of steps with position error > threshold
    n_steps: int

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  Tracking Performance Metrics\n"
            f"{'='*50}\n"
            f"  Position RMSE     : {self.pos_rmse:.1f} m\n"
            f"  Position MAE      : {self.pos_mae:.1f} m\n"
            f"  Position max err  : {self.pos_max_error:.1f} m\n"
            f"  Velocity RMSE     : {self.vel_rmse:.2f} m/s\n"
            f"  Velocity MAE      : {self.vel_mae:.2f} m/s\n"
            f"  Mean NIS          : {self.mean_NIS:.2f}  (ideal ≈ 2.0)\n"
            f"  NIS in 95% bounds : {self.pct_NIS_in_bounds:.1f}%\n"
            f"  Track loss (<100m): {self.track_loss_pct:.1f}%\n"
            f"  Steps evaluated   : {self.n_steps}\n"
            f"{'='*50}"
        )


def compute_metrics(
    states: List[KalmanState],
    sim: SimulationResult,
    track_loss_threshold_m: float = 100.0,
) -> TrackingMetrics:
    """
    Compute standard tracking metrics given filter outputs and ground truth.

    Parameters
    ----------
    states      : list of KalmanState from KalmanFilter.process()
    sim         : SimulationResult from RadarSimulator
    track_loss_threshold_m : position error above which a step is a "loss"

    Returns
    -------
    TrackingMetrics
    """
    n = min(len(states), len(sim.true_positions))

    est_pos = np.array([s.position for s in states[:n]])
    est_vel = np.array([s.velocity for s in states[:n]])
    true_pos = sim.true_positions[:n]
    true_vel = sim.true_velocities[:n]

    # ── Position errors ─────────────────────────────────────────
    pos_errors = np.linalg.norm(est_pos - true_pos, axis=1)
    pos_rmse = float(np.sqrt(np.mean(pos_errors**2)))
    pos_mae = float(np.mean(pos_errors))
    pos_max = float(np.max(pos_errors))

    # ── Velocity errors ──────────────────────────────────────────
    vel_errors = np.linalg.norm(est_vel - true_vel, axis=1)
    vel_rmse = float(np.sqrt(np.mean(vel_errors**2)))
    vel_mae = float(np.mean(vel_errors))

    # ── NIS consistency ──────────────────────────────────────────
    nis_values = [s.NIS for s in states[:n] if s.NIS is not None]
    mean_NIS = float(np.mean(nis_values)) if nis_values else float("nan")

    # 95% chi-squared bounds for 2 degrees of freedom
    NIS_lo, NIS_hi = 0.0506, 7.378
    pct_in_bounds = float(
        np.mean([(NIS_lo <= v <= NIS_hi) for v in nis_values]) * 100
    ) if nis_values else float("nan")

    # ── Track loss ───────────────────────────────────────────────
    track_loss_pct = float(np.mean(pos_errors > track_loss_threshold_m) * 100)

    return TrackingMetrics(
        pos_rmse=pos_rmse,
        pos_mae=pos_mae,
        pos_max_error=pos_max,
        vel_rmse=vel_rmse,
        vel_mae=vel_mae,
        mean_NIS=mean_NIS,
        pct_NIS_in_bounds=pct_in_bounds,
        track_loss_pct=track_loss_pct,
        n_steps=n,
    )


def compare_configs(results: dict[str, TrackingMetrics]) -> None:
    """
    Print a side-by-side comparison table of multiple filter configurations.

    Usage:
        compare_configs({
            "Q_low":  metrics_q_low,
            "Q_high": metrics_q_high,
        })
    """
    configs = list(results.keys())
    print(f"\n{'Config':<18}", end="")
    print(f"{'Pos RMSE (m)':<16}{'Vel RMSE (m/s)':<18}{'Mean NIS':<12}{'NIS ok %'}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<18}"
            f"{m.pos_rmse:<16.1f}"
            f"{m.vel_rmse:<18.3f}"
            f"{m.mean_NIS:<12.2f}"
            f"{m.pct_NIS_in_bounds:.1f}%"
        )
