"""
visualise.py
============
Plotting utilities for radar tracking results.

Requires: matplotlib (pip install matplotlib)

Plots available:
  plot_tracks()     : ground truth vs filtered trajectory
  plot_errors()     : position error over time
  plot_NIS()        : filter consistency (NIS) over time
  plot_uncertainty(): estimated position covariance ellipses
  plot_all()        : 2×2 dashboard of all plots
"""

from __future__ import annotations
import numpy as np
from typing import List

from .kalman import KalmanState
from .simulator import SimulationResult


def _check_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for visualisation. "
            "Install with: pip install matplotlib"
        )


def plot_tracks(
    states: List[KalmanState],
    sim: SimulationResult,
    title: str = "Radar Track",
    ax=None,
) -> None:
    """
    Plot true trajectory, noisy radar returns, and Kalman-filtered track.
    """
    plt = _check_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    # Convert measurements to Cartesian for plotting
    meas_x = sim.measurements[:, 0] * np.cos(sim.measurements[:, 1])
    meas_y = sim.measurements[:, 0] * np.sin(sim.measurements[:, 1])

    est_pos = np.array([s.position for s in states])

    ax.scatter(meas_x, meas_y, s=12, alpha=0.35, color="#6B9ED2",
               label="Radar returns (noisy)", zorder=2)
    ax.plot(sim.true_positions[:, 0], sim.true_positions[:, 1],
            "g-", linewidth=2.5, label="True track", zorder=3)
    ax.plot(est_pos[:, 0], est_pos[:, 1],
            "r--", linewidth=2, label="Kalman filter", zorder=4)

    # Mark start / end
    ax.plot(*sim.true_positions[0], "g^", markersize=10, zorder=5, label="Start")
    ax.plot(*sim.true_positions[-1], "rs", markersize=10, zorder=5, label="End")

    # Radar at origin
    ax.plot(0, 0, "k*", markersize=14, label="Radar", zorder=6)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if fig:
        plt.tight_layout()


def plot_errors(
    states: List[KalmanState],
    sim: SimulationResult,
    ax=None,
) -> None:
    """Plot position estimation error over time with 1-sigma uncertainty band."""
    plt = _check_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    n = min(len(states), len(sim.true_positions))
    est_pos = np.array([s.position for s in states[:n]])
    pos_errors = np.linalg.norm(est_pos - sim.true_positions[:n], axis=1)

    # 1-sigma uncertainty from covariance diagonal
    sigma_pos = np.array([
        np.sqrt(np.trace(s.P[:2, :2])) for s in states[:n]
    ])

    t = sim.timestamps[:n]
    ax.plot(t, pos_errors, color="#D04848", linewidth=1.5, label="Position error")
    ax.fill_between(t, 0, sigma_pos, alpha=0.2, color="#4A90D9",
                    label="1σ uncertainty")
    ax.axhline(np.mean(pos_errors), color="gray", linestyle="--",
               linewidth=1, label=f"Mean error: {np.mean(pos_errors):.0f} m")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position Estimation Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if fig:
        plt.tight_layout()


def plot_NIS(
    states: List[KalmanState],
    sim: SimulationResult,
    ax=None,
) -> None:
    """
    Plot Normalised Innovation Squared over time.
    NIS should stay within the 95% chi-squared bounds [0.05, 7.38]
    for a well-tuned filter.
    """
    plt = _check_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    nis_values = [s.NIS for s in states if s.NIS is not None]
    t = sim.timestamps[:len(nis_values)]

    ax.plot(t, nis_values, color="#7B5EA7", linewidth=1.5, label="NIS")
    ax.axhline(7.378, color="red", linestyle="--", linewidth=1,
               label="95% upper bound (χ²=7.38)")
    ax.axhline(0.051, color="orange", linestyle="--", linewidth=1,
               label="95% lower bound (χ²=0.05)")
    ax.axhline(2.0, color="green", linestyle=":", linewidth=1,
               label="Ideal mean (2.0)")

    pct = np.mean([(0.051 <= v <= 7.378) for v in nis_values]) * 100
    ax.set_title(f"Filter Consistency (NIS) — {pct:.0f}% within 95% bounds")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NIS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if fig:
        plt.tight_layout()


def plot_uncertainty(
    states: List[KalmanState],
    sim: SimulationResult,
    ax=None,
    every: int = 10,
) -> None:
    """
    Plot covariance ellipses (1-sigma position uncertainty) along the track.
    Shows how the filter's confidence evolves over time.
    """
    plt = _check_matplotlib()
    from matplotlib.patches import Ellipse
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    est_pos = np.array([s.position for s in states])
    ax.plot(sim.true_positions[:, 0], sim.true_positions[:, 1],
            "g-", linewidth=2, label="True track", alpha=0.5)
    ax.plot(est_pos[:, 0], est_pos[:, 1],
            "r--", linewidth=2, label="Filtered track")

    for i, state in enumerate(states):
        if i % every != 0:
            continue
        P_pos = state.P[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(P_pos)
        eigenvalues = np.maximum(eigenvalues, 0)
        width = 2 * np.sqrt(eigenvalues[0])
        height = 2 * np.sqrt(eigenvalues[1])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        ellipse = Ellipse(
            xy=state.position,
            width=width, height=height, angle=angle,
            edgecolor="#4A90D9", facecolor="#4A90D9",
            alpha=0.25, linewidth=1,
        )
        ax.add_patch(ellipse)

    ax.plot(0, 0, "k*", markersize=12, label="Radar")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Track with 1σ Position Uncertainty Ellipses")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if fig:
        plt.tight_layout()


def plot_all(
    states: List[KalmanState],
    sim: SimulationResult,
    scenario: str = "",
    save_path: str | None = None,
) -> None:
    """2×2 dashboard: track, errors, NIS, uncertainty ellipses."""
    plt = _check_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Radar Tracking Dashboard — {scenario}", fontsize=14, y=1.01)

    plot_tracks(states, sim, title="Trajectory", ax=axes[0, 0])
    plot_errors(states, sim, ax=axes[0, 1])
    plot_NIS(states, sim, ax=axes[1, 0])
    plot_uncertainty(states, sim, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
