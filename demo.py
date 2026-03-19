"""
examples/demo.py
================
Full demonstration: simulate → filter → evaluate → visualise.

Run:
    python examples/demo.py                  # all three scenarios
    python examples/demo.py --no-plot        # metrics only, no matplotlib
    python examples/demo.py --scenario turn  # single scenario
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar_tracker import KalmanFilter, RadarSimulator, RadarConfig, compute_metrics
from radar_tracker.metrics import compare_configs


SCENARIOS = {
    "straight":  "constant_velocity",
    "turn":      "coordinated_turn",
    "evasive":   "evasive_manoeuvre",
}


def run_scenario(name: str, motion: str, plot: bool = True) -> None:
    print(f"\n{'─'*55}")
    print(f"  Scenario: {name.upper()}")
    print(f"{'─'*55}")

    # ── Simulate ──────────────────────────────────────────────────
    sim = RadarSimulator(dt=1.0, n_steps=120, seed=42)
    scenario_fn = getattr(sim, motion)
    result = scenario_fn()

    noise_stats = result.measurement_errors()
    print(f"  Radar noise — range: {noise_stats['range_rmse_m']:.0f} m, "
          f"bearing: {noise_stats['bearing_rmse_deg']:.2f}°")

    # ── Filter ────────────────────────────────────────────────────
    kf = KalmanFilter(
        dt=1.0,
        process_noise_std=0.5,
        meas_range_std=50.0,
        meas_bearing_std=0.02,
    )
    states = kf.process(result.measurements)

    # ── Evaluate ──────────────────────────────────────────────────
    metrics = compute_metrics(states, result)
    print(metrics)

    # ── Visualise ─────────────────────────────────────────────────
    if plot:
        try:
            from radar_tracker.visualise import plot_all
            plot_all(states, result, scenario=name, save_path=f"{name}_track.png")
        except ImportError:
            print("  (matplotlib not installed — skipping plots)")


def run_noise_sensitivity(plot: bool = True) -> None:
    """
    Show how filter performance degrades as radar noise increases.
    Useful for understanding filter tuning.
    """
    print(f"\n{'─'*55}")
    print("  Noise Sensitivity Analysis")
    print(f"{'─'*55}")

    configs = {
        "low noise (σ=20m)":    20.0,
        "nominal (σ=50m)":      50.0,
        "high noise (σ=150m)":  150.0,
    }

    all_metrics = {}
    for label, noise in configs.items():
        sim = RadarSimulator(dt=1.0, n_steps=120,
                             radar=RadarConfig(range_std=noise), seed=42)
        result = sim.evasive_manoeuvre()
        kf = KalmanFilter(dt=1.0, meas_range_std=noise)
        states = kf.process(result.measurements)
        all_metrics[label] = compute_metrics(states, result)

    compare_configs(all_metrics)


def main():
    plot = "--no-plot" not in sys.argv
    specific = None
    for arg in sys.argv[1:]:
        if arg.startswith("--scenario="):
            specific = arg.split("=")[1]

    if specific:
        if specific not in SCENARIOS:
            print(f"Unknown scenario '{specific}'. Choose: {list(SCENARIOS.keys())}")
            sys.exit(1)
        run_scenario(specific, SCENARIOS[specific], plot)
    else:
        for name, motion in SCENARIOS.items():
            run_scenario(name, motion, plot)
        run_noise_sensitivity(plot)


if __name__ == "__main__":
    main()
