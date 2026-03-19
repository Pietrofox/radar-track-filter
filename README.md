# radar-track-filter

> **Extended Kalman Filter for 2D maritime radar tracking — pure Python & NumPy.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies: NumPy](https://img.shields.io/badge/deps-numpy-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Tracks a moving vessel from noisy radar measurements using an **Extended Kalman Filter (EKF)**. Built for clarity and correctness — every matrix, every equation has a comment explaining *why*, not just *what*.

---

## Motivation

Radar tracking is a core problem in maritime surveillance, autonomous navigation, and defence systems. This repo implements the full EKF pipeline from scratch:

```
Radar (range, bearing) → EKF → Cartesian position + velocity estimate
```

The filter handles:
- **Non-linear measurement model** (polar → Cartesian conversion via EKF Jacobian)
- **Manoeuvring targets** (constant-velocity model + tunable process noise)
- **Filter health monitoring** via Normalised Innovation Squared (NIS)
- **Three motion scenarios**: straight cruise, coordinated turn, evasive manoeuvre

---

## Quick start

```bash
git clone https://github.com/your-username/radar-track-filter
cd radar-track-filter
pip install numpy matplotlib   # matplotlib only for plots
python examples/demo.py --no-plot   # metrics only, no display needed
```

**Minimal usage:**

```python
from radar_tracker import KalmanFilter, RadarSimulator, compute_metrics

# Simulate a vessel doing an evasive manoeuvre
sim = RadarSimulator(dt=1.0, n_steps=120, seed=42)
result = sim.evasive_manoeuvre()

# Run the filter
kf = KalmanFilter(dt=1.0, process_noise_std=0.5,
                  meas_range_std=50.0, meas_bearing_std=0.02)
states = kf.process(result.measurements)

# Evaluate
metrics = compute_metrics(states, result)
print(metrics)
# Position RMSE : 38.4 m
# Mean NIS      : 2.1  (ideal ≈ 2.0)
# NIS in bounds : 94.2%
```

---

## Architecture

```
radar_tracker/
├── kalman.py      EKF — predict / update / Joseph-form covariance
├── simulator.py   Radar + motion model (3 scenarios)
├── metrics.py     RMSE, NIS, ANEES, track-loss evaluation
└── visualise.py   4-panel plotting dashboard (matplotlib)
```

---

## The mathematics

### State vector

```
x = [px, py, vx, vy]   (Cartesian position + velocity)
```

### Prediction step

```
x_k|k-1 = F · x_k-1|k-1
P_k|k-1 = F · P_k-1|k-1 · Fᵀ + Q
```

where `F` is the constant-velocity transition matrix and `Q` is the discretised white-noise acceleration model.

### Update step (EKF)

The measurement is polar `z = [r, θ]`, non-linear in the state. The EKF linearises the measurement function `h(x)` around the current prediction:

```
H = ∂h/∂x |_{x_k|k-1}         (Jacobian, computed analytically)

K  = P_k|k-1 · Hᵀ · (H · P_k|k-1 · Hᵀ + R)⁻¹    (Kalman gain)
x  = x_k|k-1 + K · (z - h(x_k|k-1))               (state update)
P  = (I - K·H) · P_k|k-1 · (I - K·H)ᵀ + K·R·Kᵀ   (Joseph form)
```

The **Joseph form** covariance update is used instead of the simpler `P = (I-KH)P` for numerical stability.

### Filter consistency: NIS

```
NIS = νᵀ · S⁻¹ · ν     where ν = z - h(x_pred),  S = H·P·Hᵀ + R
```

NIS follows a χ²(2) distribution when the filter is consistent. The 95% acceptance region is `[0.051, 7.378]`. If NIS is consistently outside these bounds, the filter is mis-tuned.

---

## Scenarios

| Scenario | Description | Challenge |
|----------|-------------|-----------|
| `constant_velocity` | Straight-line cruise | Baseline |
| `coordinated_turn`  | Constant-rate circular arc | Model mismatch during turn |
| `evasive_manoeuvre` | Two sharp course changes | Process noise tuning |

---

## Filter tuning

| Parameter | Effect | Increase when... |
|-----------|--------|-----------------|
| `process_noise_std` | How quickly filter adapts to manoeuvres | Target makes sharp turns |
| `meas_range_std` | Trust in range measurements | Radar has high range noise |
| `meas_bearing_std` | Trust in bearing measurements | Radar has high angular noise |

Run the noise sensitivity analysis to see trade-offs:

```bash
python examples/demo.py  # includes noise sensitivity table
```

---

## Running tests

```bash
python -m pytest tests/ -v     # requires pytest
python tests/test_kalman.py    # standard library only
```

19 tests, all offline, no API keys required.

---

## References

- Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. ASME Journal of Basic Engineering.
- Brown & Hwang (2012). *Introduction to Random Signals and Applied Kalman Filtering*, 4th ed.
- Bar-Shalom, Li & Kirubarajan (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.

---

## License

MIT
