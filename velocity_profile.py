"""
velocity_profile.py
====================
Generate a speed profile along a CSV waypoint path.

Strategy
--------
1. Estimate curvature at every waypoint using a three-point circle fit.
2. Map curvature → target speed:
       v = V_MAX / (1 + k_gain * kappa)
   where k_gain is tuned so the sharpest turn hits V_MIN.
3. Apply a forward/backward smoothing pass so the car can actually
   brake in time before curves and accelerate out of them.

Units : km/h throughout the profile (m/s internally).
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── Tuneable parameters ────────────────────────────────────────────────────────

PATH_CSV   = Path(r"C:\Users\vegar\Documents\Isacsim\git\local\Master\Rosbag\path.csv")
V_MAX_KMH  = 15.0   # Maximum straight-line speed  [km/h]  ← change me
V_MIN_KMH  = 10.0   # Speed at the tightest curve  [km/h]

# Deceleration / acceleration limits used for the smoothing pass [m/s²]
A_MAX_BRAKE = 1   # Max comfortable braking
A_MAX_ACCEL = 1   # Max comfortable acceleration

# Smoothing — Gaussian sigma in metres.
# Larger = smoother profile but corners are anticipated earlier.
SMOOTH_SIGMA_M = 10.0  # ← tune this (try 10–40 m)

# Exponent for the curvature→speed mapping.
# Formula:  v = v_min + (v_max - v_min) * (1 - kappa_norm^(1/n))
# n=1 : linear drop.
# n=2 : stays near V_MAX on gentle curves, drops sharply into tight corners.
# n=3 : even flatter on straights, almost cliff-edge into corners.
SPEED_EXPONENT = 2.0   # ← try 1.5 – 4.0

# Percentile used as kappa_max for normalisation.
# 100 = absolute max (one spike compresses everything).
# 95–99 = robust: genuine straights map to ~v_max, tight corners still clip to v_min.
KAPPA_PERCENTILE = 95.0  # ← lower = more sections treated as "straight"

# Curvature floor: baseline road bend that counts as "straight".
# Any kappa below this → full speed.  Set as a low percentile of the path curvature
# so the gentlest sections always reach V_MAX regardless of GPS noise.
KAPPA_FLOOR_PERCENTILE = 15.0  # ← raise to widen the "full speed" zone

OUTPUT_CSV  = Path("velocity_profile.csv")   # Set None to skip saving

# ── Helpers ────────────────────────────────────────────────────────────────────

KMH_TO_MS = 1 / 3.6
MS_TO_KMH = 3.6


def load_path(csv_path: Path):
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs), np.array(ys)


def arc_length(xs, ys):
    """Cumulative arc-length along the polyline."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.hypot(dx, dy)
    return np.concatenate([[0.0], np.cumsum(ds)])


def curvature(xs, ys):
    """
    Three-point curvature estimate at every interior point.
    Endpoints are assigned the curvature of their nearest interior neighbour.
    """
    n = len(xs)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        ax, ay = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        bx, by = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        cross   = ax * by - ay * bx
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        lc = math.hypot(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1])
        denom = la * lb * lc
        kappa[i] = 2 * abs(cross) / denom if denom > 1e-12 else 0.0
    kappa[0]  = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def curvature_to_speed(kappa, v_max_ms, v_min_ms,
                       exponent=2.0,
                       kappa_percentile=99.0,
                       kappa_floor_percentile=15.0):
    """
    Map curvature → speed using a power-law on the *excess* curvature:

        kappa_floor = percentile(kappa, kappa_floor_percentile)
        kappa_ref   = percentile(kappa, kappa_percentile)
        kappa_norm  = clip((kappa - kappa_floor) / (kappa_ref - kappa_floor), 0, 1)
        v = v_min + (v_max - v_min) * (1 - kappa_norm^(1/n))

    kappa_floor: the baseline road bend treated as "straight" → full speed.
                 Sections at or below this always get v_max.
    kappa_ref:   the reference tight corner → v_min.  Taken as a high percentile
                 so one noisy spike doesn't compress the whole range.
    n=1 : linear.
    n=2 : flat near v_max, sharp drop into tight corners.
    n>2 : even more cliff-like.
    """
    kappa_floor = np.percentile(kappa, kappa_floor_percentile)
    kappa_ref   = np.percentile(kappa, kappa_percentile)
    spread = kappa_ref - kappa_floor
    if spread < 1e-9:
        return np.full_like(kappa, v_max_ms)

    kappa_excess = np.clip(kappa - kappa_floor, 0.0, None)
    kappa_norm   = np.clip(kappa_excess / spread, 0.0, 1.0)
    v = v_min_ms + (v_max_ms - v_min_ms) * (1.0 - kappa_norm ** (1.0 / exponent))
    return np.clip(v, v_min_ms, v_max_ms)


def smooth_backward(v, s, a_brake):
    """
    Backward pass: make sure the car can brake in time.
    v[i] ≤ sqrt(v[i+1]² + 2*a_brake*ds)
    """
    v = v.copy()
    for i in range(len(v) - 2, -1, -1):
        ds = s[i + 1] - s[i]
        v_limit = math.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds)
        if v[i] > v_limit:
            v[i] = v_limit
    return v


def gaussian_smooth_arclength(values, s, sigma_m):
    """
    Gaussian smooth in arc-length space.
    Converts sigma from metres to an equivalent number of waypoint samples
    using the mean waypoint spacing, then applies gaussian_filter1d.
    """
    mean_ds = (s[-1] - s[0]) / (len(s) - 1)
    sigma_samples = max(sigma_m / mean_ds, 0.5)
    return gaussian_filter1d(values, sigma=sigma_samples)


def smooth_forward(v, s, a_accel):
    """
    Forward pass: make sure the car can accelerate in time.
    v[i+1] ≤ sqrt(v[i]² + 2*a_accel*ds)
    """
    v = v.copy()
    for i in range(len(v) - 1):
        ds = s[i + 1] - s[i]
        v_limit = math.sqrt(v[i] ** 2 + 2 * a_accel * ds)
        if v[i + 1] > v_limit:
            v[i + 1] = v_limit
    return v


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_velocity_profile(
    path_csv:               Path  = PATH_CSV,
    v_max_kmh:              float = V_MAX_KMH,
    v_min_kmh:              float = V_MIN_KMH,
    a_brake:                float = A_MAX_BRAKE,
    a_accel:                float = A_MAX_ACCEL,
    smooth_sigma_m:         float = SMOOTH_SIGMA_M,
    exponent:               float = SPEED_EXPONENT,
    kappa_percentile:       float = KAPPA_PERCENTILE,
    kappa_floor_percentile: float = KAPPA_FLOOR_PERCENTILE,
    output_csv:             Path  = OUTPUT_CSV,
    plot:                   bool  = True,
):
    v_max_ms = v_max_kmh * KMH_TO_MS
    v_min_ms = v_min_kmh * KMH_TO_MS

    # 1. Load path
    xs, ys = load_path(path_csv)
    s      = arc_length(xs, ys)
    print(f"Path loaded: {len(xs)} waypoints, {s[-1]:.1f} m total")

    # 2. Curvature — smooth first to remove point-to-point noise
    kappa_raw    = curvature(xs, ys)
    kappa        = gaussian_smooth_arclength(kappa_raw, s, smooth_sigma_m)
    kappa        = np.clip(kappa, 0.0, None)   # smoothing can create tiny negatives
    print(f"Curvature   max={kappa.max():.4f}  mean={kappa.mean():.4f}  "
          f"(sigma={smooth_sigma_m} m)")

    # 3. Curvature → speed
    v_raw = curvature_to_speed(kappa, v_max_ms, v_min_ms,
                               exponent=exponent,
                               kappa_percentile=kappa_percentile,
                               kappa_floor_percentile=kappa_floor_percentile)

    # 4. Kinematic smoothing (brake / accel feasibility)
    v_smooth = smooth_backward(v_raw,    s, a_brake)
    v_smooth = smooth_forward (v_smooth, s, a_accel)

    # 5. Final Gaussian pass to remove any remaining kinks
    v_smooth = gaussian_smooth_arclength(v_smooth, s, smooth_sigma_m * 0.5)
    v_smooth = np.clip(v_smooth, v_min_ms, v_max_ms)

    v_kmh = v_smooth * MS_TO_KMH

    print(f"Speed profile  max={v_kmh.max():.2f}  "
          f"min={v_kmh.min():.2f}  "
          f"mean={v_kmh.mean():.2f} km/h")

    # 6. Save CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "s_m", "kappa", "v_kmh", "v_ms"])
            for i in range(len(xs)):
                writer.writerow([
                    f"{xs[i]:.4f}", f"{ys[i]:.4f}",
                    f"{s[i]:.3f}",  f"{kappa[i]:.6f}",
                    f"{v_kmh[i]:.4f}", f"{v_smooth[i]:.4f}",
                ])
        print(f"Saved: {output_csv}")

    # 7. Plot
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        fig.suptitle(
            f"Velocity Profile  |  V_max={v_max_kmh} km/h  V_min={v_min_kmh} km/h",
            fontsize=13,
        )

        # — Speed vs distance —
        ax = axes[0]
        ax.plot(s, v_kmh, color="tab:blue", linewidth=1.8, label="Speed")
        ax.axhline(v_max_kmh, color="tab:green",  linestyle="--", linewidth=1, label=f"V_max {v_max_kmh} km/h")
        ax.axhline(v_min_kmh, color="tab:orange", linestyle="--", linewidth=1, label=f"V_min {v_min_kmh} km/h")
        ax.set_ylabel("Speed [km/h]")
        ax.set_xlabel("Distance along path [m]")
        ax.legend(); ax.grid(True)

        # — Curvature vs distance —
        ax = axes[1]
        ax.plot(s, kappa_raw, color="tab:red",  linewidth=0.8, alpha=0.4, label="Raw")
        ax.plot(s, kappa,     color="tab:red",  linewidth=1.6,             label=f"Smoothed ({smooth_sigma_m} m)")
        ax.set_ylabel("Curvature [1/m]")
        ax.set_xlabel("Distance along path [m]")
        ax.legend(); ax.grid(True)

        # — Path coloured by speed —
        ax = axes[2]
        sc = ax.scatter(xs, ys, c=v_kmh, cmap="RdYlGn",
                        s=8, vmin=v_min_kmh, vmax=v_max_kmh, zorder=2)
        plt.colorbar(sc, ax=ax, label="Speed [km/h]")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.axis("equal"); ax.grid(True)
        ax.set_title("Path coloured by speed")

        fig.tight_layout()
        plt.show()

    return xs, ys, s, kappa, v_smooth


if __name__ == "__main__":
    generate_velocity_profile()
