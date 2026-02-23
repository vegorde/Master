import numpy as np


def resample_waypoints_linear_xyv(waypoints, ds):
    """
    Linear interpolation of (x,y,v) waypoints with fixed spatial spacing ds.

    - Geometry: straight segments
    - Velocity: linearly interpolated along segment
    - Output spacing ~ ds

    Returns:
        xref, yref, vref
    """
    wp = np.asarray(waypoints, dtype=float)

    assert wp.ndim == 2 and wp.shape[1] == 3, "Waypoints must be (x,y,v)"
    assert len(wp) >= 2, "Need at least 2 waypoints"

    xw, yw, vw = wp[:, 0], wp[:, 1], wp[:, 2]

    x_out = [xw[0]]
    y_out = [yw[0]]
    v_out = [vw[0]]

    for i in range(len(wp) - 1):
        x0, y0, v0 = xw[i], yw[i], vw[i]
        x1, y1, v1 = xw[i + 1], yw[i + 1], vw[i + 1]

        dx = x1 - x0
        dy = y1 - y0
        seg_len = float(np.hypot(dx, dy))

        if seg_len < 1e-9:
            continue

        # number of samples along segment
        n = int(np.floor(seg_len / ds))

        for k in range(1, n + 1):
            s = k * ds
            a = s / seg_len

            x_out.append(x0 + a * dx)
            y_out.append(y0 + a * dy)
            v_out.append(v0 + a * (v1 - v0))

        # ensure segment endpoint included exactly
        if (x_out[-1] != x1) or (y_out[-1] != y1):
            x_out.append(x1)
            y_out.append(y1)
            v_out.append(v1)

    return np.array(x_out), np.array(y_out), np.array(v_out)


def export_waypoints_py(xref, yref, vref, filename="waypoints_linear.py"):
    """Save to Python file in your exact format."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("waypoints = [\n")
        for x, y, v in zip(xref, yref, vref):
            f.write(f"    ({x:.3f}, {y:.3f}, {v:.3f}),\n")
        f.write("]\n")
    print(f"[saved] {filename}")


if __name__ == "__main__":

    # -------- YOUR WAYPOINTS --------
    waypoints = [
        (0.0,  0.0, 1.0),
        (6.0,  0.0, 1.0),
        (10.0, 0.0, 0.1),
        (10.0, 4.0, 1.0),
        (10.0, 10.0, 1.0),
    ]

    ds = 2   # desired spacing in meters

    xref, yref, vref = resample_waypoints_linear_xyv(waypoints, ds)

    print("\nwaypoints = [")
    for x, y, v in zip(xref, yref, vref):
        print(f"    ({x:.3f}, {y:.3f}, {v:.3f}),")
    print("]\n")

    # Optional: save to file
   # export_waypoints_py(xref, yref, vref)