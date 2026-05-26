#!/usr/bin/env python3
"""
Rosbag plotter for vegardMpc runs.

Set BAG_PATH below and just run:  python rosbagreader.py
Or pass the path as a command-line argument to override:
    python rosbagreader.py <path/to/rosbag_folder>
    python rosbagreader.py <path/to/rosbag_folder> --save
"""


BAG_PATH = r"C:\Users\vegar\Documents\Isacsim\git\local\Master\Rosbag"
SAVE_PLOTS = True          # Set True to save PNGs automatically
SHOW_PLOTS = True           # Set False to suppress plot windows
OUTPUT_DIR = "rosbag_plots" # Folder for saved PNGs (relative or absolute)


from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rosbags.highlevel import AnyReader


# ── Topic lists ──────────────────────────────────────────────────────────────

FLOAT64_TOPICS = [
    "/lateral_mpc/cte_m",
    "/lateral_mpc/heading_error_deg",
    "/lateral_mpc/desired_delta_deg",
    "/lateral_mpc/actual_delta_deg",
    "/lateral_mpc/torque_cmd",
    "/lateral_mpc/progress_m",
    "/lateral_error",
    "/heading_error",
]

TWIST_TOPICS = [
    "/cmd_vel",
    "/path_follower/cmd_vel",
]

TWIST_STAMPED_TOPICS = [
    "/gnss/velocity",
]

VECTOR3_STAMPED_TOPICS = [
    "/gnss/accel",
    "/gnss/gyro",
]

POSE_STAMPED_TOPICS = [
    "/gnss/pose",
]

ALL_TOPICS = (
    FLOAT64_TOPICS
    + TWIST_TOPICS
    + TWIST_STAMPED_TOPICS
    + VECTOR3_STAMPED_TOPICS
    + POSE_STAMPED_TOPICS
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ns_to_sec(ts_ns, t0_ns):
    return (ts_ns - t0_ns) / 1e9


def empty_f64():
    return {"t": [], "y": []}


def empty_twist():
    return {"t": [], "lx": [], "ly": [], "lz": [], "ax": [], "ay": [], "az": []}


def empty_vec3():
    return {"t": [], "x": [], "y": [], "z": []}


def empty_pose():
    return {"t": [], "x": [], "y": [], "z": []}


def load_path_csv(bag_dir: Path):
    """Load reference path from path.csv next to the bag folder (or inside it)."""
    candidates = [
        bag_dir / "path.csv",
        bag_dir.parent / "path.csv",
    ]
    for p in candidates:
        if p.exists():
            xs, ys = [], []
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    xs.append(float(row["x"]))
                    ys.append(float(row["y"]))
            print(f"Loaded reference path: {p}  ({len(xs)} waypoints)")
            return xs, ys
    return None, None


# ── Read bag ──────────────────────────────────────────────────────────────────

def read_rosbag(bag_path: Path):
    f64   = {t: empty_f64()   for t in FLOAT64_TOPICS}
    twist = {t: empty_twist() for t in TWIST_TOPICS}
    twstp = {t: empty_twist() for t in TWIST_STAMPED_TOPICS}
    vec3  = {t: empty_vec3()  for t in VECTOR3_STAMPED_TOPICS}
    pose  = {t: empty_pose()  for t in POSE_STAMPED_TOPICS}

    with AnyReader([bag_path]) as reader:
        available = sorted({c.topic for c in reader.connections})
        print("\nAvailable topics in bag:")
        for t in available:
            print(f"  {t}")

        connections = [c for c in reader.connections if c.topic in ALL_TOPICS]
        if not connections:
            print("\nNo matching topics found.")
            return f64, twist, twstp, vec3, pose

        print("\nReading topics:")
        for c in connections:
            print(f"  {c.topic}  [{c.msgtype}]")

        t0 = None
        for conn, ts_ns, raw in reader.messages(connections=connections):
            if t0 is None:
                t0 = ts_ns
            t = ns_to_sec(ts_ns, t0)
            topic = conn.topic

            try:
                msg = reader.deserialize(raw, conn.msgtype)
            except Exception as e:
                print(f"  [skip] could not deserialize {topic}: {e}")
                continue

            if topic in FLOAT64_TOPICS:
                f64[topic]["t"].append(t)
                f64[topic]["y"].append(float(msg.data))

            elif topic in TWIST_TOPICS:
                d = twist[topic]
                d["t"].append(t)
                d["lx"].append(float(msg.linear.x))
                d["ly"].append(float(msg.linear.y))
                d["lz"].append(float(msg.linear.z))
                d["ax"].append(float(msg.angular.x))
                d["ay"].append(float(msg.angular.y))
                d["az"].append(float(msg.angular.z))

            elif topic in TWIST_STAMPED_TOPICS:
                d = twstp[topic]
                d["t"].append(t)
                d["lx"].append(float(msg.twist.linear.x))
                d["ly"].append(float(msg.twist.linear.y))
                d["lz"].append(float(msg.twist.linear.z))
                d["ax"].append(float(msg.twist.angular.x))
                d["ay"].append(float(msg.twist.angular.y))
                d["az"].append(float(msg.twist.angular.z))

            elif topic in VECTOR3_STAMPED_TOPICS:
                d = vec3[topic]
                d["t"].append(t)
                d["x"].append(float(msg.vector.x))
                d["y"].append(float(msg.vector.y))
                d["z"].append(float(msg.vector.z))

            elif topic in POSE_STAMPED_TOPICS:
                d = pose[topic]
                d["t"].append(t)
                d["x"].append(float(msg.pose.position.x))
                d["y"].append(float(msg.pose.position.y))
                d["z"].append(float(msg.pose.position.z))

    return f64, twist, twstp, vec3, pose


# ── Plotting ──────────────────────────────────────────────────────────────────

def _f(topic, f64):
    """Return (t, y) for a Float64 topic, or ([], []) if empty."""
    d = f64.get(topic, {})
    return d.get("t", []), d.get("y", [])


def plot_mpc_errors(f64):
    """Figure 1 – cross-track error and heading error over time."""
    t_cte, y_cte = _f("/lateral_mpc/cte_m", f64)
    t_lat, y_lat = _f("/lateral_error", f64)
    t_hed, y_hed = _f("/lateral_mpc/heading_error_deg", f64)
    t_hraw, y_hraw = _f("/heading_error", f64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("MPC Tracking Errors", fontsize=14)

    ax = axes[0]
    if t_cte:
        ax.plot(t_cte, y_cte, label="/lateral_mpc/cte_m", color="tab:blue")
    if t_lat:
        ax.plot(t_lat, y_lat, label="/lateral_error", color="tab:orange", linestyle="--")
    ax.set_ylabel("Cross-track error [m]")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if t_hed:
        ax.plot(t_hed, y_hed, label="/lateral_mpc/heading_error_deg", color="tab:green")
    if t_hraw:
        ax.plot(t_hraw, y_hraw, label="/heading_error", color="tab:red", linestyle="--")
    ax.set_ylabel("Heading error [deg]")
    ax.set_xlabel("Time [s]")
    ax.legend(); ax.grid(True)

    fig.tight_layout()


def plot_steering(f64):
    """Figure 2 – desired vs actual steering angle and torque command."""
    t_des, y_des = _f("/lateral_mpc/desired_delta_deg", f64)
    t_act, y_act = _f("/lateral_mpc/actual_delta_deg", f64)
    t_tor, y_tor = _f("/lateral_mpc/torque_cmd", f64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Steering Control", fontsize=14)

    ax = axes[0]
    if t_des:
        ax.plot(t_des, y_des, label="desired δ", color="tab:blue")
    if t_act:
        ax.plot(t_act, y_act, label="actual δ", color="tab:orange", linestyle="--")
    ax.set_ylabel("Steering angle [deg]")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if t_tor:
        ax.plot(t_tor, y_tor, color="tab:purple")
    ax.set_ylabel("Torque cmd")
    ax.set_xlabel("Time [s]")
    ax.grid(True)

    fig.tight_layout()


def plot_speed(f64, twist, twstp):
    """Figure 3 – commanded and actual forward speed."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Speed", fontsize=14)

    d = twstp.get("/gnss/velocity", {})
    if d.get("t"):
        ax.plot(d["t"], d["lx"], label="GNSS speed (linear.x)", color="tab:blue")

    d = twist.get("/cmd_vel", {})
    if d.get("t"):
        ax.plot(d["t"], d["lx"], label="/cmd_vel linear.x", color="tab:orange", linestyle="--")

    d = twist.get("/path_follower/cmd_vel", {})
    if d.get("t"):
        ax.plot(d["t"], d["lx"], label="/path_follower/cmd_vel linear.x",
                color="tab:green", linestyle=":")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.legend(); ax.grid(True)
    fig.tight_layout()


def plot_path(pose, ref_xs, ref_ys):
    """Figure 4 – driven path vs reference path (UTM x/y)."""
    d = pose.get("/gnss/pose", {})
    if not d.get("x") and (ref_xs is None):
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Path: driven vs reference", fontsize=14)

    if ref_xs:
        ax.plot(ref_xs, ref_ys, "k--", linewidth=1.5, label="Reference path", zorder=1)

    if d.get("x"):
        ax.plot(d["x"], d["y"], color="tab:blue", linewidth=2,
                label="Driven path (/gnss/pose)", zorder=2)
        ax.plot(d["x"][0],  d["y"][0],  "go", markersize=8, label="Start", zorder=3)
        ax.plot(d["x"][-1], d["y"][-1], "rs", markersize=8, label="End",   zorder=3)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.legend(); ax.grid(True)
    fig.tight_layout()


def plot_imu(vec3):
    """Figure 5 – accelerometer and gyro from GNSS/IMU."""
    d_acc = vec3.get("/gnss/accel", {})
    d_gyr = vec3.get("/gnss/gyro", {})

    if not d_acc.get("t") and not d_gyr.get("t"):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("IMU (GNSS accel & gyro)", fontsize=14)

    ax = axes[0]
    if d_acc.get("t"):
        ax.plot(d_acc["t"], d_acc["x"], label="x")
        ax.plot(d_acc["t"], d_acc["y"], label="y")
        ax.plot(d_acc["t"], d_acc["z"], label="z")
    ax.set_ylabel("Accel [m/s²]")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if d_gyr.get("t"):
        ax.plot(d_gyr["t"], d_gyr["x"], label="x")
        ax.plot(d_gyr["t"], d_gyr["y"], label="y")
        ax.plot(d_gyr["t"], d_gyr["z"], label="z")
    ax.set_ylabel("Gyro [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.legend(); ax.grid(True)

    fig.tight_layout()


def plot_progress(f64):
    """Figure 6 – path progress over time."""
    t, y = _f("/lateral_mpc/progress_m", f64)
    if not t:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Path Progress", fontsize=14)
    ax.plot(t, y, color="teal")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Progress [m]")
    ax.grid(True)
    fig.tight_layout()


def save_all_figures(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        title = fig.texts[0].get_text() if fig.texts else f"figure_{i}"
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
        path = output_dir / f"{i:02d}_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot data from a ROS 2 rosbag (.mcap)."
    )
    parser.add_argument("bag_path", nargs="?", default=None,
                        help="Path to rosbag folder (overrides BAG_PATH in script)")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG files")
    parser.add_argument("--output-dir", default=None, help="Output folder for PNGs")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    args = parser.parse_args()

    # CLI argument overrides the hardcoded path; otherwise use BAG_PATH above
    bag_path   = Path(args.bag_path) if args.bag_path else Path(BAG_PATH)
    save       = args.save or SAVE_PLOTS
    show       = not args.no_show and SHOW_PLOTS
    output_dir = Path(args.output_dir) if args.output_dir else Path(OUTPUT_DIR)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    if not (bag_path / "metadata.yaml").exists():
        raise FileNotFoundError(f"No metadata.yaml in: {bag_path}")

    ref_xs, ref_ys = load_path_csv(bag_path)
    f64, twist, twstp, vec3, pose = read_rosbag(bag_path)

    plot_mpc_errors(f64)
    plot_steering(f64)
    plot_speed(f64, twist, twstp)
    plot_path(pose, ref_xs, ref_ys)
    plot_imu(vec3)
    plot_progress(f64)

    if save:
        save_all_figures(output_dir)

    if show:
        plt.show()


if __name__ == "__main__":
    main()
