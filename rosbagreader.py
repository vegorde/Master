#!/usr/bin/env python3

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader


FLOAT64_TOPICS = [
    "/lateral_mpc/cte_m",
    "/lateral_mpc/heading_error_deg",
    "/lateral_mpc/desired_delta_deg",
    "/lateral_mpc/actual_delta_deg",
]

TWIST_TOPICS = [
    "/cmd_vel",
]

TWIST_STAMPED_TOPICS = [
    "/gnss/velocity",
]

GPS_TOPICS = [
    "/gnss/fix",
]


def ns_to_sec(timestamp_ns, t0_ns):
    return (timestamp_ns - t0_ns) / 1e9


def get_nested_attr(obj, path):
    """
    Example:
        get_nested_attr(msg, "twist.linear.x")
    """
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    return value


def read_rosbag(bag_path: Path):
    float_data = {
        topic: {"t": [], "y": []}
        for topic in FLOAT64_TOPICS
    }

    twist_data = {
        topic: {
            "t": [],
            "linear_x": [],
            "linear_y": [],
            "linear_z": [],
            "angular_x": [],
            "angular_y": [],
            "angular_z": [],
        }
        for topic in TWIST_TOPICS
    }

    twist_stamped_data = {
        topic: {
            "t": [],
            "linear_x": [],
            "linear_y": [],
            "linear_z": [],
            "angular_x": [],
            "angular_y": [],
            "angular_z": [],
        }
        for topic in TWIST_STAMPED_TOPICS
    }

    gps_data = {
        topic: {
            "t": [],
            "latitude": [],
            "longitude": [],
            "altitude": [],
        }
        for topic in GPS_TOPICS
    }

    all_topics = (
        FLOAT64_TOPICS
        + TWIST_TOPICS
        + TWIST_STAMPED_TOPICS
        + GPS_TOPICS
    )

    with AnyReader([bag_path]) as reader:
        available_topics = sorted({c.topic for c in reader.connections})

        print("\nAvailable topics in bag:")
        for topic in available_topics:
            print(f"  {topic}")

        connections = [
            c for c in reader.connections
            if c.topic in all_topics
        ]

        if not connections:
            print("\nNo matching topics found.")
            return float_data, twist_data, twist_stamped_data, gps_data

        print("\nPlotting these topics:")
        for c in connections:
            print(f"  {c.topic}  [{c.msgtype}]")

        t0_ns = None

        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            if t0_ns is None:
                t0_ns = timestamp_ns

            t = ns_to_sec(timestamp_ns, t0_ns)
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if topic in FLOAT64_TOPICS:
                float_data[topic]["t"].append(t)
                float_data[topic]["y"].append(float(msg.data))

            elif topic in TWIST_TOPICS:
                d = twist_data[topic]
                d["t"].append(t)
                d["linear_x"].append(float(msg.linear.x))
                d["linear_y"].append(float(msg.linear.y))
                d["linear_z"].append(float(msg.linear.z))
                d["angular_x"].append(float(msg.angular.x))
                d["angular_y"].append(float(msg.angular.y))
                d["angular_z"].append(float(msg.angular.z))

            elif topic in TWIST_STAMPED_TOPICS:
                d = twist_stamped_data[topic]
                d["t"].append(t)
                d["linear_x"].append(float(msg.twist.linear.x))
                d["linear_y"].append(float(msg.twist.linear.y))
                d["linear_z"].append(float(msg.twist.linear.z))
                d["angular_x"].append(float(msg.twist.angular.x))
                d["angular_y"].append(float(msg.twist.angular.y))
                d["angular_z"].append(float(msg.twist.angular.z))

            elif topic in GPS_TOPICS:
                d = gps_data[topic]
                d["t"].append(t)
                d["latitude"].append(float(msg.latitude))
                d["longitude"].append(float(msg.longitude))
                d["altitude"].append(float(msg.altitude))

    return float_data, twist_data, twist_stamped_data, gps_data


def plot_float64_topics(float_data):
    for topic, d in float_data.items():
        if not d["t"]:
            continue

        plt.figure()
        plt.plot(d["t"], d["y"])
        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.title(topic)
        plt.grid(True)


def plot_twist_topics(twist_data):
    for topic, d in twist_data.items():
        if not d["t"]:
            continue

        plt.figure()
        plt.plot(d["t"], d["linear_x"], label="linear.x")
        plt.plot(d["t"], d["linear_y"], label="linear.y")
        plt.plot(d["t"], d["linear_z"], label="linear.z")
        plt.xlabel("Time [s]")
        plt.ylabel("Linear velocity")
        plt.title(f"{topic} linear velocity")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(d["t"], d["angular_x"], label="angular.x")
        plt.plot(d["t"], d["angular_y"], label="angular.y")
        plt.plot(d["t"], d["angular_z"], label="angular.z")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity")
        plt.title(f"{topic} angular velocity")
        plt.legend()
        plt.grid(True)


def plot_twist_stamped_topics(twist_stamped_data):
    for topic, d in twist_stamped_data.items():
        if not d["t"]:
            continue

        plt.figure()
        plt.plot(d["t"], d["linear_x"], label="twist.linear.x")
        plt.plot(d["t"], d["linear_y"], label="twist.linear.y")
        plt.plot(d["t"], d["linear_z"], label="twist.linear.z")
        plt.xlabel("Time [s]")
        plt.ylabel("Linear velocity")
        plt.title(f"{topic} linear velocity")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(d["t"], d["angular_x"], label="twist.angular.x")
        plt.plot(d["t"], d["angular_y"], label="twist.angular.y")
        plt.plot(d["t"], d["angular_z"], label="twist.angular.z")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity")
        plt.title(f"{topic} angular velocity")
        plt.legend()
        plt.grid(True)


def plot_gps_topics(gps_data):
    for topic, d in gps_data.items():
        if not d["latitude"]:
            continue

        plt.figure()
        plt.plot(d["longitude"], d["latitude"], marker=".")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{topic} GPS path")
        plt.axis("equal")
        plt.grid(True)

        plt.figure()
        plt.plot(d["t"], d["altitude"])
        plt.xlabel("Time [s]")
        plt.ylabel("Altitude [m]")
        plt.title(f"{topic} altitude")
        plt.grid(True)


def save_all_figures(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        filename = output_dir / f"plot_{i:02d}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot useful data from a ROS 2 rosbag."
    )

    parser.add_argument(
        "bag_path",
        type=str,
        help="Path to rosbag folder containing metadata.yaml and .mcap files",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots as PNG files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="rosbag_plots",
        help="Folder where PNG plots are saved",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows",
    )

    args = parser.parse_args()

    bag_path = Path(args.bag_path)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    if bag_path.is_file():
        raise ValueError(
            "bag_path must be the rosbag folder, not metadata.yaml directly."
        )

    metadata_file = bag_path / "metadata.yaml"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No metadata.yaml found in: {bag_path}"
        )

    float_data, twist_data, twist_stamped_data, gps_data = read_rosbag(bag_path)

    plot_float64_topics(float_data)
    plot_twist_topics(twist_data)
    plot_twist_stamped_topics(twist_stamped_data)
    plot_gps_topics(gps_data)

    if args.save:
        save_all_figures(Path(args.output_dir))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()