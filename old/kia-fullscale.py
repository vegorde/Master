#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mpc_v4_car_compat.py

ROS2-kompatibel MPCv4-erstatning som snakker samme topic-/enhetsgrensesnitt
som lateral_mpc_node.cpp.

Input:
  - gnss/pose                 (geometry_msgs/PoseStamped)
  - vehicle/state             (car_control/VehicleState)
  - enable_path_following     (std_msgs/Bool)
  - gnss/gyro                 (geometry_msgs/Vector3Stamped)  # valgfri

Output:
  - cmd_vel                   (geometry_msgs/Twist)
      linear.x  = accel_cmd  [-1, 1]
      angular.z = torque_cmd [-1, 1]

  - path_following_status     (std_msgs/Bool)
  - path_visualization        (nav_msgs/Path)

  - lateral_mpc/cte_m
  - lateral_mpc/heading_error_deg
  - lateral_mpc/desired_delta_deg
  - lateral_mpc/actual_delta_deg
  - lateral_mpc/torque_cmd
  - lateral_mpc/progress_m
  - lateral_mpc/integral_cte

  - lateral_error
  - heading_error
  - path_follower/cmd_vel

Merknader:
  - Bruker samme fortegn og dynamikk som implementasjonen i lateral_mpc_node.cpp
  - Steering input fra bilen er rattgrader, internt brukes framaksel-radianer
  - Hvis python-osqp/scipy ikke finnes, brukes en trygg fallback-kontroller
"""

import math
import threading
from typing import List, Tuple, Optional

import numpy as np

try:
    import scipy.sparse as sp
    import osqp
    HAS_OSQP = True
except Exception:
    HAS_OSQP = False

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Path as PathMsg
from std_msgs.msg import Bool, Float64

# NB: Juster denne importen hvis VehicleState ligger i et annet Python-navnerom
from car_control.msg import VehicleState


# ============================================================
# Konstanter
# ============================================================

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ          # 0.05 s
WHEELBASE = 2.79               # Kia Niro [m]
STEERING_RATIO = 15.33         # rattgrader per road-wheel grad
MIN_SPEED = 0.3                # [m/s]
SOFT_START_DURATION = 3.0      # [s]


# ============================================================
# Hjelpefunksjoner
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Samme planar yaw-ekstraksjon som i lateral_mpc_node.cpp
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


def now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


# ============================================================
# Piecewise-linear bane i map/ENU-ramme
# ============================================================

class ReferencePath:
    def __init__(self) -> None:
        self.wpts: List[Tuple[float, float]] = []
        self.s: List[float] = []

    def clear(self) -> None:
        self.wpts.clear()
        self.s.clear()

    def is_empty(self) -> bool:
        return len(self.wpts) == 0

    def add_waypoint(self, x: float, y: float) -> None:
        if not self.wpts:
            self.s.append(0.0)
        else:
            dx = x - self.wpts[-1][0]
            dy = y - self.wpts[-1][1]
            self.s.append(self.s[-1] + math.hypot(dx, dy))
        self.wpts.append((x, y))

    def total_length(self) -> float:
        return self.s[-1] if self.s else 0.0

    def smooth(self, passes: int = 2) -> None:
        """
        Savitzky-Golay 9-punkt, orden 3, samme koeffisienter som C++-noden.
        """
        W = 9
        H = W // 2
        c = np.array([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], dtype=float)
        norm = 231.0

        n = len(self.wpts)
        if n < W:
            return

        xs = np.array([p[0] for p in self.wpts], dtype=float)
        ys = np.array([p[1] for p in self.wpts], dtype=float)

        for _ in range(passes):
            x_new = np.zeros(n, dtype=float)
            y_new = np.zeros(n, dtype=float)
            for i in range(n):
                sx = 0.0
                sy = 0.0
                for k in range(-H, H + 1):
                    j = int(np.clip(i + k, 0, n - 1))
                    sx += c[k + H] * xs[j]
                    sy += c[k + H] * ys[j]
                x_new[i] = sx / norm
                y_new[i] = sy / norm
            xs = x_new
            ys = y_new

        self.wpts = [(float(xs[i]), float(ys[i])) for i in range(n)]

        self.s = [0.0]
        for i in range(1, n):
            dx = self.wpts[i][0] - self.wpts[i - 1][0]
            dy = self.wpts[i][1] - self.wpts[i - 1][1]
            self.s.append(self.s[-1] + math.hypot(dx, dy))

    def _interp(self, s_query: float) -> Tuple[float, float]:
        if not self.wpts:
            return 0.0, 0.0
        if s_query <= 0.0:
            return self.wpts[0]
        if s_query >= self.s[-1]:
            return self.wpts[-1]

        # lineær interpolasjon i arclength
        idx = int(np.searchsorted(np.array(self.s), s_query, side="left"))
        if idx <= 0:
            return self.wpts[0]
        idx = min(idx, len(self.wpts) - 1)

        s0 = self.s[idx - 1]
        s1 = self.s[idx]
        if abs(s1 - s0) < 1e-12:
            t = 0.0
        else:
            t = (s_query - s0) / (s1 - s0)
        t = clamp(t, 0.0, 1.0)

        x = self.wpts[idx - 1][0] + t * (self.wpts[idx][0] - self.wpts[idx - 1][0])
        y = self.wpts[idx - 1][1] + t * (self.wpts[idx][1] - self.wpts[idx - 1][1])
        return x, y

    def position(self, s_query: float) -> Tuple[float, float]:
        return self._interp(s_query)

    def heading(self, s_query: float) -> float:
        ds = 0.2
        s0 = max(0.0, s_query - ds)
        s1 = min(self.total_length(), s_query + ds)
        x0, y0 = self._interp(s0)
        x1, y1 = self._interp(s1)
        return math.atan2(y1 - y0, x1 - x0)

    def curvature(self, s_query: float) -> float:
        ds = 0.5
        h0 = self.heading(max(0.0, s_query - ds))
        h1 = self.heading(min(self.total_length(), s_query + ds))
        dh = wrap_angle(h1 - h0)
        return dh / (2.0 * ds)

    def cross_track_error(self, qx: float, qy: float, s_query: float) -> float:
        """
        Bruker SAMME implementasjon som lateral_mpc_node.cpp.

        Merk:
          Kommentaren i C++ sier "positive = left of path", men selve implementasjonen
          gir i praksis motsatt fortegn i standard ENU for en østgående bane.
          Her matcher vi IMPLEMENTASJONEN for full kompatibilitet.
        """
        px, py = self._interp(s_query)
        h = self.heading(s_query)
        nx = math.sin(h)
        ny = -math.cos(h)
        return (qx - px) * nx + (qy - py) * ny

    def heading_error(self, s_query: float, car_heading: float) -> float:
        """
        Samme som C++:
          err = path_heading - car_heading
        Positiv betyr: bilen peker til høyre for banen.
        """
        return wrap_angle(self.heading(s_query) - car_heading)

    def find_closest(self, qx: float, qy: float, hint_idx: int) -> Tuple[float, int]:
        if len(self.wpts) < 2:
            return 0.0, 0

        hint_idx = int(clamp(hint_idx, 0, len(self.wpts) - 2))
        end_idx = min(len(self.wpts) - 1, hint_idx + 300)

        best_sq = float("inf")
        best_s = self.s[hint_idx]
        best_idx = hint_idx

        for i in range(hint_idx, end_idx):
            ax, ay = self.wpts[i]
            bx, by = self.wpts[i + 1]
            dx = bx - ax
            dy = by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue

            t = ((qx - ax) * dx + (qy - ay) * dy) / seg2
            t = clamp(t, 0.0, 1.0)

            px = ax + t * dx
            py = ay + t * dy
            d2 = (qx - px) ** 2 + (qy - py) ** 2

            if d2 < best_sq:
                best_sq = d2
                best_s = self.s[i] + t * math.sqrt(seg2)
                best_idx = i

        return best_s, best_idx


# ============================================================
# MPCv4 kompatibel node
# ============================================================

class MpcV4CarCompatNode(Node):
    def __init__(self) -> None:
        super().__init__("mpc_v4_car_compat_node")

        # ----------------------------------------------------
        # Parametere
        # ----------------------------------------------------
        self.declare_parameter("desired_speed_mps", 4.0)
        self.declare_parameter("stop_distance", 3.0)
        self.declare_parameter("horizon", 40)

        self.declare_parameter("weight_cte", 2.0)
        self.declare_parameter("weight_psi", 2.0)
        self.declare_parameter("weight_torque", 0.1)

        self.declare_parameter("tau_r", 0.78)
        self.declare_parameter("gain_r", 36.0)          # sw-deg/s per torque unit
        self.declare_parameter("rate_up", 3.1 / 3.0)
        self.declare_parameter("rate_down", 5.5 / 3.0)
        self.declare_parameter("torque_limit", 1.0)

        self.declare_parameter("kp_speed", 0.3)

        self.declare_parameter("tau_i_cte", 8.0)
        self.declare_parameter("ki_cte", 0.0)

        self.declare_parameter("path_csv_file", "")
        self.declare_parameter("path_frame", "map")
        self.declare_parameter("auto_enable", False)

        # Fallback hvis OSQP ikke finnes
        self.declare_parameter("fallback_kp_cte", 0.30)
        self.declare_parameter("fallback_kp_psi", 1.40)
        self.declare_parameter("fallback_kp_delta", 0.25)
        self.declare_parameter("fallback_kd_drate", 0.08)

        self.N = int(clamp(
            int(self.get_parameter("horizon").value),
            1,
            64
        ))

        # ----------------------------------------------------
        # Intern state
        # ----------------------------------------------------
        self.state = "IDLE"              # IDLE / FOLLOWING / STOPPING
        self.path_start_time_sec = now_sec(self)

        self.lock = threading.Lock()

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_heading = 0.0
        self.car_speed_mps = 0.0
        self.car_delta_rad = 0.0          # framaksel-vinkel [rad]
        self.car_delta_rate = 0.0         # framaksel-rate [rad/s]
        self.prev_delta_rad = 0.0
        self.prev_delta_time = 0.0

        self.yaw_rate = 0.0               # fra gnss/gyro, valgfri, brukes ikke i solver
        self.gnss_valid = False

        self.auto_enable = bool(self.get_parameter("auto_enable").value)
        self.auto_enable_fired = False

        self.u_prev = 0.0
        self.integral_cte = 0.0

        self.path = ReferencePath()
        self.hint_idx = 0

        # ----------------------------------------------------
        # QoS
        # ----------------------------------------------------
        debug_qos = QoSProfile(depth=10)

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------
        self.sub_pose = self.create_subscription(
            PoseStamped,
            "gnss/pose",
            self.gnss_pose_callback,
            10
        )

        self.sub_vehicle = self.create_subscription(
            VehicleState,
            "vehicle/state",
            self.vehicle_state_callback,
            10
        )

        self.sub_enable = self.create_subscription(
            Bool,
            "enable_path_following",
            self.enable_callback,
            10
        )

        self.sub_gyro = self.create_subscription(
            Vector3Stamped,
            "gnss/gyro",
            self.gnss_gyro_callback,
            10
        )

        # ----------------------------------------------------
        # Publishers
        # ----------------------------------------------------
        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_status = self.create_publisher(Bool, "path_following_status", latched_qos)
        self.pub_path_vis = self.create_publisher(PathMsg, "path_visualization", latched_qos)

        self.pub_cte = self.create_publisher(Float64, "lateral_mpc/cte_m", debug_qos)
        self.pub_hdg_err_deg = self.create_publisher(Float64, "lateral_mpc/heading_error_deg", debug_qos)
        self.pub_desired_delta_deg = self.create_publisher(Float64, "lateral_mpc/desired_delta_deg", debug_qos)
        self.pub_actual_delta_deg = self.create_publisher(Float64, "lateral_mpc/actual_delta_deg", debug_qos)
        self.pub_torque_cmd = self.create_publisher(Float64, "lateral_mpc/torque_cmd", debug_qos)
        self.pub_progress = self.create_publisher(Float64, "lateral_mpc/progress_m", debug_qos)
        self.pub_integral_cte = self.create_publisher(Float64, "lateral_mpc/integral_cte", debug_qos)

        self.pub_lateral_error = self.create_publisher(Float64, "lateral_error", debug_qos)
        self.pub_heading_error = self.create_publisher(Float64, "heading_error", debug_qos)
        self.pub_pf_cmd_vel = self.create_publisher(Twist, "path_follower/cmd_vel", debug_qos)

        # ----------------------------------------------------
        # Bygg bane
        # ----------------------------------------------------
        csv_file = str(self.get_parameter("path_csv_file").value).strip()
        if csv_file:
            self.load_path_from_csv(csv_file)
        else:
            self.create_sinusoidal_path(
                total_length=500.0,
                init_amp=3.0,
                final_amp=8.0,
                init_wl=80.0,
                final_wl=50.0,
                spacing=1.0
            )

        self.publish_path_visualization()

        # ----------------------------------------------------
        # Timer
        # ----------------------------------------------------
        self.timer = self.create_timer(DT, self.control_loop)

        self.get_logger().info(
            f"MpcV4CarCompatNode ready | path={self.path.total_length():.1f} m | N={self.N} | "
            f"osqp={'yes' if HAS_OSQP else 'no (fallback active)'}"
        )

        if self.auto_enable:
            self.get_logger().info("auto_enable=true: starter automatisk etter første GNSS-fix.")

    # ========================================================
    # Callback-er
    # ========================================================

    def gnss_pose_callback(self, msg: PoseStamped) -> None:
        with self.lock:
            self.car_x = float(msg.pose.position.x)
            self.car_y = float(msg.pose.position.y)

            q = msg.pose.orientation
            self.car_heading = quaternion_to_yaw(q.x, q.y, q.z, q.w)

            was_valid = self.gnss_valid
            self.gnss_valid = True

        if self.auto_enable and (not was_valid) and (not self.auto_enable_fired):
            self.auto_enable_fired = True
            self.get_logger().info("Første GNSS-fix mottatt. Auto-enable om 1 sekund.")
            self.create_timer(1.0, self._auto_enable_once)

    def _auto_enable_once(self) -> None:
        # Enkel one-shot semantikk: start bare hvis fortsatt idle
        if self.state == "IDLE":
            msg = Bool()
            msg.data = True
            self.enable_callback(msg)

    def vehicle_state_callback(self, msg: VehicleState) -> None:
        with self.lock:
            # v_ego [km/h] -> [m/s]
            self.car_speed_mps = float(msg.v_ego) / 3.6

            # steering_angle_deg [rattgrader] -> framaksel [rad]
            new_delta = math.radians(float(msg.steering_angle_deg)) / STEERING_RATIO

            t = now_sec(self)
            if self.prev_delta_time > 0.0:
                dt_meas = t - self.prev_delta_time
                if 0.0 < dt_meas < 1.0:
                    self.car_delta_rate = (new_delta - self.prev_delta_rad) / dt_meas

            self.prev_delta_rad = new_delta
            self.prev_delta_time = t
            self.car_delta_rad = new_delta

    def gnss_gyro_callback(self, msg: Vector3Stamped) -> None:
        with self.lock:
            self.yaw_rate = float(msg.vector.z)

    def enable_callback(self, msg: Bool) -> None:
        if not msg.data:
            if self.state in ("FOLLOWING", "STOPPING"):
                self.get_logger().info("Path following STOPPED by user.")
                self.state = "IDLE"
                self.publish_cmd(0.0, 0.0)
                self.publish_status(False)
            return

        # true -> start hvis IDLE, ellers stopp (samme toggle-semantikk)
        if self.state == "IDLE":
            if not self.gnss_valid:
                self.get_logger().warn("Kan ikke starte: ingen GNSS-fix mottatt ennå.")
                return
            if self.path.is_empty():
                self.get_logger().warn("Kan ikke starte: banen er tom.")
                return

            self.hint_idx = 0
            self.u_prev = 0.0
            self.integral_cte = 0.0
            self.path_start_time_sec = now_sec(self)
            self.state = "FOLLOWING"
            self.get_logger().info("Path following STARTED.")
            self.publish_status(True)

        elif self.state in ("FOLLOWING", "STOPPING"):
            self.get_logger().info("Path following STOPPED by user.")
            self.state = "IDLE"
            self.publish_cmd(0.0, 0.0)
            self.publish_status(False)

    # ========================================================
    # Path-hjelpere
    # ========================================================

    def create_sinusoidal_path(
        self,
        total_length: float,
        init_amp: float,
        final_amp: float,
        init_wl: float,
        final_wl: float,
        spacing: float
    ) -> None:
        self.path.clear()
        self.hint_idx = 0
        self.integral_cte = 0.0

        sinusoid_len = total_length - 60.0
        n_sin = int(sinusoid_len / spacing)
        n_tot = int(total_length / spacing)

        self.path.add_waypoint(0.0, 0.0)

        for i in range(1, n_sin + 1):
            x = i * spacing
            progress = x / sinusoid_len
            amp = init_amp + (final_amp - init_amp) * progress
            wl = init_wl + (final_wl - init_wl) * progress
            y = amp * math.sin(2.0 * math.pi / wl * x)
            self.path.add_waypoint(x, y)

        last_s = sinusoid_len
        last_amp = final_amp
        last_wl = final_wl
        last_y = last_amp * math.sin(2.0 * math.pi / last_wl * last_s)
        taper_len = 15.0

        for i in range(n_sin + 1, n_tot + 1):
            x = i * spacing
            d_end = x - sinusoid_len
            taper = max(0.0, 1.0 - d_end / taper_len)
            self.path.add_waypoint(x, last_y * taper)

        self.get_logger().info(
            f"Sinusoidal path created: {self.path.total_length():.1f} m total, {n_tot + 1} waypoints."
        )

    def load_path_from_csv(self, filename: str) -> None:
        self.path.clear()
        self.hint_idx = 0
        self.integral_cte = 0.0

        count = 0
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        continue
                    if line[0] in ("x", "X"):
                        continue

                    line = line.replace(",", " ")
                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                    except ValueError:
                        continue

                    self.path.add_waypoint(x, y)
                    count += 1

        except Exception as exc:
            self.get_logger().error(f"Kan ikke åpne path CSV: {filename} | {exc}")
            self.create_sinusoidal_path(500.0, 3.0, 8.0, 80.0, 50.0, 1.0)
            return

        if count < 2:
            self.get_logger().error(
                f"CSV '{filename}' har færre enn 2 waypoints – faller tilbake til sinusoidal bane."
            )
            self.create_sinusoidal_path(500.0, 3.0, 8.0, 80.0, 50.0, 1.0)
            return

        self.get_logger().info(
            f"Loaded path from '{filename}': {count} waypoints, {self.path.total_length():.1f} m total."
        )
        self.path.smooth()
        self.get_logger().info(
            f"Path smoothed (SG 9-pt order-3, 2 passes). New length: {self.path.total_length():.1f} m."
        )

    # ========================================================
    # Publisering
    # ========================================================

    def publish_cmd(self, accel_cmd: float, torque_cmd: float) -> None:
        msg = Twist()
        msg.linear.x = float(accel_cmd)    # accel [-1, 1]
        msg.angular.z = float(torque_cmd)  # torque [-1, 1]
        self.pub_cmd_vel.publish(msg)

    def publish_status(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self.pub_status.publish(msg)

    def publish_f64(self, pub, value: float) -> None:
        m = Float64()
        m.data = float(value)
        pub.publish(m)

    def publish_path_visualization(self) -> None:
        if self.path.is_empty():
            return

        msg = PathMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("path_frame").value)

        total = self.path.total_length()
        n_vis = 200
        step = total / max(1, (n_vis - 1))

        for i in range(n_vis):
            s = i * step
            x, y = self.path.position(s)
            h = self.path.heading(s)

            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = math.cos(h / 2.0)
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = math.sin(h / 2.0)

            msg.poses.append(ps)

        self.pub_path_vis.publish(msg)

    # ========================================================
    # Kontrollsløyfe
    # ========================================================

    def control_loop(self) -> None:
        if self.state == "IDLE":
            return

        with self.lock:
            car_x = self.car_x
            car_y = self.car_y
            car_heading = self.car_heading
            car_speed = self.car_speed_mps
            car_delta = self.car_delta_rad
            car_delta_rate = self.car_delta_rate

        # Stopp ferdig
        if self.state == "STOPPING" and car_speed <= MIN_SPEED:
            self.get_logger().info("Path end reached. Returning to IDLE.")
            self.state = "IDLE"
            self.publish_cmd(0.0, 0.0)
            self.publish_status(False)
            return

        # Finn progress
        s_rear, self.hint_idx = self.path.find_closest(car_x, car_y, self.hint_idx)

        # Sjekk baneende
        remaining = self.path.total_length() - s_rear
        stop_distance = float(self.get_parameter("stop_distance").value)
        if remaining < stop_distance and self.state == "FOLLOWING":
            self.get_logger().info(
                f"Approaching path end ({remaining:.1f} m remaining). Slowing down..."
            )
            self.state = "STOPPING"

        # Soft-start
        elapsed = now_sec(self) - self.path_start_time_sec
        ramp = min(1.0, elapsed / SOFT_START_DURATION)

        # Feil i SAMME konvensjon som lateral_mpc_node.cpp
        cte = self.path.cross_track_error(car_x, car_y, s_rear)
        dpsi = self.path.heading_error(s_rear, car_heading)

        # Feedforward for debug
        desired_delta_rad = self.path.curvature(s_rear) * WHEELBASE

        # Leaky integrator
        tau_i = float(self.get_parameter("tau_i_cte").value)
        tau_i = max(tau_i, DT)
        self.integral_cte = self.integral_cte * (1.0 - DT / tau_i) + DT * cte
        ki_cte = float(self.get_parameter("ki_cte").value)
        cte_biased = cte + ki_cte * self.integral_cte

        # Lateral styring
        torque_cmd = 0.0
        if self.state == "FOLLOWING":
            torque_cmd = ramp * self.solve_mpc(
                cte0=cte_biased,
                dpsi0=dpsi,
                delta0=car_delta,
                drate0=car_delta_rate,
                v=car_speed,
                s_ref=s_rear
            )

        torque_cmd = clamp(torque_cmd, -1.0, 1.0)
        self.u_prev = torque_cmd

        # Longitudinal: speed P -> accel_cmd [-1, 1]
        desired_speed = 0.0 if self.state == "STOPPING" else float(self.get_parameter("desired_speed_mps").value)
        kp_speed = float(self.get_parameter("kp_speed").value)
        accel_cmd = clamp(kp_speed * (desired_speed - car_speed), -1.0, 1.0)

        # Publiser hovedkommando
        self.publish_cmd(accel_cmd, torque_cmd)

        # Debug
        self.publish_f64(self.pub_cte, cte)
        self.publish_f64(self.pub_hdg_err_deg, math.degrees(dpsi))
        self.publish_f64(self.pub_desired_delta_deg, math.degrees(desired_delta_rad))
        self.publish_f64(self.pub_actual_delta_deg, math.degrees(car_delta))
        self.publish_f64(self.pub_torque_cmd, torque_cmd)
        self.publish_f64(self.pub_progress, s_rear)
        self.publish_f64(self.pub_integral_cte, self.integral_cte)

        # Dashboard-kompatibilitet
        self.publish_f64(self.pub_lateral_error, cte)
        self.publish_f64(self.pub_heading_error, dpsi)   # rad

        pf_cmd = Twist()
        pf_cmd.linear.x = float(desired_speed)           # m/s
        pf_cmd.angular.z = float(desired_delta_rad)      # framaksel [rad]
        self.pub_pf_cmd_vel.publish(pf_cmd)

    # ========================================================
    # MPC / fallback
    # ========================================================

    def solve_mpc(
        self,
        cte0: float,
        dpsi0: float,
        delta0: float,
        drate0: float,
        v: float,
        s_ref: float
    ) -> float:
        """
        Samme modellstruktur som lateral_mpc_node.cpp:

          x = [CTE, dPsi, delta, dRate]
          u = torque_cmd

          dRate_{k+1} = a_rate * dRate_k + b_rate * u_k
          delta_{k+1} = delta_k + dt * dRate_{k+1}
          dPsi_{k+1}  = dPsi_k - (v*delta/L - kappa*v)*dt
          CTE_{k+1}   = CTE_k + v*dPsi_k*dt
        """
        if not HAS_OSQP:
            return self.fallback_torque(cte0, dpsi0, delta0, drate0)

        n = self.N
        if n < 1:
            return self.u_prev

        dt = DT
        tau_r = float(self.get_parameter("tau_r").value)
        gain_r = float(self.get_parameter("gain_r").value)
        w_cte = float(self.get_parameter("weight_cte").value)
        w_psi = float(self.get_parameter("weight_psi").value)
        w_t = float(self.get_parameter("weight_torque").value)
        rate_up = float(self.get_parameter("rate_up").value)
        rate_down = float(self.get_parameter("rate_down").value)
        tlim = float(self.get_parameter("torque_limit").value)

        alpha = dt / (tau_r + dt)
        a_rate = 1.0 - alpha

        # 36 sw-deg/s/unit -> framaksel rad/s/unit
        gain_r_rad = math.radians(gain_r) / STEERING_RATIO
        b_rate = alpha * gain_r_rad

        v_eff = max(0.5, v)

        # Dense utgave først, deretter csc for OSQP
        P = np.zeros((n, n), dtype=float)
        q = np.zeros(n, dtype=float)

        c_rate = drate0
        c_delta = delta0
        c_psi = dpsi0
        c_cte = cte0

        G_rate = np.zeros(n, dtype=float)
        G_delta = np.zeros(n, dtype=float)
        G_psi = np.zeros(n, dtype=float)
        G_cte = np.zeros(n, dtype=float)

        for k in range(n):
            s_k = min(s_ref + k * dt * v_eff, self.path.total_length())
            kappa = self.path.curvature(s_k)

            # dRate
            c_rate_new = a_rate * c_rate
            G_rate_new = a_rate * G_rate
            G_rate_new[k] += b_rate

            # delta
            c_delta_new = c_delta + dt * c_rate_new
            G_delta_new = G_delta + dt * G_rate_new

            # dPsi (merk samme fortegn som C++-implementasjonen)
            yaw_rate_k = v_eff * c_delta / WHEELBASE
            c_psi_new = c_psi - (yaw_rate_k - kappa * v_eff) * dt
            G_psi_new = G_psi - (v_eff * dt / WHEELBASE) * G_delta

            # CTE
            c_cte_new = c_cte + v_eff * c_psi * dt
            G_cte_new = G_cte + v_eff * dt * G_psi

            c_rate = c_rate_new
            G_rate = G_rate_new
            c_delta = c_delta_new
            G_delta = G_delta_new
            c_psi = c_psi_new
            G_psi = G_psi_new
            c_cte = c_cte_new
            G_cte = G_cte_new

            # Kost
            P += 2.0 * w_cte * np.outer(G_cte, G_cte)
            P += 2.0 * w_psi * np.outer(G_psi, G_psi)
            q += 2.0 * w_cte * c_cte * G_cte + 2.0 * w_psi * c_psi * G_psi
            P[k, k] += 2.0 * w_t

        # Lett regularisering
        P += 1e-9 * np.eye(n)

        # Begrensninger
        # row 4k+0:  U_k - U_{k-1} <= rate_up*dt
        # row 4k+1:  U_{k-1} - U_k <= rate_down*dt
        # row 4k+2:  U_k <= tlim
        # row 4k+3: -U_k <= tlim

        m = 4 * n
        A = np.zeros((m, n), dtype=float)
        l = -np.inf * np.ones(m, dtype=float)
        u = np.inf * np.ones(m, dtype=float)

        for k in range(n):
            # Rate up
            A[4 * k + 0, k] = 1.0
            # Rate down
            A[4 * k + 1, k] = -1.0
            # Magnitude upper
            A[4 * k + 2, k] = 1.0
            # Magnitude lower
            A[4 * k + 3, k] = -1.0

            if k == 0:
                u[4 * k + 0] = self.u_prev + rate_up * dt
                u[4 * k + 1] = -self.u_prev + rate_down * dt
            else:
                A[4 * k + 0, k - 1] = -1.0
                A[4 * k + 1, k - 1] = 1.0
                u[4 * k + 0] = rate_up * dt
                u[4 * k + 1] = rate_down * dt

            u[4 * k + 2] = tlim
            u[4 * k + 3] = tlim

        try:
            solver = osqp.OSQP()
            solver.setup(
                P=sp.csc_matrix(np.triu(P)),
                q=q,
                A=sp.csc_matrix(A),
                l=l,
                u=u,
                verbose=False,
                polish=False,
                warm_start=False,
                adaptive_rho=True,
                max_iter=4000
            )
            res = solver.solve()

            if res.x is None:
                self.get_logger().warn("OSQP returned no solution; holder previous torque.")
                return self.u_prev

            status = str(res.info.status).lower()
            if "solved" not in status:
                self.get_logger().warn(f"OSQP solve failed ({res.info.status}); holder previous torque.")
                return self.u_prev

            u0 = float(res.x[0])
            return clamp(u0, -tlim, tlim)

        except Exception as exc:
            self.get_logger().warn(f"OSQP exception: {exc}; using fallback torque.")
            return self.fallback_torque(cte0, dpsi0, delta0, drate0)

    def fallback_torque(self, cte: float, dpsi: float, delta: float, drate: float) -> float:
        """
        Konservativ fallback hvis OSQP ikke finnes.
        Matcher samme fortegn som resten av noden:
          positiv CTE/dPsi -> positiv torque (venstrestyring)
        """
        kp_cte = float(self.get_parameter("fallback_kp_cte").value)
        kp_psi = float(self.get_parameter("fallback_kp_psi").value)
        kp_delta = float(self.get_parameter("fallback_kp_delta").value)
        kd_drate = float(self.get_parameter("fallback_kd_drate").value)
        rate_up = float(self.get_parameter("rate_up").value)
        rate_down = float(self.get_parameter("rate_down").value)
        tlim = float(self.get_parameter("torque_limit").value)

        u = (
            kp_cte * cte +
            kp_psi * dpsi -
            kp_delta * delta -
            kd_drate * drate
        )

        # Samme slew-limit-idé som i C++
        u = min(u, self.u_prev + rate_up * DT)
        u = max(u, self.u_prev - rate_down * DT)

        u = clamp(u, -tlim, tlim)
        return u


# ============================================================
# main
# ============================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcV4CarCompatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd(0.0, 0.0)
            node.publish_status(False)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
