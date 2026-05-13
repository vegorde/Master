#!/usr/bin/env python3
"""
mpc_v4_ipopt_car_minimal.py

Minimal ROS2 adaptation of the original mpc v4.py:
- Keeps the nonlinear CasADi/IPOPT MPC structure, but embeds the steering actuator:
- states [e, psi_err, theta_sw_deg]
- input normalized steering torque u in [-1, 1]
- Replaces TCP/JSON server with ROS2 car interface.
- Subscribes:
    gnss/pose             geometry_msgs/PoseStamped
    vehicle/state         car_control/VehicleState
    enable_path_following std_msgs/Bool
- Publishes:
    cmd_vel               geometry_msgs/Twist
        linear.x  = accel_cmd [-1, 1]
        angular.z = torque_cmd [-1, 1]
- Publishes debug topics compatible with current lateral_mpc_node.cpp.

Important:
The nonlinear MPC now optimizes normalized steering torque directly. Steering-wheel angle
is part of the MPC state and is propagated using the identified speed-scheduled first-order
steering model from message(5). No post-MPC delta-to-torque PID/PD adapter is used.

If IPOPT fails, no PD fallback MPC is used. The node either holds the last torque command
or sends zero depending on parameter hold_last_on_solver_fail.
"""

import math
import time
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline, PchipInterpolator

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float64
from nav_msgs.msg import Path as RosPath

from car_control.msg import VehicleState


# ---------------- Vehicle / interface constants ----------------
CONTROL_HZ = 10.0
ROS_DT = 1.0 / CONTROL_HZ
WHEELBASE = 2.79
STEERING_RATIO = 15.33
MIN_SPEED = 0.3
SOFT_START_DURATION = 3.0


# ---------------- MPC setup, kept nonlinear/IPOPT ----------------
# State: [e, psi_err, theta_sw_deg]
# Input: normalized steering torque [-1, 1]
nx, nu = 3, 1


# ---------------- Identified steering actuator model ----------------
# Model from message(5):
#   theta[k+1] = a(v) * theta[k] + b(v) * u[k]
# where theta is steering-wheel angle [deg] and u is normalized torque [-1, 1].
SCHED_V_KMH = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0, 40.0, 45.0, 50.0], dtype=float)
SCHED_TAU_R = np.array([2.953, 1.386, 0.722, 0.627, 0.437, 0.390, 0.342, 0.295, 0.247, 0.200], dtype=float)
SCHED_KSS = np.array([501.9, 314.9, 187.4, 143.2, 105.5, 84.6, 71.2, 63.3, 53.9, 44.1], dtype=float)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def closest_index_windowed(x, y, xref, yref, last_idx=0, window=100):
    n = len(xref)
    i0 = max(0, last_idx - window)
    i1 = min(n, last_idx + window + 1)
    dx = xref[i0:i1] - x
    dy = yref[i0:i1] - y
    d2 = dx * dx + dy * dy
    return i0 + int(np.argmin(d2))


def steering_ab_from_speed(v_mps: float, dt: float) -> Tuple[float, float]:
    """
    Return the discrete steering actuator coefficients a, b for:
        theta_sw_deg[k+1] = a * theta_sw_deg[k] + b * torque[k]

    theta_sw_deg is steering-wheel angle in degrees.
    torque is normalized steering torque in [-1, 1].
    """
    v_kmh = float(v_mps) * 3.6
    tau = float(np.interp(v_kmh, SCHED_V_KMH, SCHED_TAU_R))
    kss = float(np.interp(v_kmh, SCHED_V_KMH, SCHED_KSS))

    a = math.exp(-dt / max(tau, 1e-3))
    b = kss * (1.0 - a)
    return a, b


def load_waypoints_csv(filename: str, default_speed: float) -> List[Tuple[float, float, float]]:
    """Load CSV with x,y or x,y,v. Skips comments and headers."""
    waypoints: List[Tuple[float, float, float]] = []
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line[0].lower() == "x":
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) < 2:
                continue
            x = float(parts[0])
            y = float(parts[1])
            v = float(parts[2]) if len(parts) >= 3 else default_speed
            waypoints.append((x, y, v))
    if len(waypoints) < 2:
        raise ValueError(f"Path CSV '{filename}' has fewer than 2 valid waypoints")
    return waypoints


def compute_heading_and_curvature_from_spline(waypoints, ds, kind_xy="cubic", kind_v="pchip"):
    wp = np.asarray(waypoints, dtype=float)
    xw, yw, vw = wp[:, 0], wp[:, 1], wp[:, 2]

    dx = np.diff(xw)
    dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    keep = np.ones(len(wp), dtype=bool)
    keep[1:] = seg > 1e-9
    xw, yw, vw = xw[keep], yw[keep], vw[keep]

    dx = np.diff(xw)
    dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    n = max(1, int(np.floor(total / ds)))
    s_ref = np.linspace(0.0, total, n + 1)

    def make_interp(kind, t, z):
        if kind == "cubic":
            return CubicSpline(t, z, bc_type="natural")
        if kind == "pchip":
            return PchipInterpolator(t, z)
        raise ValueError(f"Unknown interpolation kind '{kind}'")

    fx = make_interp(kind_xy, s, xw)
    fy = make_interp(kind_xy, s, yw)
    fv = make_interp(kind_v, s, vw)

    xref = np.asarray(fx(s_ref))
    yref = np.asarray(fy(s_ref))
    vref = np.asarray(fv(s_ref))

    x_s = fx.derivative(1)(s_ref)
    y_s = fy.derivative(1)(s_ref)
    x_ss = fx.derivative(2)(s_ref)
    y_ss = fy.derivative(2)(s_ref)

    psiref = np.unwrap(np.arctan2(y_s, x_s))

    denom = np.maximum((x_s ** 2 + y_s ** 2) ** 1.5, 1e-12)
    kappa_points = (x_s * y_ss - y_s * x_ss) / denom
    kappa_path = 0.5 * (kappa_points[:-1] + kappa_points[1:])
    kappa_path = smooth(np.asarray(kappa_path), w=9)

    return xref, yref, psiref, vref, kappa_path, s_ref


class NonlinearIpoptMpc:
    """
    Nonlinear CasADi/IPOPT MPC with embedded steering actuator dynamics.

    State:
        x = [e, psi_err, theta_sw_deg]

    Input:
        u = normalized steering torque [-1, 1]
    """

    def __init__(self, dt: float, horizon: int, wheelbase: float,
                 q_e: float, q_psi: float, r_torque: float, r_d_torque: float,
                 delta_max: float, steering_ratio: float):
        self.dt = dt
        self.N = int(horizon)
        self.L = wheelbase
        self.Qe = q_e
        self.Qpsi = q_psi
        self.Rtorque = r_torque
        self.Rdtorque = r_d_torque
        self.delta_max = delta_max
        self.steering_ratio = steering_ratio
        self.theta_sw_max_deg = math.degrees(delta_max * steering_ratio)
        self.prev_z: Optional[np.ndarray] = None
        self.solver, self.lbg, self.ubg, self.lbz, self.ubz, self.unpack = self._build_mpc_nlp()

    def _build_mpc_nlp(self):
        N = self.N
        dt = self.dt
        L = self.L
        steering_ratio = self.steering_ratio

        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)

        x0_p = ca.SX.sym("x0", nx)
        kappa_p = ca.SX.sym("kappa", N)
        v_p = ca.SX.sym("v", N)
        steer_a_p = ca.SX.sym("steer_a", N)
        steer_b_p = ca.SX.sym("steer_b", N)

        def f_step(xk, uk, kappak, vk, steer_a, steer_b):
            # Nonlinear path-coordinate statespace with embedded steering actuator:
            # x = [e, psi_err, theta_sw_deg], u = normalized steering torque.
            e = xk[0]
            psi = xk[1]
            theta_sw_deg = xk[2]
            torque = uk[0]

            # Convert steering-wheel angle [deg] to front-wheel angle [rad].
            delta = (theta_sw_deg * math.pi / 180.0) / steering_ratio

            denom = 1 - kappak * e
            # avoid singularity in symbolic expression with small soft lower bound
            denom = ca.if_else(ca.fabs(denom) < 1e-3, ca.sign(denom) * 1e-3, denom)

            s_dot = vk * ca.cos(psi) / denom
            e_dot = vk * ca.sin(psi)
            psi_dot = (vk / L) * ca.tan(delta) - s_dot * kappak

            e_next = e + dt * e_dot
            psi_next = psi + dt * psi_dot
            theta_next = steer_a * theta_sw_deg + steer_b * torque

            return ca.vertcat(e_next, psi_next, theta_next)

        obj = 0
        g = []
        g.append(X[:, 0] - x0_p)

        for k in range(N):
            x_next = f_step(X[:, k], U[:, k], kappa_p[k], v_p[k], steer_a_p[k], steer_b_p[k])
            g.append(X[:, k + 1] - x_next)

            obj += self.Qe * (X[0, k] ** 2)
            obj += self.Qpsi * (X[1, k] ** 2)
            obj += self.Rtorque * ca.sumsqr(U[:, k])
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                obj += self.Rdtorque * ca.sumsqr(du)

        g = ca.vertcat(*g)
        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        p = ca.vertcat(x0_p, kappa_p, v_p, steer_a_p, steer_b_p)
        nlp = {"x": z, "f": obj, "g": g, "p": p}

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 1e-4,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        ng = g.size1()
        lbg = np.zeros(ng)
        ubg = np.zeros(ng)

        nX = nx * (N + 1)
        nU = nu * N
        nz = nX + nU
        lbz = -np.inf * np.ones(nz)
        ubz = np.inf * np.ones(nz)

        # Bound steering-wheel angle via equivalent front-wheel steering limit.
        for k in range(N + 1):
            idx = k * nx + 2
            lbz[idx] = -self.theta_sw_max_deg
            ubz[idx] = self.theta_sw_max_deg

        # Bound normalized torque command.
        for k in range(N):
            idx = nX + k * nu
            lbz[idx] = -1.0
            ubz[idx] = 1.0

        def unpack(z_val):
            z_val = np.asarray(z_val).reshape(-1)
            Xv = z_val[:nX].reshape((nx, N + 1), order="F")
            Uv = z_val[nX:].reshape((nu, N), order="F")
            return Xv, Uv

        return solver, lbg, ubg, lbz, ubz, unpack

    def step(self, x0, kappa_seq, v_seq, steer_a_seq, steer_b_seq) -> Tuple[bool, float, float]:
        """Return (success, torque_cmd, runtime_s). No controller fallback here."""
        x0 = np.asarray(x0).reshape(-1)
        kappa_seq = np.asarray(kappa_seq).reshape(-1)
        v_seq = np.asarray(v_seq).reshape(-1)
        steer_a_seq = np.asarray(steer_a_seq).reshape(-1)
        steer_b_seq = np.asarray(steer_b_seq).reshape(-1)

        if x0.size != nx:
            raise ValueError(f"x0 must have length {nx}")
        if kappa_seq.size != self.N or v_seq.size != self.N:
            raise ValueError("kappa_seq and v_seq must have length N")
        if steer_a_seq.size != self.N or steer_b_seq.size != self.N:
            raise ValueError("steer_a_seq and steer_b_seq must have length N")

        p = np.concatenate([x0, kappa_seq, v_seq, steer_a_seq, steer_b_seq])

        if self.prev_z is None:
            x_guess = np.zeros((nx, self.N + 1))
            u_guess = np.zeros((nu, self.N))
            x_guess[:, 0] = x0
            z0 = np.concatenate([x_guess.reshape(-1, order="F"), u_guess.reshape(-1, order="F")])
        else:
            z0 = self.prev_z

        start = time.time()
        try:
            sol = self.solver(
                x0=z0,
                p=p,
                lbg=self.lbg,
                ubg=self.ubg,
                lbx=self.lbz,
                ubx=self.ubz,
            )
        except RuntimeError:
            return False, 0.0, time.time() - start

        runtime = time.time() - start
        z_opt = np.array(sol["x"]).reshape(-1)
        X_opt, U_opt = self.unpack(z_opt)

        if not np.isfinite(U_opt[:, 0]).all():
            return False, 0.0, runtime

        self.prev_z = z_opt
        torque_cmd = float(np.clip(U_opt[:, 0].item(), -1.0, 1.0))
        return True, torque_cmd, runtime


class State(Enum):
    IDLE = 0
    FOLLOWING = 1
    STOPPING = 2


class MpcV4IpoptCarNode(Node):
    def __init__(self):
        super().__init__("mpc_v4_ipopt_car")

        # Parameters: kept close to your original MPC but made car-interface configurable.
        self.declare_parameter("path_csv_file", "")
        self.declare_parameter("default_speed_mps", 25/3.6)
        self.declare_parameter("desired_speed_mps", 4.0)
        self.declare_parameter("stop_distance", 3.0)
        self.declare_parameter("ds", 0.1)
        self.declare_parameter("mpc_dt", 0.1)
        self.declare_parameter("horizon", 60)
        self.declare_parameter("wheelbase", WHEELBASE)
        self.declare_parameter("weight_e", 0.01)
        self.declare_parameter("weight_psi", 0.1)
        self.declare_parameter("weight_delta", 0.01)
        self.declare_parameter("weight_d_delta", 1.0)
        self.declare_parameter("delta_max_deg", 15.0)
        self.declare_parameter("kp_speed", 0.3)

        # The MPC now outputs normalized steering torque directly.
        self.declare_parameter("hold_last_on_solver_fail", True)
        self.declare_parameter("auto_enable", False)

        self.mpc_dt = float(self.get_parameter("mpc_dt").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.delta_max = math.radians(float(self.get_parameter("delta_max_deg").value))

        self.mpc = NonlinearIpoptMpc(
            dt=self.mpc_dt,
            horizon=self.horizon,
            wheelbase=float(self.get_parameter("wheelbase").value),
            q_e=float(self.get_parameter("weight_e").value),
            q_psi=float(self.get_parameter("weight_psi").value),
            r_torque=float(self.get_parameter("weight_delta").value),
            r_d_torque=float(self.get_parameter("weight_d_delta").value),
            delta_max=self.delta_max,
            steering_ratio=STEERING_RATIO,
        )

        self.state = State.IDLE
        self.start_time = self.get_clock().now()
        self.last_mpc_time = -1e9
        self.last_idx = 0

        self.gnss_valid = False
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.car_speed_mps = 0.0
        self.theta_sw_deg = 0.0
        self.delta_meas = 0.0
        self.prev_delta_meas = 0.0
        self.delta_rate_meas = 0.0
        self.prev_delta_time = 0.0

        self.last_torque_cmd = 0.0
        self.integral_cte = 0.0

        self.xref, self.yref, self.psiref, self.vref, self.kappa_path, self.s_ref = self._load_path()

        # Subscribers
        self.create_subscription(PoseStamped, "gnss/pose", self.gnss_pose_cb, 10)
        self.create_subscription(VehicleState, "vehicle/state", self.vehicle_state_cb, 10)
        self.create_subscription(Bool, "enable_path_following", self.enable_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(Bool, "path_following_status", latched_qos)
        self.path_pub = self.create_publisher(RosPath, "path_visualization", latched_qos)

        self.pub_cte = self.create_publisher(Float64, "lateral_mpc/cte_m", 10)
        self.pub_hdg_deg = self.create_publisher(Float64, "lateral_mpc/heading_error_deg", 10)
        self.pub_desired_delta = self.create_publisher(Float64, "lateral_mpc/desired_delta_deg", 10)
        self.pub_actual_delta = self.create_publisher(Float64, "lateral_mpc/actual_delta_deg", 10)
        self.pub_torque = self.create_publisher(Float64, "lateral_mpc/torque_cmd", 10)
        self.pub_progress = self.create_publisher(Float64, "lateral_mpc/progress_m", 10)
        self.pub_lateral_error = self.create_publisher(Float64, "lateral_error", 10)
        self.pub_heading_error = self.create_publisher(Float64, "heading_error", 10)
        self.pub_pf_cmd = self.create_publisher(Twist, "path_follower/cmd_vel", 10)

        self.publish_path_visualization()

        self.timer = self.create_timer(ROS_DT, self.control_loop)

        self.auto_enable = bool(self.get_parameter("auto_enable").value)
        self.auto_enable_fired = False

        self.get_logger().info(
            f"MPCv4 IPOPT car node ready. Nonlinear MPC kept. N={self.horizon}, mpc_dt={self.mpc_dt}, path={len(self.xref)} points"
        )

    def _load_path(self):
        path_csv = str(self.get_parameter("path_csv_file").value)
        default_speed = float(self.get_parameter("default_speed_mps").value)
        ds = float(self.get_parameter("ds").value)

        waypoints = load_waypoints_csv(path_csv, default_speed)
        self.get_logger().info(f"Loaded path CSV '{path_csv}' with {len(waypoints)} waypoints")

        return compute_heading_and_curvature_from_spline(waypoints, ds, kind_xy="cubic", kind_v="pchip")

    def gnss_pose_cb(self, msg: PoseStamped):
        self.car_x = float(msg.pose.position.x)
        self.car_y = float(msg.pose.position.y)
        self.car_yaw = yaw_from_quat(msg.pose.orientation)
        was_valid = self.gnss_valid
        self.gnss_valid = True

        if self.auto_enable and not was_valid and not self.auto_enable_fired:
            self.auto_enable_fired = True
            b = Bool()
            b.data = True
            self.enable_cb(b)

    def vehicle_state_cb(self, msg: VehicleState):
        self.car_speed_mps = float(msg.v_ego) / 3.6
        self.theta_sw_deg = float(msg.steering_angle_deg)
        new_delta = math.radians(self.theta_sw_deg) / STEERING_RATIO

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_delta_time > 0.0:
            dt = now_s - self.prev_delta_time
            if 0.0 < dt < 1.0:
                self.delta_rate_meas = (new_delta - self.prev_delta_meas) / dt
        self.prev_delta_meas = new_delta
        self.prev_delta_time = now_s
        self.delta_meas = new_delta

    def enable_cb(self, msg: Bool):
        if not msg.data:
            if self.state in (State.FOLLOWING, State.STOPPING):
                self.get_logger().info("Path following STOPPED by user")
                self.state = State.IDLE
                self.publish_cmd(0.0, 0.0)
                self.publish_status(False)
            return

        if self.state == State.IDLE:
            if not self.gnss_valid:
                self.get_logger().warn("Cannot start: no GNSS fix received yet")
                return
            self.last_idx = 0
            self.last_mpc_time = -1e9
            self.last_torque_cmd = 0.0
            self.mpc.prev_z = None
            self.start_time = self.get_clock().now()
            self.state = State.FOLLOWING
            self.get_logger().info("Path following STARTED")
            self.publish_status(True)
        else:
            self.get_logger().info("Path following STOPPED by user")
            self.state = State.IDLE
            self.publish_cmd(0.0, 0.0)
            self.publish_status(False)

    def compute_errors(self, x, y, yaw, xr, yr, psir):
        # This matches your original nonlinear state convention:
        # e positive = left of path, psi_err = yaw - psi_ref.
        dx = x - xr
        dy = y - yr
        e = -math.sin(psir) * dx + math.cos(psir) * dy
        psi_err = wrap_angle(yaw - psir)
        return np.array([e, psi_err], dtype=float)


    def control_loop(self):
        if self.state == State.IDLE:
            return

        if not self.gnss_valid:
            self.publish_cmd(0.0, 0.0)
            return

        idx = closest_index_windowed(
            self.car_x, self.car_y, self.xref, self.yref,
            last_idx=self.last_idx,
            window=80,
        )
        idx = max(idx, self.last_idx)  # prevent backwards due to noise
        self.last_idx = idx

        progress = float(self.s_ref[min(idx, len(self.s_ref) - 1)])
        total = float(self.s_ref[-1])
        remaining = total - progress

        if remaining < float(self.get_parameter("stop_distance").value) and self.state == State.FOLLOWING:
            self.get_logger().info(f"Approaching path end ({remaining:.1f} m remaining). Slowing down...")
            self.state = State.STOPPING

        if self.state == State.STOPPING and self.car_speed_mps <= MIN_SPEED:
            self.get_logger().info("Path end reached. Returning to IDLE")
            self.state = State.IDLE
            self.publish_cmd(0.0, 0.0)
            self.publish_status(False)
            return

        xr = float(self.xref[idx])
        yr = float(self.yref[idx])
        psir = float(self.psiref[idx])
        path_err = self.compute_errors(self.car_x, self.car_y, self.car_yaw, xr, yr, psir)
        x0 = np.array([path_err[0], path_err[1], self.theta_sw_deg], dtype=float)

        # Build reference sequences for nonlinear MPC.
        kappa_seq = np.array([
            self.kappa_path[min(idx + k, len(self.kappa_path) - 1)]
            for k in range(self.horizon)
        ], dtype=float)

        # Use path speed profile, capped by desired_speed_mps if desired.
        desired_speed_cap = float(self.get_parameter("desired_speed_mps").value)
        v_seq = np.array([
            min(float(self.vref[min(idx + k, len(self.vref) - 1)]), desired_speed_cap)
            for k in range(self.horizon)
        ], dtype=float)
        desired_speed = 0.0 if self.state == State.STOPPING else float(v_seq[0])

        steer_a_seq = np.zeros(self.horizon, dtype=float)
        steer_b_seq = np.zeros(self.horizon, dtype=float)
        for k in range(self.horizon):
            steer_a_seq[k], steer_b_seq[k] = steering_ab_from_speed(v_seq[k], self.mpc_dt)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        solve_success = True
        runtime = 0.0

        if self.state == State.FOLLOWING and (now_s - self.last_mpc_time) >= self.mpc_dt:
            solve_success, torque_cmd_raw, runtime = self.mpc.step(
                x0,
                kappa_seq,
                v_seq,
                steer_a_seq,
                steer_b_seq,
            )
            self.last_mpc_time = now_s
            if solve_success:
                self.last_torque_cmd = torque_cmd_raw
            else:
                if not bool(self.get_parameter("hold_last_on_solver_fail").value):
                    self.last_torque_cmd = 0.0
                self.get_logger().warn("IPOPT solve failed. Holding previous torque or sending zero.")

        if self.state == State.FOLLOWING:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            ramp = min(1.0, elapsed / SOFT_START_DURATION)
            torque_cmd = ramp * self.last_torque_cmd
        else:
            torque_cmd = 0.0

        kp_speed = float(self.get_parameter("kp_speed").value)
        accel_cmd = max(-1.0, min(1.0, kp_speed * (desired_speed - self.car_speed_mps)))

        self.publish_cmd(accel_cmd, torque_cmd)
        self.publish_debug(path_err, idx, desired_speed, torque_cmd, progress)

        self.get_logger().debug(
            f"idx={idx} e={path_err[0]:+.3f} psi={math.degrees(path_err[1]):+.2f}deg "
            f"theta_sw={self.theta_sw_deg:+.2f}deg torque={torque_cmd:+.3f} "
            f"v={self.car_speed_mps:.2f} runtime={runtime*1000:.1f}ms success={solve_success}"
        )

    def publish_cmd(self, accel_cmd: float, torque_cmd: float):
        msg = Twist()
        msg.linear.x = float(max(-1.0, min(1.0, accel_cmd)))
        msg.angular.z = float(max(-1.0, min(1.0, torque_cmd)))
        self.cmd_pub.publish(msg)

    def publish_status(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.status_pub.publish(msg)

    def f64(self, v: float) -> Float64:
        msg = Float64()
        msg.data = float(v)
        return msg

    def publish_debug(self, x_err, idx, desired_speed, torque_cmd, progress):
        cte = float(x_err[0])
        psi = float(x_err[1])
        kappa = float(self.kappa_path[min(idx, len(self.kappa_path) - 1)])
        desired_delta = float(np.clip(math.atan(float(self.get_parameter("wheelbase").value) * kappa), -self.delta_max, self.delta_max))

        self.pub_cte.publish(self.f64(cte))
        self.pub_hdg_deg.publish(self.f64(math.degrees(psi)))
        self.pub_desired_delta.publish(self.f64(math.degrees(desired_delta)))
        self.pub_actual_delta.publish(self.f64(math.degrees(self.delta_meas)))
        self.pub_torque.publish(self.f64(torque_cmd))
        self.pub_progress.publish(self.f64(progress))
        self.pub_lateral_error.publish(self.f64(cte))
        self.pub_heading_error.publish(self.f64(psi))

        pf = Twist()
        pf.linear.x = float(desired_speed)
        pf.angular.z = float(desired_delta)
        self.pub_pf_cmd.publish(pf)

    def publish_path_visualization(self):
        msg = RosPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        step = max(1, len(self.xref) // 200)
        for i in range(0, len(self.xref), step):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(self.xref[i])
            ps.pose.position.y = float(self.yref[i])
            ps.pose.position.z = 0.0
            h = float(self.psiref[i])
            ps.pose.orientation.w = math.cos(h / 2.0)
            ps.pose.orientation.z = math.sin(h / 2.0)
            msg.poses.append(ps)

        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MpcV4IpoptCarNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
 

if __name__ == "__main__":
    main()
