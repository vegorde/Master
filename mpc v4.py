# mpc_run.py
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import json
from scipy.interpolate import CubicSpline, PchipInterpolator
import casadi as ca

HOST = "127.0.0.1"
PORT = 5555

# ── Edit this ─────────────────────────────────────────────────────────────────
# Set to a path.csv file to load a real path, or leave as "" to use the
# hardcoded waypoints below.
PATH_CSV = r"C:\Users\vegar\Documents\Isacsim\git\local\Master\Rosbag\path.csv"
# ─────────────────────────────────────────────────────────────────────────────


# ---------------- Vehicle constants (matched to Kia-rosnode_v2) ----------------
WHEELBASE       = 2.79
STEERING_RATIO  = 15.33
CONTROL_HZ      = 10.0
dt              = 1.0 / CONTROL_HZ
simtime         = 200.0

# MPC horizon and weights  (same as Kia-rosnode_v2 defaults)
N            = 50
delta_max    = math.radians(35.0)           # max front-wheel angle [rad]
theta_sw_max = math.degrees(delta_max * STEERING_RATIO)  # max steering-wheel angle [deg]

Qe       = 0.1 * 10    # cross-track error weight
Qpsi     = 0.1*1     # heading error weight
Rdelta   = 100 * 0.001       # Steering angle rate weight
Rtorque  = 0.01    # torque magnitude weight
Rdtorque = 0.1     # torque rate weight

nx, nu = 3, 1      # states: [e, psi_err, theta_sw_deg],  input: torque [-1, 1]

DEFAULT_SPEED_MPS = 15 / 3.6   # ~5.56 m/s


# ---------------- Speed-scheduled steering actuator (from Kia-rosnode_v2) ----
#   theta_sw[k+1] = a(v)*theta_sw[k] + b(v)*torque[k]
SCHED_V_KMH  = np.array([5.,10.,15.,20.,25.,30.,36.,40.,45.,50.], dtype=float)
SCHED_TAU_R  = np.array([2.953,1.386,0.722,0.627,0.437,0.390,0.342,0.295,0.247,0.200], dtype=float)
SCHED_KSS    = np.array([501.9,314.9,187.4,143.2,105.5,84.6,71.2,63.3,53.9,44.1], dtype=float)

def steering_ab(v_mps):
    """Return discrete actuator coefficients a, b for the current speed."""
    v_kmh = float(v_mps) * 3.6
    tau = float(np.interp(v_kmh, SCHED_V_KMH, SCHED_TAU_R))
    kss = float(np.interp(v_kmh, SCHED_V_KMH, SCHED_KSS))
    a = math.exp(-dt / max(tau, 1e-3))
    b = kss * (1.0 - a)
    return a, b


# ---------------- Helpers ─────────────────────────────────────────────────────
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def smooth(x, w=7):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")

def compute_errors(x, y, yaw, xr, yr, psir):
    dx = x - xr;  dy = y - yr
    e       = -math.sin(psir) * dx + math.cos(psir) * dy
    psi_err = wrap_angle(yaw - psir)
    return np.array([e, psi_err], dtype=float)

def closest_index_windowed(x, y, xref, yref, last_idx=0, window=100):
    n  = len(xref)
    i0 = max(0, last_idx - window)
    i1 = min(n, last_idx + window + 1)
    dx = xref[i0:i1] - x;  dy = yref[i0:i1] - y
    return i0 + int(np.argmin(dx*dx + dy*dy))

def recv_line(conn, buf):
    try:
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return None, buf
            buf += chunk
        line, buf = buf.split(b"\n", 1)
        return line, buf
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
        print(f"[MPC SERVER] Client disconnected during recv: {e}")
        return None, buf


# ---------------- Path loading ────────────────────────────────────────────────
def load_waypoints_csv(filename, default_speed=DEFAULT_SPEED_MPS):
    waypoints = []
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
            x = float(parts[0]);  y = float(parts[1])
            v = float(parts[2]) if len(parts) >= 3 else default_speed
            waypoints.append((x, y, v))
    if len(waypoints) < 2:
        raise ValueError(f"CSV '{filename}' has fewer than 2 valid waypoints")
    return waypoints

def compute_heading_and_curvature_from_spline(waypoints, ds, kind_xy="cubic", kind_v="pchip"):
    wp  = np.asarray(waypoints, dtype=float)
    xw, yw, vw = wp[:, 0], wp[:, 1], wp[:, 2]

    dx = np.diff(xw);  dy = np.diff(yw)
    seg  = np.hypot(dx, dy)
    keep = np.ones(len(wp), dtype=bool);  keep[1:] = seg > 1e-9
    xw, yw, vw = xw[keep], yw[keep], vw[keep]

    dx = np.diff(xw);  dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    n     = max(1, int(np.floor(total / ds)))
    s_ref = np.linspace(0.0, total, n + 1)

    def make_interp(kind, t, z):
        if kind == "cubic":  return CubicSpline(t, z, bc_type="natural")
        if kind == "pchip":  return PchipInterpolator(t, z)
        raise ValueError(f"Unknown kind '{kind}'")

    fx = make_interp(kind_xy, s, xw)
    fy = make_interp(kind_xy, s, yw)
    fv = make_interp(kind_v,  s, vw)

    xref = np.asarray(fx(s_ref));  yref = np.asarray(fy(s_ref))
    vref = np.asarray(fv(s_ref))

    x_s  = fx.derivative(1)(s_ref);  y_s  = fy.derivative(1)(s_ref)
    x_ss = fx.derivative(2)(s_ref);  y_ss = fy.derivative(2)(s_ref)

    psiref = np.unwrap(np.arctan2(y_s, x_s))
    denom  = np.maximum((x_s**2 + y_s**2)**1.5, 1e-12)
    kappa_points = (x_s * y_ss - y_s * x_ss) / denom
    kappa_path   = smooth(0.5 * (kappa_points[:-1] + kappa_points[1:]), w=9)

    return xref, yref, psiref, vref, kappa_path, s_ref

def build_ref(waypoints, ds=0.5):
    return compute_heading_and_curvature_from_spline(
        waypoints, ds, kind_xy="cubic", kind_v="pchip"
    )


# ---------------- NLP (built once) ────────────────────────────────────────────
def build_mpc_nlp():
    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)

    x0_p      = ca.SX.sym("x0",      nx)
    kappa_p   = ca.SX.sym("kappa",   N)
    v_p       = ca.SX.sym("v",       N)
    steer_a_p = ca.SX.sym("steer_a", N)
    steer_b_p = ca.SX.sym("steer_b", N)

    def f_step(xk, uk, kappak, vk, steer_a, steer_b):
        e            = xk[0]
        psi          = xk[1]
        theta_sw_deg = xk[2]
        torque       = uk[0]

        # Steering-wheel angle → front-wheel angle [rad]
        delta = (theta_sw_deg * math.pi / 180.0) / STEERING_RATIO

        denom = 1.0 - kappak * e
        denom = ca.if_else(ca.fabs(denom) < 1e-3, ca.sign(denom) * 1e-3, denom)

        s_dot = vk * ca.cos(psi) / denom
        e_dot   = vk * ca.sin(psi)
        psi_dot = (vk / WHEELBASE) * ca.tan(delta) - s_dot * kappak

        e_next     = e   + dt * e_dot
        psi_next   = psi + dt * psi_dot
        theta_next = steer_a * theta_sw_deg + steer_b * torque   # actuator model

        return ca.vertcat(e_next, psi_next, theta_next)

    obj = 0
    g   = [X[:, 0] - x0_p]

    for k in range(N):
        g.append(X[:, k + 1] - f_step(
            X[:, k], U[:, k], kappa_p[k], v_p[k], steer_a_p[k], steer_b_p[k]
        ))
        obj += Qe      * (X[0, k] ** 2)
        obj += Qpsi    * (X[1, k] ** 2)
        obj += Rtorque * ca.sumsqr(U[:, k])
        if k > 0:
            obj += Rdtorque * ca.sumsqr(U[:, k] - U[:, k - 1])
            # Rate of actual front-wheel angle: delta = theta_sw_deg * pi/180 / STEERING_RATIO
            d_delta = (X[2, k] - X[2, k - 1]) * (math.pi / 180.0 / STEERING_RATIO)
            obj += Rdelta * d_delta ** 2

    g = ca.vertcat(*g)
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    p = ca.vertcat(x0_p, kappa_p, v_p, steer_a_p, steer_b_p)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 200,
        "ipopt.tol": 1e-5,
        "ipopt.acceptable_tol": 1e-4,
    }
    solver = ca.nlpsol("solver", "ipopt", {"x": z, "f": obj, "g": g, "p": p}, opts)

    ng  = g.size1()
    lbg = np.zeros(ng);  ubg = np.zeros(ng)

    nX = nx * (N + 1);  nU = nu * N;  nz = nX + nU
    lbz = -np.inf * np.ones(nz);  ubz = np.inf * np.ones(nz)

    # Bound steering-wheel angle state
    for k in range(N + 1):
        idx = k * nx + 2
        lbz[idx] = -theta_sw_max
        ubz[idx] =  theta_sw_max

    # Bound torque input
    for k in range(N):
        idx = nX + k * nu
        lbz[idx] = -1.0
        ubz[idx] =  1.0

    def unpack(z_val):
        z_val = np.asarray(z_val).reshape(-1)
        Xv = z_val[:nX].reshape((nx, N + 1), order="F")
        Uv = z_val[nX:].reshape((nu, N), order="F")
        return Xv, Uv

    return solver, lbg, ubg, lbz, ubz, unpack

print("[MPC] Building NLP solver (once)...")
_solver, _lbg, _ubg, _lbz, _ubz, _unpack = build_mpc_nlp()
_prev_z = None

def mpc_step(x0, kappa_seq, v_seq, steer_a_seq, steer_b_seq):
    """Solve one MPC step. Returns (success, torque_cmd)."""
    global _prev_z
    p = np.concatenate([
        np.asarray(x0).reshape(-1),
        np.asarray(kappa_seq).reshape(-1),
        np.asarray(v_seq).reshape(-1),
        np.asarray(steer_a_seq).reshape(-1),
        np.asarray(steer_b_seq).reshape(-1),
    ])

    if _prev_z is None:
        x_guess = np.zeros((nx, N + 1));  x_guess[:, 0] = x0
        u_guess = np.zeros((nu, N))
        z0 = np.concatenate([x_guess.reshape(-1, order="F"), u_guess.reshape(-1, order="F")])
    else:
        z0 = _prev_z

    try:
        sol = _solver(x0=z0, p=p, lbg=_lbg, ubg=_ubg, lbx=_lbz, ubx=_ubz)
    except RuntimeError:
        return False, 0.0

    z_opt = np.array(sol["x"]).reshape(-1)
    _, U_opt = _unpack(z_opt)

    if not np.isfinite(U_opt[:, 0]).all():
        return False, 0.0

    _prev_z = z_opt
    torque_cmd = float(np.clip(U_opt[:, 0].item(), -1.0, 1.0))
    return True, torque_cmd


# ---------------- Main (TCP server) ───────────────────────────────────────────
if __name__ == "__main__":
    # ---- Load reference path ----
    if PATH_CSV:
        print(f"[MPC] Loading path from {PATH_CSV}")
        waypoints = load_waypoints_csv(PATH_CSV)
    else:
        # Fallback hardcoded path
        waypoints = [
            (0.,0.,5.),(5.,0.,5.),(10.,0.,5.),(15.,0.,5.),(20.,0.,5.),
            (25.,0.,5.),(30.,0.,5.),(35.,0.,0.23,),(40.,0.5,5.),(50.,1.,5.),
            (80.,2.3,5.),(100.,2.3,5.),(130.,2.3,5.),(150.,2.3,5.),
        ]

    xref, yref, psiref, vref, kappa_path, s_ref = build_ref(waypoints, ds=0.5)
    print(f"[MPC] Path ready: {len(xref)} points, {s_ref[-1]:.0f} m total")

    # ---- Logging buffers ----
    ts, xs, ys      = [], [], []
    es, psis        = [], []
    torques         = []
    vels, reqv      = [], []
    delta_measured  = []   # actual front-wheel angle from sim [rad]
    delta_target    = []   # front-wheel angle commanded by MPC [rad]
    largest_runtime = 0.0

    last_mpc_t  = -1e9
    last_torque = 0.0
    last_idx    = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[MPC SERVER] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[MPC SERVER] Connected from {addr}")
            buf = b""

            while True:
                line, buf = recv_line(conn, buf)
                if line is None:
                    print("[MPC SERVER] Connection closed")
                    break

                msg = json.loads(line.decode("utf-8"))
                t_sim = float(msg["t"])
                x     = float(msg["x"])
                y     = float(msg["y"])
                yaw   = float(msg["yaw"])
                vel   = float(msg["v"])
                # delta from sim is front-wheel angle [rad]; convert to steering-wheel [deg]
                delta_rad    = float(msg["delta"])
                theta_sw_deg = math.degrees(delta_rad * STEERING_RATIO)

                # ---- Closest path point ----
                idx      = closest_index_windowed(x, y, xref, yref, last_idx=last_idx, window=80)
                idx      = max(idx, last_idx)   # don't step backwards
                last_idx = idx

                # ---- Error state ----
                x_err = compute_errors(x, y, yaw, xref[idx], yref[idx], psiref[idx])
                x0    = np.array([x_err[0], x_err[1], theta_sw_deg], dtype=float)

                # ---- Reference sequences ----
                kappa_seq   = [kappa_path[min(idx + k, len(kappa_path) - 1)] for k in range(N)]
                v_seq       = [float(vref[min(idx + k, len(vref) - 1)])      for k in range(N)]
                steer_a_seq = [steering_ab(v_seq[k])[0] for k in range(N)]
                steer_b_seq = [steering_ab(v_seq[k])[1] for k in range(N)]

                # ---- MPC solve (rate-limited to dt) ----
                if t_sim - last_mpc_t >= dt:
                    t0 = time.time()
                    ok, torque_cmd = mpc_step(x0, kappa_seq, v_seq, steer_a_seq, steer_b_seq)
                    runtime = time.time() - t0
                    largest_runtime = max(largest_runtime, runtime)
                    last_mpc_t = t_sim
                    if ok:
                        last_torque = torque_cmd
                    else:
                        print("[MPC] IPOPT failed — holding previous torque")

                # Convert torque → front-wheel angle for the sim via actuator steady-state.
                v_now = max(vel, 0.5)
                a, b  = steering_ab(v_now)
                kss   = b / max(1.0 - a, 1e-6)
                theta_sw_cmd = float(np.clip(kss * last_torque, -theta_sw_max, theta_sw_max))
                delta_cmd    = float(np.clip(
                    math.radians(theta_sw_cmd) / STEERING_RATIO,
                    -delta_max, delta_max
                ))

                reply = {
                    "torque": last_torque,   # normalized torque (Kia interface)
                    "delta":  delta_cmd,     # equivalent front-wheel angle [rad] (Isaac interface)
                    "v":      float(vref[min(idx, len(vref) - 1)]),
                }
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
                    print(f"[MPC SERVER] Send error: {e}")
                    break

                # ---- Log ----
                ts.append(t_sim);  xs.append(x);  ys.append(y)
                es.append(x_err[0]);  psis.append(x_err[1])
                torques.append(last_torque)
                vels.append(vel);  reqv.append(float(vref[min(idx, len(vref) - 1)]))
                delta_measured.append(delta_rad)
                delta_target.append(delta_cmd)

                if len(ts) % 100 == 0:
                    print(
                        f"[MPC] t={t_sim:5.2f}  e={x_err[0]:+.3f}m  "
                        f"psi={math.degrees(x_err[1]):+.1f}deg  "
                        f"torque={last_torque:+.3f}  v={vel:.2f}m/s"
                    )

    # ---- Performance metrics ----
    es_a  = np.asarray(es);  tor_a = np.asarray(torques)
    print("\n========= PERFORMANCE METRICS =========")
    print(f"Weights: Qe={Qe}, Qpsi={Qpsi}, Rtorque={Rtorque}, Rdtorque={Rdtorque}")
    print(f"CTE  — mean={np.mean(es_a**2):.4e}  max={np.max(np.abs(es_a)):.3f} m")
    print(f"Torque saturations: {np.sum(np.abs(tor_a) > 0.99)} / {len(tor_a)} steps")
    print(f"Largest solve time: {largest_runtime*1000:.1f} ms")
    print("=======================================\n")

    # ---- Plots ----
    plt.figure()
    plt.plot(xref, yref, "k--", label="Reference path")
    plt.plot(xs, ys, label="Driven path")
    plt.xlabel("x [m]");  plt.ylabel("y [m]")
    plt.title("Path tracking");  plt.legend();  plt.grid(True)

    plt.figure()
    plt.plot(ts, es, label="CTE [m]")
    plt.plot(ts, np.degrees(psis), label="Heading error [deg]")
    plt.xlabel("time [s]");  plt.title("Tracking errors")
    plt.legend();  plt.grid(True)

    plt.figure()
    plt.plot(ts, torques, label="Torque cmd")
    plt.axhline( 1.0, color="r", linestyle="--", alpha=0.4)
    plt.axhline(-1.0, color="r", linestyle="--", alpha=0.4)
    plt.xlabel("time [s]");  plt.ylabel("Normalized torque [-1,1]")
    plt.title("Steering torque command");  plt.legend();  plt.grid(True)

    plt.figure()
    plt.plot(ts, np.degrees(delta_target),  label="Target δ (MPC cmd)")
    plt.plot(ts, np.degrees(delta_measured), label="Actual δ (sim)", linestyle="--")
    plt.xlabel("time [s]");  plt.ylabel("Front-wheel angle [deg]")
    plt.title("Steering angle: target vs actual");  plt.legend();  plt.grid(True)

    plt.figure()
    plt.plot(ts, vels, label="Actual speed")
    plt.plot(ts, reqv, label="Reference speed")
    plt.xlabel("time [s]");  plt.ylabel("Speed [m/s]")
    plt.title("Speed");  plt.legend();  plt.grid(True)

    plt.show()
