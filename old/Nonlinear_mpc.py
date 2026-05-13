# mpc_run.py
import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
# Server tcp setup
import socket
import json
from scipy.interpolate import CubicSpline, PchipInterpolator

HOST = "127.0.0.1"
PORT = 5555

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





# ---------------- MPC setup (yours) ----------------
simtime = 20.0 
dt, v, L = 0.1, 1, 0.32
N = 50
capang = dt * 3.14 * 50
A = np.array([[1.0, dt * v],
              [0.0, 1.0]])

B = np.array([[0.0],
              [dt * v / L]])

nx, nu = 2, 1

Qe, Qpsi = 1, 1
Rdelta, Rdd = 1, 1

delta_max = np.deg2rad(35)
ddelta_max = np.deg2rad(60) * dt

import numpy as np
import casadi as ca

# Assumes you already have:
# nx, nu, N
# dt, v
# A, B  (can still use these inside f_step if you want)
# delta_max, ddelta_max
# Qe, Qpsi, Rdelta, Rdd

def build_mpc_nlp():
    """
    Build the NLP once, return a callable solver + helper pack/unpack functions.
    """
    # Decision variables
    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)

    # Parameters (things that change each solve)
    x0_p = ca.SX.sym("x0", nx)          # initial state
    kappa_p = ca.SX.sym("kappa", N)     # curvature sequence

    def f_step(xk, uk, kappak):
        """
        Nonlinear error dynamics (continuous -> discrete with forward Euler).

        State x = [e, psi]
        Input u = [delta]
        Parameter kappak = curvature at step k
        """
        e   = xk[0]
        psi = xk[1]
        delta = uk[0]
        s_dot = v * ca.cos(psi) / (1 - kappak * e)

        e_dot   = v * ca.sin(psi)
        psi_dot = (v / L) * ca.tan(delta) - s_dot * kappak

        e_next   = e   + dt * e_dot
        psi_next = psi + dt * psi_dot

        return ca.vertcat(e_next, psi_next)


    # Objective and constraints
    obj = 0
    g = []

    # Initial condition constraint
    g.append(X[:, 0] - x0_p)

    for k in range(N):
        # Dynamics constraint
        x_next = f_step(X[:, k], U[:, k], kappa_p[k])
        g.append(X[:, k + 1] - x_next)

        # Stage cost
        
        obj += Qe   * (X[0, k] ** 2)
        obj += Qpsi * (X[1, k] ** 2)
        obj += Rdelta * ca.sumsqr(U[:, k])
        """
        obj += Qe   * (X[0, k])
        obj += Qpsi * (X[1, k])
        obj += Rdelta * ca.sumsqr(U[:, k])
        """
        # Rate cost (and later rate constraint)
        if k > 0:
            du = U[:, k] - U[:, k - 1]
            obj += Rdd * ca.sumsqr(du)

    # Stack constraints into a single vector
    g = ca.vertcat(*g)

    # Flatten decision vars
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    p = ca.vertcat(x0_p, kappa_p)

    nlp = {"x": z, "f": obj, "g": g, "p": p}

    # IPOPT options (tweak as needed)
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 20000,
        "ipopt.tol": 1e-6,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ---- Bounds ----
    # g bounds: all equality constraints => g == 0
    ng = g.size1()
    lbg = np.zeros(ng)
    ubg = np.zeros(ng)

    # z bounds: bounds on X are free (±inf) unless you want state constraints
    nX = nx * (N + 1)
    nU = nu * N
    nz = nX + nU

    lbz = -np.inf * np.ones(nz)
    ubz =  np.inf * np.ones(nz)

    # Input bounds: |u_k| <= delta_max
    # U starts at index nX in z
    for k in range(N):
        for i in range(nu):
            idx = nX + k * nu + i
            lbz[idx] = -delta_max
            ubz[idx] =  delta_max

    # Rate constraints: |u_k - u_{k-1}| <= ddelta_max  for k>=1
    # Implement as additional constraints by extending g (cleanest),
    # BUT we already built g. Easiest now: add them as *extra g* in the NLP build
    # if you need hard constraints.
    #
    # If you DO need hard rate constraints, see the version below (add_rate_constraints=True).

    # Helpers to unpack
    def unpack(z_val):
        z_val = np.asarray(z_val).reshape(-1)
        Xv = z_val[:nX].reshape((nx, N + 1), order="F")
        Uv = z_val[nX:].reshape((nu, N), order="F")
        return Xv, Uv

    return solver, lbg, ubg, lbz, ubz, unpack
# Build once (important for speed)
_solver, _lbg, _ubg, _lbz, _ubz, _unpack = build_mpc_nlp()
_prev_z = None  # for warm-start initial guess

def mpc_step_nonlinear(x0, kappa_seq):
    global _prev_z

    x0 = np.asarray(x0).reshape(-1)
    kappa_seq = np.asarray(kappa_seq).reshape(-1)
    assert kappa_seq.size == N

    p = np.concatenate([x0, kappa_seq])

    # Initial guess (warm start): either previous solution shifted, or zeros
    if _prev_z is None:
        x_guess = np.zeros((nx, N + 1))
        u_guess = np.zeros((nu, N))
        # set initial state guess
        x_guess[:, 0] = x0
        z0 = np.concatenate([x_guess.reshape(-1, order="F"), u_guess.reshape(-1, order="F")])
    else:
        z0 = _prev_z

    try:
        sol = _solver(x0=z0, p=p, lbg=_lbg, ubg=_ubg, lbx=_lbz, ubx=_ubz)
    except RuntimeError:
        return 0.0

    z_opt = np.array(sol["x"]).reshape(-1)
    _prev_z = z_opt  # save for warm-start next call

    X_opt, U_opt = _unpack(z_opt)

    if not np.isfinite(U_opt[:, 0]).all():
        return 0.0
    return float(U_opt[:, 0].item())


def closest_index_windowed(x, y, xref, yref, last_idx=0, window=100):
    n = len(xref)
    i0 = max(0, last_idx - window)
    i1 = min(n, last_idx + window + 1)

    dx = xref[i0:i1] - x
    dy = yref[i0:i1] - y
    d2 = dx*dx + dy*dy

    return i0 + int(np.argmin(d2))


def resample_waypoints_xyv(waypoints, ds, kind_xy="cubic", kind_v="pchip"):
    """
    Spline-resample waypoints into a smooth path with ~constant spatial spacing ds.

    waypoints: list of (x, y, v)
    ds: desired arc-length step [m]

    kind_xy:
      - "cubic"  -> CubicSpline for x(s), y(s)  (smooth, can overshoot on sharp corners)
      - "pchip"  -> PchipInterpolator for x(s), y(s) (shape-preserving, less overshoot)

    kind_v:
      - "pchip"  -> recommended for speed profile (avoids overshoot)
      - "cubic"  -> smoother but can overshoot

    Returns: xref, yref, vref sampled at uniform s
    """
    wp = np.asarray(waypoints, dtype=float)
    assert wp.shape[1] == 3 and wp.shape[0] >= 2, "Need at least 2 waypoints of (x,y,v)."

    xw, yw, vw = wp[:, 0], wp[:, 1], wp[:, 2]

    # Build cumulative chord-length parameter s
    dx = np.diff(xw)
    dy = np.diff(yw)
    seg = np.hypot(dx, dy)

    # Remove duplicate consecutive points (seg == 0) to keep s strictly increasing
    keep = np.ones(len(wp), dtype=bool)
    keep[1:] = seg > 1e-9
    xw, yw, vw = xw[keep], yw[keep], vw[keep]
    assert len(xw) >= 2, "All waypoints collapsed or duplicates."

    dx = np.diff(xw)
    dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        # Degenerate: everything is same point
        return np.array([xw[0]]), np.array([yw[0]]), np.array([vw[0]])

    # Sample s uniformly
    n = int(np.floor(total / ds))
    s_ref = np.linspace(0.0, total, n + 1)

    # Choose spline types
    def make_interp(kind, t, z):
        if kind == "cubic":
            return CubicSpline(t, z, bc_type="natural")
        elif kind == "pchip":
            return PchipInterpolator(t, z)
        else:
            raise ValueError(f"Unknown kind '{kind}' (use 'cubic' or 'pchip').")

    fx = make_interp(kind_xy, s, xw)
    fy = make_interp(kind_xy, s, yw)
    fv = make_interp(kind_v,  s, vw)

    xref = fx(s_ref)
    yref = fy(s_ref)
    vref = fv(s_ref)

    return np.asarray(xref), np.asarray(yref), np.asarray(vref)


def compute_heading_and_curvature_from_spline(waypoints, ds, kind_xy="cubic", kind_v="pchip"):
    """
    One-shot: spline resample + heading + curvature using spline derivatives.
    Returns xref, yref, psiref, vref, kappa_path
    """
    wp = np.asarray(waypoints, dtype=float)
    xw, yw, vw = wp[:, 0], wp[:, 1], wp[:, 2]

    # Build cumulative chord-length parameter s
    dx = np.diff(xw); dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    keep = np.ones(len(wp), dtype=bool)
    keep[1:] = seg > 1e-9
    xw, yw, vw = xw[keep], yw[keep], vw[keep]

    dx = np.diff(xw); dy = np.diff(yw)
    seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    n = int(np.floor(total / ds))
    s_ref = np.linspace(0.0, total, n + 1)

    from scipy.interpolate import CubicSpline, PchipInterpolator

    def make_interp(kind, t, z):
        if kind == "cubic":
            return CubicSpline(t, z, bc_type="natural")
        elif kind == "pchip":
            return PchipInterpolator(t, z)
        else:
            raise ValueError(f"Unknown kind '{kind}'")

    fx = make_interp(kind_xy, s, xw)
    fy = make_interp(kind_xy, s, yw)
    fv = make_interp(kind_v,  s, vw)

    xref = fx(s_ref)
    yref = fy(s_ref)
    vref = fv(s_ref)

    # First and second derivatives wrt s
    x_s  = fx.derivative(1)(s_ref)
    y_s  = fy.derivative(1)(s_ref)
    x_ss = fx.derivative(2)(s_ref)
    y_ss = fy.derivative(2)(s_ref)

    # Heading at points
    psiref = np.unwrap(np.arctan2(y_s, x_s))

    # Curvature at points (length M)
    denom = (x_s**2 + y_s**2)**1.5
    denom = np.maximum(denom, 1e-12)
    kappa_points = (x_s * y_ss - y_s * x_ss) / denom

    # Your MPC expects curvature per segment (length M-1).
    # Take segment curvature as the average of adjacent point curvatures.
    kappa_path = 0.5 * (kappa_points[:-1] + kappa_points[1:])

    return np.asarray(xref), np.asarray(yref), np.asarray(psiref), np.asarray(vref), np.asarray(kappa_path)


def build_ref_from_waypoints_xyv(waypoints, ds):
    # Best quality: compute heading+curvature from spline derivatives
    xref, yref, psiref, vref, kappa_path = compute_heading_and_curvature_from_spline(
        waypoints, ds, kind_xy="cubic", kind_v="pchip"
    )

    kappa_path = smooth(kappa_path, w=9)

    return xref, yref, psiref, vref, kappa_path

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def smooth(x, w=7):
    """
    Simple moving average smoothing.
    w = window size (odd numbers like 5,7,9 work best)
    """
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

def compute_errors(x, y, yaw, xr, yr, psir):
    dx = x - xr
    dy = y - yr
    e = -np.sin(psir) * dx + np.cos(psir) * dy
    psi_err = wrap_angle(yaw - psir)
    return np.array([e, psi_err], dtype=float)

# ---------------- Run closed-loop ----------------
last_mpc_t = -1e9               # last time (sim time) MPC updated
largest_sampletime = 0
if __name__ == "__main__":
    print("[MPC SERVER] Starting...")
    # ---------- logging buffers ----------
    xs, ys = [], []
    deltas, vels, reqv = [], [], []
    es, psis = [], []
    ts = []

    # --- Reference defined as waypoints (x, y, v) ---
    waypoints = [
        (0.000, 0.000, 1.000),
        (2.000, 0.000, 1.000),
        (4.000, 0.000, 1.000),
        (6.000, 0.000, 1.000),
        (8.000, 0.000, 1.000),
        (10.000, 0.000, 0.100),
        (10.000, 2.000, 1.000),
        (10.000, 4.000, 1.000),
        (10.000, 6.000, 1.000),
        (10.000, 8.000, 1.000),
        (10.000, 10.000, 1.000),
    ]

    ds = v * dt  # fixed spatial spacing that matches your internal model
    xref, yref, psiref, vref, kappa_path = build_ref_from_waypoints_xyv(waypoints, ds)

    # Optional: ensure the path is long enough for simtime + horizon indexing
    # (so min(idx+k) doesn't clamp too early)
    min_points_needed = int(np.ceil(simtime / dt)) + N + 5
    if len(xref) < min_points_needed and False:
        # If your waypoints are short, extend last point forward along last heading:
        extra = min_points_needed - len(xref)
        last_psi = psiref[-1]
        x_ext = xref[-1] + ds * np.cos(last_psi) * np.arange(1, extra + 1)
        y_ext = yref[-1] + ds * np.sin(last_psi) * np.arange(1, extra + 1)
        v_ext = np.full(extra, vref[-1])
        xref = np.concatenate([xref, x_ext])
        yref = np.concatenate([yref, y_ext])
        vref = np.concatenate([vref, v_ext])
        # headings + curvature for the extension
        psiref, kappa_path = compute_heading_and_curvature_from_spline(xref, yref)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[MPC SERVER] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[MPC SERVER] Connected from {addr}")
            buf = b""
            
            idx = 0
            last_idx = 0
            while True:
                line, buf = recv_line(conn, buf)
                if line is None:
                    print("[MPC SERVER] Connection closed")
                    break

                msg = json.loads(line.decode("utf-8"))

                # ---------- Isaac state ----------
                t = float(msg["t"])
                x = float(msg["x"])
                y = float(msg["y"])
                yaw = float(msg["yaw"])
                vel = float(msg["v"])

                # ---------- Reference index ----------
                #idx = min(int(t / dt), len(xref) - 1)
                # --- Closest point on reference path ---
                # Keep idx from last iteration (define idx before the loop starts)
                idx = closest_index_windowed(x, y, xref, yref, last_idx=idx, window=80)
                #idx = min(int(t / dt), len(xref) - 1)
                # Optional: prevent going backwards due to noise
                idx = max(idx, last_idx)
                last_idx = idx

                # ---------- MPC error state ----------
                x_err = compute_errors(
                    x, y, yaw,
                    xref[idx], yref[idx], psiref[idx]
                )

                # ---------- MPC solve ----------
                kappa_seq = [
                    kappa_path[min(idx + k, len(kappa_path) - 1)]
                    for k in range(N)
                ]
                if t - last_mpc_t >= dt:
                    startime = time.time()
                    delta_cmd = mpc_step_nonlinear(x_err, kappa_seq)
                    runtime = time.time() - startime
                    if runtime > largest_sampletime:
                        largest_sampletime = runtime
                    delta_cmd = float(np.clip(delta_cmd, -delta_max, delta_max))
                    last_mpc_t = t
                    print(f"NEW MPC:   {
                        f"[MPC] t={t:5.2f} "
                        f"x={x:+.2f} y={y:+.2f} "
                        f"e={x_err[0]:+.2f} "
                        f"delta={delta_cmd:+.3f}"
                        f"runtime={runtime}"
                    }")

                # ---------- Send command back ----------
                reply = {"delta": delta_cmd, "v": float(vref[idx])}
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
                    print(f"[MPC SERVER] Client disconnected during send: {e}")
                    break


                # ---------- Log data ----------
                ts.append(t)
                xs.append(x)
                ys.append(y)
                es.append(x_err[0])
                psis.append(x_err[1])
                deltas.append(delta_cmd)
                vels.append(vel)
                reqv.append(float(vref[idx]))
                if len(xs) % 100 == 0:
                    print(
                        f"[MPC] t={t:5.2f} "
                        f"x={x:+.2f} y={y:+.2f} "
                        f"e={x_err[0]:+.2f} "
                        f"delta={delta_cmd:+.3f}"
                    )

    # ===================== PLOTS =====================
    # ===================== PERFORMANCE METRICS =====================

    # Convert to numpy arrays (safe even if already lists)
    es_arr = np.asarray(es)
    psis_arr = np.asarray(psis)
    deltas_arr = np.asarray(deltas)

    # Steering rate (rate-normalized)
    steer_rate = np.diff(deltas_arr) / dt

    # ---- Cross-track error statistics ----
    mse_e_mean = np.mean(es_arr**2)
    mse_e_p90  = np.percentile(es_arr**2, 90)
    mse_e_max  = np.max(es_arr**2)

    # ---- Heading error statistics ----
    mse_psi_mean = np.mean(psis_arr**2)
    mse_psi_p90  = np.percentile(psis_arr**2, 90)
    mse_psi_max  = np.max(psis_arr**2)

    # ---- Steering rate statistics ----
    j_deldot_mean = np.mean(steer_rate**2)
    j_deldot_p90  = np.percentile(steer_rate**2, 90)
    j_deldot_max  = np.max(steer_rate**2)

    print("\n================ PERFORMANCE METRICS ================")
    print(f"Qe, Qpsi, Rd, Rdd = {Qe}, {Qpsi}, {Rdelta}, {Rdd}")
    print("Cross-track error e:")
    print(f"  Mean (MSE_e)       : {mse_e_mean:.6e} [m^2]")
    print(f"  90th percentile   : {mse_e_p90:.6e} [m^2]")
    print(f"  Max               : {mse_e_max:.6e} [m^2]")

    print("\nHeading error ψ:")
    print(f"  Mean (MSE_psi)     : {mse_psi_mean:.6e} [rad^2]")
    print(f"  90th percentile   : {mse_psi_p90:.6e} [rad^2]")
    print(f"  Max               : {mse_psi_max:.6e} [rad^2]")

    print("\nSteering rate:")
    print(f"  Mean (J_deldot)    : {j_deldot_mean:.6e} [rad^2/s^2]")
    print(f"  90th percentile   : {j_deldot_p90:.6e} [rad^2/s^2]")
    print(f"  Max               : {j_deldot_max:.6e} [rad^2/s^2]")
    print("====================================================\n")
    print(f" Largest sampletime {largest_sampletime}")


    plt.figure()
    plt.plot(xref, yref, "--", label="Reference path")
    plt.plot(xs, ys, label="Driven path")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.title("Path tracking")
    plt.grid(True)

    plt.figure()
    plt.plot(ts, es, label="Cross-track error [m]")
    plt.plot(ts, psis, label="Heading error ψ [rad]")
    plt.xlabel("time [s]")
    plt.ylabel("Error")
    plt.title("Tracking errors")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(ts, np.rad2deg(deltas))
    plt.xlabel("time [s]")
    plt.ylabel("Steering command δ [deg]")
    plt.title("Steering command")
    plt.grid(True)

    plt.figure()
    plt.plot(ts, vels, label="Velocity")
    plt.plot(ts, reqv, label="Requested Velocity")
    plt.xlabel("time [s]")
    plt.ylabel("Linear velocity [m/s]")
    plt.title("Linear velocity")
    plt.legend()
    plt.grid(True)
    
    plt.show()
    








