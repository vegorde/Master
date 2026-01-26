# mpc_run.py
import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
# Server tcp setup
import socket
import json

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
N = 10
capang = dt * 3.14 * 50
A = np.array([[1.0, dt * v],
              [0.0, 1.0]])

B = np.array([[0.0],
              [dt * v / L]])

nx, nu = 2, 1

Qe, Qpsi = 150, 1
Rdelta, Rdd = 1, 5
alpha_e = 50
p_e = 1

alpha_psi = 1
p_psi = 2

delta_max = np.deg2rad(35)
ddelta_max = np.deg2rad(60) * dt
Kmax = 0.5
def mpc_step(x0, kappa_seq):
    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))

    obj = 0
    cons = [x[:, 0] == x0]
    Qe_seq = [Qe / (1 + alpha_e * (abs(kappa_seq[_])/Kmax)) for _ in range(N)]
    Qpsi_seq = [Qpsi * (1 + alpha_psi * (abs(kappa_seq[_])/Kmax)) for _ in range(N)]
    for k in range(N):
        d = np.array([0.0, -dt * v * kappa_seq[k]])
        cons += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + d]
        cons += [cp.abs(u[:, k]) <= delta_max]

        if k > 0:
            cons += [cp.abs(u[:, k] - u[:, k - 1]) <= ddelta_max]

        obj += Qe_seq[k] * cp.square(x[0, k]) + Qpsi_seq[k] * cp.square(x[1, k]) + Rdelta * cp.square(u[:, k])
        if k > 0:
            obj += Rdd * cp.square(u[:, k] - u[:, k - 1])

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-4, eps_rel=1e-4)

    if u.value is None:
        return 0.0
    return float(u.value[:, 0].item())

def closest_index_windowed(x, y, xref, yref, last_idx=0, window=100):
    n = len(xref)
    i0 = max(0, last_idx - window)
    i1 = min(n, last_idx + window + 1)

    dx = xref[i0:i1] - x
    dy = yref[i0:i1] - y
    d2 = dx*dx + dy*dy

    return i0 + int(np.argmin(d2))


def build_ref_from_kappa(kappa_path, x0=0.0, y0=0.0, psi0=0.0, v=v, dt=dt):
    ds = v * dt
    Np = len(kappa_path)
    xref = np.zeros(Np + 1)
    yref = np.zeros(Np + 1)
    psiref = np.zeros(Np + 1)
    xref[0], yref[0], psiref[0] = x0, y0, psi0
    for k in range(Np):
        psiref[k + 1] = psiref[k] + kappa_path[k] * ds
        xref[k + 1] = xref[k] + ds * np.cos(psiref[k])
        yref[k + 1] = yref[k] + ds * np.sin(psiref[k])
    return xref, yref, psiref

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

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

    # ---------- Reference path ----------                # seconds you intend to run
    kappa_len = int(np.ceil(simtime / dt)) + N
    kappa_path = []
    """
    for i in range(kappa_len):
        if i > kappa_len/4 and i < kappa_len * 3/4:
            kappa_path.append(0.1)
        else:
            kappa_path.append(0)
    """
    """
    for i in range(kappa_len):
        if i == int(kappa_len/2):
            kappa_path.append(capang)
        else:
            kappa_path.append(0)
    """
    for i in range(kappa_len):
        t = i / dt 
        kappa_path.append(0.5*np.cos(i * 0.05))
    xref, yref, psiref = build_ref_from_kappa(kappa_path)
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
                    delta_cmd = mpc_step(x_err, kappa_seq)
                    runtime = time.time() - startime
                    if runtime > largest_sampletime:
                        largest_sampletime = runtime
                    delta_cmd = float(np.clip(delta_cmd, -delta_max, delta_max))
                    last_mpc_t = t
                    """
                    print(f"NEW MPC:   {
                        f"[MPC] t={t:5.2f} "
                        f"x={x:+.2f} y={y:+.2f} "
                        f"e={x_err[0]:+.2f} "
                        f"delta={delta_cmd:+.3f}"
                        f"runtime={runtime}"
                    }")
                    """

                # ---------- Send command back ----------
                reply = {"delta": delta_cmd, "v": v}
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
                reqv.append(v)
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
    








