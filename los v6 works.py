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
capang = 3.14 * 2


delta_max = np.deg2rad(35)
ddelta_max = np.deg2rad(60) * dt

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



def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def compute_errors(x, y, yaw, xr, yr, psir):
    dx = x - xr
    dy = y - yr
    e = -np.sin(psir) * dx + np.cos(psir) * dy          # cross-track error
    psi_err = wrap_angle(yaw - psir)                     # heading error vs path heading
    return np.array([e, psi_err], dtype=float)

class LOSYawController:
    """
    LOS guidance producing desired yaw ψ_d.
    """
    def __init__(self,
                 lookahead: float = 5.0,
                 yaw_rate_limit: float = np.deg2rad(45)):
        self.lookahead = float(lookahead)
        self.yaw_rate_limit = float(yaw_rate_limit)
        self._prev_psi_d = None

    def reset(self):
        self._prev_psi_d = None

    def step(self, e: float, psir: float, dt: float) -> float:
        """
        Inputs:
            e    : cross-track error (m)
            psir : path heading (rad)
            dt   : timestep (s)

        Output:
            psi_d : desired yaw (rad)
        """

        # --- LOS guidance (THIS replaces yaw-rate control) ---
        psi_d = psir - np.arctan2(e, self.lookahead)
        psi_d = wrap_angle(psi_d)

        
        # --- Optional yaw-rate limiting ---
        if self._prev_psi_d is not None and dt > 0:
            dpsi = wrap_angle(psi_d - self._prev_psi_d)
            max_dpsi = self.yaw_rate_limit * dt
            dpsi = np.clip(dpsi, -max_dpsi, max_dpsi)
            psi_d = wrap_angle(self._prev_psi_d + dpsi)
        self._prev_psi_d = psi_d
        
        return psi_d


"""
ctrl = LOSHeadingController(lookahead=8.0, k_psi=1.2, k_d=0.15, r_max=np.deg2rad(45))

    # vehicle state
    x, y, yaw = 0.0, 2.0, np.deg2rad(10)

    # reference point on path + path heading at that point
    xr, yr, psir = 0.0, 0.0, 0.0

    dt = 0.02

    e, psi_err = compute_errors(x, y, yaw, xr, yr, psir)
    r_cmd = ctrl.step(e=e, psi_err=psi_err, dt=dt)

    print("cross-track e:", e, "psi_err:", psi_err, "yaw_rate_cmd:", r_cmd)

"""


if __name__ == "__main__":
    print("[LOS SERVER] Starting...")
    ctrl = LOSYawController(lookahead=1, yaw_rate_limit=np.deg2rad(180))
    Kp = 1
    #ctrl = ILOSGuidanceYaw(lookahead=10, sigma=0.5, yaw_rate_limit=np.deg2rad(45), int_state_limit=10)
    # ---------- logging buffers ----------
    xs, ys = [], []
    deltas, vels, reqv = [], [], []
    es, psis = [], []
    ts = []

    # ---------- Reference path ----------                # seconds you intend to run
    kappa_len = int(np.ceil(simtime / dt)) + N
    kappa_path = []
    capang = dt * 3.14 * 50 
    """
    for i in range(kappa_len):
        t = i / dt 
        kappa_path.append(0.5*np.cos(i * 0.05))
    """
    for i in range(kappa_len):
        if i == int(kappa_len/2):
            kappa_path.append(capang)       
        else:
            kappa_path.append(0)
    
    """
    capang = dt * 3.14 * 50 / (20 * 2)
    center = kappa_len // 2
    half_width = 20  # 10 points total

    for i in range(kappa_len):
        if center - half_width <= i < center + half_width:
            kappa_path.append(capang)
        else:
            kappa_path.append(0)
    """
    
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
                start_time = time.time()

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
                idx = closest_index_windowed(x, y, xref, yref, last_idx=idx, window=80)
                idx = max(idx, last_idx)
                last_idx = idx

                # ---------- MPC error state ----------
                x_err = compute_errors(
                    x, y, yaw,
                    xref[idx], yref[idx], psiref[idx]
                )

                
                if t - last_mpc_t >= dt:

                    Chi_desired_NED = ctrl.step(x_err[0], psiref[idx], dt=dt)
                    delta_cmd = (Chi_desired_NED - yaw) * Kp
                    delta_cmd = np.clip(delta_cmd, np.deg2rad(-35), np.deg2rad(35))
                    #print(f"PSIREF {psiref[idx]}")
                    #delta_cmd = float(np.clip(delta_cmd, -delta_max, delta_max))
                    last_mpc_t = t
                    #print(f"NEW MPC:   {
                    #    f"[MPC] t={t:5.2f} "
                    #    f"x={x:+.2f} y={y:+.2f} "
                    #    f"e={x_err[0]:+.2f} "
                    #    f"delta={delta_cmd:+.3f}"
                    #    f"yaw={yaw}"
                    #}")

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
                #loop_time = time.time() - start_time
                #print(f"Loop runtime: {loop_time:.6f} s")
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








