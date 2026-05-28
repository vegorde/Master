# close_isac_bicycle_standalone.py
import socket
import json
import math
import time
import csv
import matplotlib.pyplot as plt

# ── Edit this ─────────────────────────────────────────────────────────────────
# Point at the same path.csv used by the MPC server.
# The simulator will start at the first waypoint, heading toward the second.
# Set to "" to fall back to x=0, y=0, yaw=0.
#PATH_CSV = r"C:\Users\vegar\Documents\Isacsim\git\local\Master\Rosbag\path.csv"
PATH_CSV = r"c:\Users\vegar\Documents\Skole\Master\Code\Master\Rosbag\path.csv"
# ─────────────────────────────────────────────────────────────────────────────


def initial_state_from_csv(path_csv):
    """Read the first two waypoints from a path CSV and return (x, y, yaw)."""
    pts = []
    with open(path_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
            if len(pts) == 2:
                break
    if len(pts) < 2:
        raise ValueError(f"Need at least 2 waypoints in {path_csv}")
    x0, y0 = pts[0]
    yaw0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    return x0, y0, yaw0


class TCPClient:
    """
    Line-based JSON TCP client.
    Sends a state and BLOCKS (with timeout) until it gets one cmd line back.
    This prevents the simulator from outrunning the MPC solver when running fast.
    """
    def __init__(self, host="127.0.0.1", port=5555, recv_timeout=1.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(recv_timeout)  # timeout for recv() while waiting for reply
        self.buf = b""

    def send_state_get_cmd(self, state_dict):
        # send
        self.sock.sendall((json.dumps(state_dict) + "\n").encode("utf-8"))

        # receive exactly one JSON line (blocking up to timeout)
        while True:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))

            chunk = self.sock.recv(4096)  # blocks until data or timeout
            if not chunk:
                return None
            self.buf += chunk

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def wrap_pi(a: float) -> float:
    # keep yaw in [-pi, pi] (optional, but often convenient)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

default_speed = 15
def main():
    # --------- user settings ----------
    HOST = "127.0.0.1"
    PORT = 5555

    # Physics dt (simulation step)
    dt_phys = 1.0 / 30.0   # 30 Hz like your Isaac script :contentReference[oaicite:1]{index=1}

    # MPC / control update period
    dt_mpc = 0.1           # 10 Hz (must match server)

    max_time = 75.0 * 20/default_speed
    max_steps = int(max_time / dt_phys)

    # Vehicle parameters (match what your MPC assumes)
    wheel_base = 2.79

    # Limits (optional but realistic)
    max_steer = math.radians(35)   # steering angle limit
    max_speed = 20                 # m/s speed limit

    # First-order actuator response (optional but helpful)
    # Set to 0.0 to make it “instant”
    tau_v = 0.20      # seconds (speed response)
    tau_delta = 1.0  # seconds (steer response)

    # --------- initial state ----------
    if PATH_CSV:
        x, y, yaw = initial_state_from_csv(PATH_CSV)
        print(f"[BICYCLE] Starting at ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg  (from path CSV)")
    else:
        x, y, yaw = 0.0, 0.0, 0.0

    # Actual simulated states (what you report as v)
    v = 1.0
    delta = 0.0

    # Command targets coming from MPC server
    v_cmd = 1.0
    delta_cmd = 0.0

    # Hold last valid command (same idea as Isaac client)
    last_cmd = {"delta": delta_cmd, "v": v_cmd}

    # TCP
    tcp = TCPClient(HOST, PORT)
    print(f"[BICYCLE] Connected to MPC server on {HOST}:{PORT}")

    simtime = 0.0
    mpc_timer = 0.0

    # Logging buffers
    log_t       = []
    log_x       = []
    log_y       = []
    log_delta   = []   # actual steering angle
    log_delta_cmd = [] # target steering angle from MPC

    # If you want the sim to run in real-time, keep this True.
    # If False, it runs as fast as it can (often better for testing).
    run_realtime = False
    wall_t0 = time.time()

    for step_count in range(1, max_steps + 1):
        simtime += dt_phys
        mpc_timer += dt_phys

        # --- TCP/MPC update at dt_mpc ---
        if mpc_timer >= dt_mpc:
            mpc_timer -= dt_mpc

            state_msg = {
                "step": int(step_count),
                "t": float(simtime),
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
                "v": float(v),
                "delta": float(delta)
            }

            cmd = tcp.send_state_get_cmd(state_msg)
            if cmd is not None:
                last_cmd = cmd

            # Extract commanded targets (same keys as Isaac client expects)
            delta_cmd = float(last_cmd.get("delta", delta_cmd))
            v_cmd = float(last_cmd.get("v", v_cmd))

            # Optional clamps
            delta_cmd = clamp(delta_cmd, -max_steer, max_steer)
            v_cmd = clamp(v_cmd, -max_speed, max_speed)

        # --- “actuators”: first-order approach to commanded v/delta ---
        if tau_v > 0:
            v += (dt_phys / tau_v) * (v_cmd - v)
        else:
            v = v_cmd

        if tau_delta > 0:
            delta += (dt_phys / tau_delta) * (delta_cmd - delta)
        else:
            delta = delta_cmd

        # --- kinematic bicycle model integration ---
        # xdot = v cos(yaw)
        # ydot = v sin(yaw)
        # yawdot = v/L * tan(delta)
        beta = 0.0  # (pure kinematic bicycle; no slip angle term)
        x += dt_phys * v * math.cos(yaw + beta)
        y += dt_phys * v * math.sin(yaw + beta)
        yaw += dt_phys * (v / wheel_base) * math.tan(delta)
        yaw = wrap_pi(yaw)

        # Logging occasionally
        if step_count % 300 == 0:
            print(
                f"[BICYCLE] dt={dt_phys:.4f}s, simtime={simtime:.2f}s, "
                f"x={x:.2f}, y={y:.2f}, yaw={yaw:+.3f}, v={v:.2f}, delta={delta:+.3f}"
            )

        # Optional real-time pacing
        if run_realtime:
            target = wall_t0 + simtime
            now = time.time()
            if target > now:
                time.sleep(target - now)

    tcp.close()
    print("[BICYCLE] Finished, closed TCP.")


if __name__ == "__main__":
    main()
