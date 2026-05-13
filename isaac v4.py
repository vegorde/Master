import asyncio
import json
import math
import socket

import numpy as np
import carb
import omni.usd

from isaacsim.core.api import World
from omni.isaac.core.prims import RigidPrim
from omni.physx import get_physx_interface
from omni.physx.bindings._physx import (
    VEHICLE_WHEEL_STATE_LOCAL_POSE_POSITION,
    VEHICLE_WHEEL_STATE_LOCAL_POSE_QUATERNION,
    VEHICLE_WHEEL_STATE_ROTATION_SPEED,
    VEHICLE_WHEEL_STATE_ROTATION_ANGLE,
    VEHICLE_WHEEL_STATE_STEER_ANGLE,
    VEHICLE_WHEEL_STATE_GROUND_PLANE,
    VEHICLE_WHEEL_STATE_GROUND_ACTOR,
    VEHICLE_WHEEL_STATE_GROUND_SHAPE,
    VEHICLE_WHEEL_STATE_GROUND_MATERIAL,
    VEHICLE_WHEEL_STATE_GROUND_HIT_POSITION,
    VEHICLE_WHEEL_STATE_SUSPENSION_JOUNCE,
    VEHICLE_WHEEL_STATE_SUSPENSION_FORCE,
    VEHICLE_WHEEL_STATE_IS_ON_GROUND,
    VEHICLE_WHEEL_STATE_TIRE_FRICTION,
    VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP,
    VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP,
    VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION,
    VEHICLE_WHEEL_STATE_TIRE_LATERAL_DIRECTION,
    VEHICLE_WHEEL_STATE_TIRE_FORCE
)


HOST = "127.0.0.1"
PORT = 5555
VEHICLE_PRIM_PATH = "/World/Kia_meters_v4"

# Control/MPC update rate in SIM TIME
DT_MPC = 0.1

# Socket settings
CONNECT_TIMEOUT = 3.0

# ---------- low-level controller gains ----------
KP_SPEED_ACCEL = 5.6
KP_SPEED_BRAKE = 0.8
KP_STEER = 1

DELTA_MAX_RAD = math.radians(15)   # must match MPC / vehicle assumptions
MAX_ACCEL_CMD = 1.0
MAX_BRAKE_CMD = 1.0
MAX_STEER_CMD = 1.0

# ---------- origin ----------
origin_set = False
x0, y0, yaw0_deg = 0.0, 0.0, 0.0

# ---------- globals ----------
vehicle_rigid = None
shutdown_started = False


def quat_to_yaw_deg(quat):
    w, x, y, z = quat
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = math.atan2(siny, cosy)
    return math.degrees(yaw_rad)


def wrap_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def ssa(a, b):
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def transform_to_local(pos, yaw_deg):
    global origin_set, x0, y0, yaw0_deg

    x = float(pos[0])
    y = float(pos[1])

    if not origin_set:
        x0, y0, yaw0_deg = x, y, yaw_deg
        origin_set = True

    dx = x - x0
    dy = y - y0

    yaw0_rad = math.radians(yaw0_deg)
    c = math.cos(-yaw0_rad)
    s = math.sin(-yaw0_rad)

    x_tmp = c * dx - s * dy
    y_tmp = s * dx + c * dy

    # forward = +x, left = +y
    x_local = y_tmp
    y_local = -x_tmp
    yaw_local_deg = wrap_angle_deg(yaw_deg - yaw0_deg)

    return x_local, y_local, yaw_local_deg


def set_vehicle_control(vehicle_prim, accel=0.0, brake=0.0, steer=0.0):
    accel = float(np.clip(accel, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    steer = float(np.clip(steer, -1.0, 1.0))

    vehicle_prim.GetAttribute("physxVehicleController:accelerator").Set(accel)
    vehicle_prim.GetAttribute("physxVehicleController:brake0").Set(brake)
    vehicle_prim.GetAttribute("physxVehicleController:steer").Set(steer)


def low_level_control(speed, v_target, delta_target):
    """
    Convert target speed [m/s] and target steering angle [rad]
    into Isaac vehicle commands: accel, brake, steer.
    """

    # ---------- speed controller ----------
    speed_error = float(v_target - speed)

    if speed_error >= 0.0:
        accel_cmd = KP_SPEED_ACCEL * speed_error
        brake_cmd = 0.0
    else:
        accel_cmd = 0.0
        brake_cmd = KP_SPEED_BRAKE * (-speed_error)

    accel_cmd = float(np.clip(accel_cmd, 0.0, MAX_ACCEL_CMD))
    brake_cmd = float(np.clip(brake_cmd, 0.0, MAX_BRAKE_CMD))

    # ---------- steering controller ----------
    # Since we do not have measured front wheel steering angle here,
    # treat delta_target as the desired steering state and map it
    # directly into normalized steer command.
    steer_error = ssa(0.0, float(delta_target))
    steer_cmd = KP_STEER * (delta_target / DELTA_MAX_RAD)
    steer_cmd = float(np.clip(steer_cmd, -MAX_STEER_CMD, MAX_STEER_CMD))

    return accel_cmd, brake_cmd, steer_cmd


class TCPClient:
    """
    Non-blocking line-based JSON client.
    Called from physics callback only at DT_MPC boundaries.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buf = b""
        self.connected = False

    def connect(self):
        self.close()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect((self.host, self.port))
        sock.setblocking(False)

        self.sock = sock
        self.buf = b""
        self.connected = True
        carb.log_info(f"TCP: connected to {self.host}:{self.port}")

    def send_state_get_cmd(self, state_dict):
        if not self.connected or self.sock is None:
            return None

        try:
            self.sock.sendall((json.dumps(state_dict) + "\n").encode("utf-8"))
        except Exception as e:
            carb.log_warn(f"TCP send failed: {e}")
            self.close()
            return None

        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    carb.log_warn("TCP: server closed connection")
                    self.close()
                    return None

                self.buf += chunk
                if b"\n" in self.buf:
                    line, self.buf = self.buf.split(b"\n", 1)
                    return json.loads(line.decode("utf-8"))

        except BlockingIOError:
            # No full line available yet; keep last command
            return None
        except Exception as e:
            carb.log_warn(f"TCP recv failed: {e}")
            self.close()
            return None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.buf = b""
        self.connected = False


async def main():
    global vehicle_rigid, shutdown_started

    dt_phys = 1.0 / 60.0

    if World.instance():
        World.instance().clear_instance()

    world = World(physics_dt=dt_phys, rendering_dt=dt_phys)
    await world.initialize_simulation_context_async()
    await world.reset_async()

    stage = omni.usd.get_context().get_stage()
    vehicle_prim = stage.GetPrimAtPath(VEHICLE_PRIM_PATH)
    if not vehicle_prim or not vehicle_prim.IsValid():
        raise RuntimeError(f"Vehicle prim not found: {VEHICLE_PRIM_PATH}")

    tcp = TCPClient(HOST, PORT)
    try:
        tcp.connect()
    except Exception as e:
        carb.log_warn(f"TCP initial connect failed: {e}")

    sim_time = 0.0
    mpc_timer = 0.0

    # Hold last valid command
    last_cmd = {
        "v": 0.0,
        "delta": 0.0,
    }
    wheel_angle_mesured = 0
    def on_physics_step(dt):
        nonlocal sim_time, mpc_timer, last_cmd
        global vehicle_rigid, shutdown_started, wheel_angle_mesured

        sim_time += dt
        mpc_timer += dt

        if sim_time >= 25.0 and not shutdown_started:
            shutdown_started = True

            set_vehicle_control(vehicle_prim, accel=0.0, brake=1.0, steer=0.0)

            try:
                world.remove_physics_callback("loop")
            except Exception:
                pass

            tcp.close()
            world.pause()
            carb.log_info("[ISAAC] Finished, paused sim and closed TCP.")
            return

        if vehicle_rigid is None:
            try:
                vehicle_rigid = RigidPrim(VEHICLE_PRIM_PATH)
                vehicle_rigid.initialize()
                carb.log_info("RigidPrim initialized")
            except Exception:
                return

        pos, quat = vehicle_rigid.get_world_pose()
        lin_vel = vehicle_rigid.get_linear_velocity()
        speed = float(np.linalg.norm(lin_vel))

        yaw_deg = quat_to_yaw_deg(quat)
        x_local, y_local, yaw_local_deg = transform_to_local(pos, yaw_deg)
        # Run TCP/MPC only at DT_MPC, driven by SIM TIME
        if mpc_timer >= DT_MPC:
            mpc_timer -= DT_MPC

            state_msg = {
                "t": float(sim_time),
                "x": float(x_local),
                "y": float(y_local),
                "yaw": float(math.radians(yaw_local_deg)),
                "v": float(speed),
                "delta": float(wheel_angle_mesured)
            }

            if not tcp.connected:
                try:
                    tcp.connect()
                except Exception as e:
                    carb.log_warn(f"TCP reconnect failed: {e}")

            if tcp.connected:
                cmd = tcp.send_state_get_cmd(state_msg)
                if cmd is not None:
                    last_cmd["delta"] = float(cmd.get("delta", last_cmd["delta"]))
                    last_cmd["v"] = float(cmd.get("v", last_cmd["v"]))

        v_target = float(last_cmd["v"])
        delta_target = float(last_cmd["delta"])

        accel_cmd, brake_cmd, steer_cmd = low_level_control(
            speed=speed,
            v_target=v_target,
            delta_target=delta_target,
        )

        set_vehicle_control(
            vehicle_prim,
            accel=accel_cmd,
            brake=brake_cmd,
            steer=steer_cmd,
        )
        # typical 4 wheel vehicle defaults
        wheel_list = [  "/LeftWheel1References",
                        "/RightWheel1References",
                        "/LeftWheel2References",
                         "/RightWheel2References"]

        # loop through all of the wheels
        _physx_interface = omni.physx.get_physx_interface()
        current_wheel = "/LeftWheel1References"
        wheel_path = VEHICLE_PRIM_PATH + current_wheel
        wheel_state = _physx_interface.get_wheel_state(wheel_path)
        wheel_angle_mesured = wheel_state[VEHICLE_WHEEL_STATE_STEER_ANGLE]

       # print(
        #f"tcp={'ON' if tcp.connected else 'OFF'} "
        #f"local=({x_local:+.3f}, {y_local:+.3f}) "
       # f"yaw_deg={yaw_local_deg:+.2f} "
       # f"v={speed:.3f} "
       # f"v_tgt={v_target:.3f} "
       # f"delta_tgt_deg={math.degrees(delta_target):+.2f} "
       # f"cmd=(a={accel_cmd:.3f}, b={brake_cmd:.3f}, s={steer_cmd:.3f})"
       # f"wheel_angle  ===  {wheel_angle_mesured}"
       # )

    world.add_physics_callback("loop", on_physics_step)
    await world.play_async()


asyncio.get_event_loop().create_task(main())