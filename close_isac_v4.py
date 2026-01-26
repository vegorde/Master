# isaac_client.py
import numpy as np
import asyncio
import socket
import json
import math

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils
import carb


def quat_to_yaw(q):
    q = np.asarray(q, dtype=float).flatten()
    w, x, y, z = q  # Isaac commonly gives (w,x,y,z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class TCPClient:
    """
    Simple line-based JSON TCP client.
    Non-blocking recv; we only call it at dt_mpc.
    """
    def __init__(self, host="127.0.0.1", port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(3.0)
        self.sock.connect((host, port))
        self.sock.setblocking(False)
        self.buf = b""

    def send_state_get_cmd(self, state_dict):
        try:
            self.sock.sendall((json.dumps(state_dict) + "\n").encode("utf-8"))
        except Exception:
            return None

        # Non-blocking read of exactly one line (if available)
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    return None
                self.buf += chunk
                if b"\n" in self.buf:
                    line, self.buf = self.buf.split(b"\n", 1)
                    return json.loads(line.decode("utf-8"))
        except BlockingIOError:
            return None
        except Exception:
            return None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


async def main():
    # --------- user settings ----------
    HOST = "127.0.0.1"
    PORT = 5555

    # Physics dt (simulation step)
    dt_phys = 1.0 / 30.0     # 60 Hz physics

    # MPC / control update period (slower than physics)
    dt_mpc = 0.1           # 10 Hz MPC updates (must match server dt_mpc)

    max_time = 20
    max_steps = max_time / dt_phys

    # Vehicle parameters (match your controller + MPC)
    wheel_base = 0.32
    track_width = 0.242
    wheel_radius = 0.052

   
     # Initial command defaults
    desired_forward_vel = 1.0
    desired_steering_angle = 0.0
    acceleration = 0.0
    steering_velocity = 0.0

    # --------- reset world ----------
    if World.instance():
        World.instance().clear_instance()

    # Create world with a fixed physics dt (this is the "actual sim dt")
    world = World(physics_dt=dt_phys, rendering_dt=dt_phys)
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # Load Leatherback
    assets_root_path = get_assets_root_path()
    leatherback_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    leatherback_prim_path = "/World/Leatherback"
    stage_utils.add_reference_to_stage(leatherback_asset_path, leatherback_prim_path)

    robot = world.scene.add(Robot(prim_path=leatherback_prim_path, name="my_leatherback"))

    # Must reset AFTER adding stage refs / scene objects
    await world.reset_async()

    # DOF names
    steering_joint_names = ["Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right"]
    wheel_joint_names = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Left",
        "Wheel__Upright__Rear_Right",
    ]
    steering_joint_indices = [robot.get_dof_index(n) for n in steering_joint_names]
    wheel_joint_indices = [robot.get_dof_index(n) for n in wheel_joint_names]

    # Controller
    controller = AckermannController(
        "tcp_controller",
        wheel_base=wheel_base,
        track_width=track_width,
        front_wheel_radius=wheel_radius,
        back_wheel_radius=wheel_radius,
    )

    # TCP
    tcp = TCPClient(HOST, PORT)
    carb.log_info(f"[ISAAC] Connected to MPC server on {HOST}:{PORT}")

    # Timers
    simtime = 0.0
    mpc_timer = 0.0
    step_count = 0

    # Hold last valid command (important)
    last_cmd = {"delta": desired_steering_angle, "v": desired_forward_vel}

    def on_physics_step(dt: float):
        nonlocal simtime, mpc_timer, step_count
        nonlocal desired_forward_vel, desired_steering_angle
        nonlocal last_cmd

        simtime += dt
        mpc_timer += dt
        step_count += 1

        if step_count >= max_steps:
            try:
                world.remove_physics_callback("tcp_ackermann")
            except Exception:
                pass
            tcp.close()
            world.pause()
            carb.log_info("[ISAAC] Finished, paused sim and closed TCP.")
            return

        # Read state every physics step
        root_pos, root_orient = robot.get_world_pose()
        root_vel = robot.get_linear_velocity()
        lin_root_vel  = np.sqrt(root_vel[0]**2 + root_vel[1]**2 + root_vel[2]**2)
        yaw = quat_to_yaw(root_orient)

        # Only run MPC/tcp at dt_mpc
        if mpc_timer >= dt_mpc:
            mpc_timer -= dt_mpc

            state_msg = {
                "step": int(step_count),
                "t": float(simtime),          # simulation time (not wall time)
                "x": float(root_pos[0]),
                "y": float(root_pos[1]),
                "yaw": float(yaw),
                "v": float(lin_root_vel),
            }

            cmd = tcp.send_state_get_cmd(state_msg)
            if cmd is not None:
                last_cmd = cmd

            desired_steering_angle = float(last_cmd.get("delta", desired_steering_angle))
            desired_forward_vel = float(last_cmd.get("v", desired_forward_vel))

        # Apply action every physics step using the latest held command
        actions = controller.forward([
            float(desired_steering_angle),
            float(steering_velocity),
            float(desired_forward_vel),
            float(acceleration),
            float(dt),   # IMPORTANT: physics dt here
        ])

        full_joint_positions = np.zeros(robot.num_dof, dtype=np.float32)
        full_joint_positions[steering_joint_indices[0]] = actions.joint_positions[0]
        full_joint_positions[steering_joint_indices[1]] = actions.joint_positions[1]

        full_joint_velocities = np.zeros(robot.num_dof, dtype=np.float32)
        full_joint_velocities[wheel_joint_indices[0]] = actions.joint_velocities[0]
        full_joint_velocities[wheel_joint_indices[1]] = actions.joint_velocities[1]
        full_joint_velocities[wheel_joint_indices[2]] = actions.joint_velocities[2]
        full_joint_velocities[wheel_joint_indices[3]] = actions.joint_velocities[3]

        robot.apply_action(ArticulationAction(
            joint_positions=full_joint_positions,
            joint_velocities=full_joint_velocities,
        ))

        # Occasional dt print
        if step_count % 300 == 0:
            carb.log_info(f"[ISAAC] dt_phys={dt:.6f}s (~{1.0/dt:.1f} Hz), simtime={simtime:.2f}s, delta={desired_steering_angle:+.4f}")

    # Register callback AFTER sim context init/reset
    world.add_physics_callback("tcp_ackermann", on_physics_step)

    # Start sim
    await world.play_async()
    carb.log_info("[ISAAC] Running...")

# Run in Isaac Script Editor
asyncio.get_event_loop().create_task(main())
