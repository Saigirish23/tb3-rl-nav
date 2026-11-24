#!/usr/bin/env python3
"""Standalone PPO Inference Node for TurtleBot3 Navigation in Gazebo Classic & Hardware.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf
Interface:
- 43-dim Observation vector:
  * 36 pooled LiDAR bins (10° width, min-pooled, [0, 1])
  * 1 goal distance normalized by arena diagonal [0, 1]
  * 2 goal bearing [sin, cos] in robot frame
  * 1 measured linear velocity v / 0.15 in [-1, 1]
  * 1 measured angular velocity omega / 0.60 in [-1, 1]
  * 2 previous normalized action u in [-1, 1]^2
- Action scaling:
  * cmd_v = 0.075 * (u[0] + 1.0) in [0.0, 0.15] m/s
  * cmd_omega = 0.60 * u[1] in [-0.60, 0.60] rad/s
- Safety criteria:
  * Goal reached: distance <= 0.20 m
  * Collision: min LiDAR range < 0.16 m
  * Stale sensor watchdog: no scan within 1.0 s -> stop
  * Timeout: step_count >= max_steps (default 350)
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.env.observation import ObservationPreprocessor, OBS_DIM, wrap_to_pi, RANGE_MAX


class PPOInferenceController:
    """Core policy inference controller decoupled from ROS execution loop."""

    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str] = None,
        goal_x: float = 1.0,
        goal_y: float = 1.0,
        goal_tolerance: float = 0.20,
        collision_threshold: float = 0.16,
        max_steps: int = 350,
        dt: float = 0.20,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.vecnorm_path = Path(vecnorm_path) if vecnorm_path else None
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_tolerance = float(goal_tolerance)
        self.collision_threshold = float(collision_threshold)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.device = device

        self.preprocessor = ObservationPreprocessor()
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.is_done = False
        self.outcome = "RUNNING"
        self.trajectory: List[Dict[str, float]] = []

        # Load SB3 model
        self.model = self._load_model()
        self.vecnorm = self._load_vecnorm()

    def _load_model(self):
        """Load PPO model from file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        from stable_baselines3 import PPO
        model = PPO.load(str(self.model_path), device=self.device)
        print(f"[PPOInference] Successfully loaded PPO model: {self.model_path} (timesteps={model.num_timesteps:,})")
        return model

    def _load_vecnorm(self):
        """Validate and load VecNormalize statistics if present."""
        if not self.vecnorm_path or not self.vecnorm_path.exists():
            print("[PPOInference] No VecNormalize path provided or file missing; using unnormalized rewards.")
            return None
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym

        class _DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        dummy = DummyVecEnv([_DummyEnv])
        vn = VecNormalize.load(str(self.vecnorm_path), dummy)
        count = getattr(getattr(vn, 'ret_rms', None), 'count', 'unknown')
        print(f"[PPOInference] Successfully loaded VecNormalize stats (count={count})")
        return vn

    def reset(self, init_x: float = 0.0, init_y: float = 0.0, init_theta: float = 0.0, goal_x: Optional[float] = None, goal_y: Optional[float] = None):
        """Reset state for a new navigation trial."""
        if goal_x is not None:
            self.goal_x = float(goal_x)
        if goal_y is not None:
            self.goal_y = float(goal_y)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.is_done = False
        self.outcome = "RUNNING"
        self.trajectory.clear()
        self.preprocessor.reset(init_x=init_x, init_y=init_y, init_theta=init_theta)
        print(f"[PPOInference] Reset trial: Start=({init_x:.2f}, {init_y:.2f}, {init_theta:.2f} rad) -> Goal=({self.goal_x:.2f}, {self.goal_y:.2f})")

    def step(self, ranges: np.ndarray, v_meas: float, omega_meas: float) -> Tuple[float, float, bool, str, Dict[str, Any]]:
        """Compute next control action from sensor feedback.

        Returns:
            (cmd_v, cmd_omega, is_done, outcome, info)
        """
        if self.is_done:
            return 0.0, 0.0, True, self.outcome, {}

        self.step_count += 1

        # 1. Check safety collision against raw LiDAR scan
        valid_ranges = ranges[(~np.isnan(ranges)) & (~np.isinf(ranges)) & (ranges > 0.0)]
        min_range = float(np.min(valid_ranges)) if len(valid_ranges) > 0 else float('inf')
        if min_range < self.collision_threshold:
            self.is_done = True
            self.outcome = "COLLISION"
            print(f"[PPOInference] Safety Stop: COLLISION detected! (min_range={min_range:.3f} m < {self.collision_threshold:.3f} m)")
            return 0.0, 0.0, True, self.outcome, {"min_range": min_range, "step": self.step_count}

        # 2. Preprocess 43-dim observation
        obs = self.preprocessor.step(
            ranges=ranges,
            v_meas=v_meas,
            omega_meas=omega_meas,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            last_action=self.prev_action,
            dt=self.dt,
        )

        # 3. Check goal reached via estimated pose
        cur_x = self.preprocessor.estimated_pose.x
        cur_y = self.preprocessor.estimated_pose.y
        cur_theta = self.preprocessor.estimated_pose.theta
        goal_dist = math.hypot(self.goal_x - cur_x, self.goal_y - cur_y)

        self.trajectory.append({
            "step": self.step_count,
            "x": round(cur_x, 3),
            "y": round(cur_y, 3),
            "theta": round(cur_theta, 3),
            "goal_dist": round(goal_dist, 3),
            "min_lidar": round(min_range, 3),
        })

        if goal_dist <= self.goal_tolerance:
            self.is_done = True
            self.outcome = "SUCCESS"
            print(f"[PPOInference] GOAL REACHED! (dist={goal_dist:.3f} m <= {self.goal_tolerance:.3f} m in {self.step_count} steps)")
            return 0.0, 0.0, True, self.outcome, {"goal_dist": goal_dist, "step": self.step_count}

        # 4. Check step timeout
        if self.step_count >= self.max_steps:
            self.is_done = True
            self.outcome = "TIMEOUT"
            print(f"[PPOInference] Step TIMEOUT reached ({self.max_steps} steps). Final goal dist: {goal_dist:.3f} m")
            return 0.0, 0.0, True, self.outcome, {"goal_dist": goal_dist, "step": self.step_count}

        # 5. Model prediction (deterministic action)
        action, _ = self.model.predict(obs, deterministic=True)
        u = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.prev_action = u.copy()

        # 6. Action conversion to velocities
        cmd_v = float(0.075 * (u[0] + 1.0))
        cmd_omega = float(0.60 * u[1])

        info = {
            "step": self.step_count,
            "goal_dist": goal_dist,
            "cmd_v": cmd_v,
            "cmd_omega": cmd_omega,
            "min_range": min_range,
        }
        return cmd_v, cmd_omega, False, self.outcome, info


def run_ros2_node(args):
    """Run standalone ROS 2 inference node connected to Gazebo or hardware robot."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist

    if not rclpy.ok():
        rclpy.init()

    node = Node('tb3_ppo_inference_node')
    print("=" * 65)
    print("TB3 Standalone PPO Inference Node Started")
    print(f"  Model Path:    {args.model_path}")
    print(f"  VecNormalize:  {args.vecnorm_path}")
    print(f"  Target Goal:   ({args.goal_x:.2f}, {args.goal_y:.2f})")
    print(f"  Goal Radius:   {args.goal_tolerance:.2f} m")
    print("=" * 65)

    controller = PPOInferenceController(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_tolerance=args.goal_tolerance,
        collision_threshold=args.collision_threshold,
        max_steps=args.max_steps,
        dt=0.20,
    )

    state = {
        "latest_scan": None,
        "latest_odom": None,
        "last_scan_time": time.monotonic(),
        "has_started": False,
    }

    def scan_cb(msg: LaserScan):
        state["latest_scan"] = msg
        state["last_scan_time"] = time.monotonic()

    def odom_cb(msg: Odometry):
        state["latest_odom"] = msg

    qos_scan = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    qos_odom = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

    node.create_subscription(LaserScan, args.scan_topic, scan_cb, qos_scan)
    node.create_subscription(Odometry, args.odom_topic, odom_cb, qos_odom)
    cmd_pub = node.create_publisher(Twist, args.cmd_vel_topic, 10)

    # Wait for sensor topics to become active
    print(f"[PPOInference] Waiting for sensor topics ({args.scan_topic}, {args.odom_topic})...")
    timeout = 10.0
    start_wait = time.monotonic()
    while time.monotonic() - start_wait < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state["latest_scan"] is not None and state["latest_odom"] is not None:
            print("[PPOInference] Sensor topics received. Beginning navigation.")
            break
    else:
        print("[PPOInference] ERROR: Timed out waiting for sensors. Exiting.")
        cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        return

    # Reset controller with current odometry position
    init_x = state["latest_odom"].pose.pose.position.x
    init_y = state["latest_odom"].pose.pose.position.y
    q = state["latest_odom"].pose.pose.orientation
    init_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    controller.reset(init_x=init_x, init_y=init_y, init_theta=init_theta)

    rate_hz = 5.0
    dt = 1.0 / rate_hz

    trial_log_file = Path(args.output_dir) / f"inference_trial_{int(time.time())}.json"
    trial_log_file.parent.mkdir(parents=True, exist_ok=True)

    nav_start_time = time.monotonic()

    try:
        while rclpy.ok() and not controller.is_done:
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.01)

            # Check scan staleness (> 1.0s)
            if time.monotonic() - state["last_scan_time"] > 1.0:
                print("[PPOInference] WARNING: Scan data stale (> 1.0s)! Halting robot for safety.")
                cmd_pub.publish(Twist())
                time.sleep(dt)
                continue

            ranges = np.array(state["latest_scan"].ranges, dtype=np.float32)
            v_meas = float(state["latest_odom"].twist.twist.linear.x)
            omega_meas = float(state["latest_odom"].twist.twist.angular.z)

            cmd_v, cmd_omega, is_done, outcome, info = controller.step(ranges, v_meas, omega_meas)

            # Publish commanded velocity
            twist = Twist()
            twist.linear.x = cmd_v
            twist.angular.z = cmd_omega
            cmd_pub.publish(twist)

            # Periodic status display (every 5 steps / 1.0s)
            if controller.step_count % 5 == 0 or is_done:
                cur_x = controller.preprocessor.estimated_pose.x
                cur_y = controller.preprocessor.estimated_pose.y
                cur_th = controller.preprocessor.estimated_pose.theta
                elapsed = time.monotonic() - nav_start_time
                print(
                    f"[PPOInference] Step {controller.step_count:3d} ({elapsed:4.1f}s) | "
                    f"Pose: ({cur_x:5.2f}, {cur_y:5.2f}, {cur_th:5.2f} rad) | "
                    f"Goal Dist: {info.get('goal_dist', 0.0):5.2f}m | "
                    f"Min LiDAR: {info.get('min_range', 0.0):5.2f}m | "
                    f"Cmd: (v={cmd_v:4.2f}, w={cmd_omega:5.2f})"
                )

            if is_done:
                break

            # Sleep remaining time to enforce 5 Hz control rate
            elapsed_step = time.monotonic() - t0
            sleep_time = max(0.0, dt - elapsed_step)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[PPOInference] Manual interruption requested. Stopping robot.")
    finally:
        # Stop robot unconditionally on exit
        for _ in range(5):
            cmd_pub.publish(Twist())
            time.sleep(0.02)

        duration = round(time.monotonic() - nav_start_time, 2)
        final_x = controller.trajectory[-1]["x"] if controller.trajectory else init_x
        final_y = controller.trajectory[-1]["y"] if controller.trajectory else init_y
        displacement = math.hypot(final_x - init_x, final_y - init_y)
        final_dist = math.hypot(args.goal_x - final_x, args.goal_y - final_y)
        min_clearance = min((pt["min_lidar"] for pt in controller.trajectory), default=float("nan"))

        # Save enriched trial metadata
        trial_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(args.model_path),
            "vecnorm_path": str(args.vecnorm_path) if args.vecnorm_path else None,
            "goal": [args.goal_x, args.goal_y],
            "goal_tolerance": args.goal_tolerance,
            "outcome": controller.outcome,
            "success": bool(controller.outcome == "SUCCESS"),
            "collision": bool(controller.outcome == "COLLISION"),
            "timeout": bool(controller.outcome == "TIMEOUT"),
            "steps": controller.step_count,
            "duration_sec": duration,
            "start_pose": [round(init_x, 3), round(init_y, 3), round(init_theta, 3)],
            "final_pose": [round(final_x, 3), round(final_y, 3)],
            "displacement_meters": round(displacement, 3),
            "final_goal_distance_meters": round(final_dist, 3),
            "min_obstacle_clearance_meters": round(min_clearance, 3),
            "trajectory": controller.trajectory,
        }
        with open(trial_log_file, "w", encoding="utf-8") as f:
            json.dump(trial_data, f, indent=2)
        print(f"[PPOInference] Trial summary saved to: {trial_log_file}")

        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Standalone TB3 PPO Policy Inference Node")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SB3 PPO model .zip")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--goal-x", type=float, default=1.0, help="Goal x coordinate in meters")
    parser.add_argument("--goal-y", type=float, default=1.0, help="Goal y coordinate in meters")
    parser.add_argument("--goal-tolerance", type=float, default=0.20, help="Goal tolerance radius in meters")
    parser.add_argument("--collision-threshold", type=float, default=0.16, help="Emergency collision stop threshold")
    parser.add_argument("--max-steps", type=int, default=350, help="Max control steps per episode")
    parser.add_argument("--scan-topic", type=str, default="/scan", help="LaserScan topic")
    parser.add_argument("--odom-topic", type=str, default="/odom", help="Odometry topic")
    parser.add_argument("--cmd-vel-topic", type=str, default="/cmd_vel", help="Twist velocity topic")
    parser.add_argument("--output-dir", type=str, default="runs/inference_logs", help="Directory to save trial logs")
    args = parser.parse_args()

    run_ros2_node(args)


if __name__ == "__main__":
    main()
