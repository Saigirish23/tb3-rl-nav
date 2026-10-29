#!/usr/bin/env python3
"""Interactive Waypoint Navigation for TurtleBot3 PPO Policy.

Allows the user to manually enter (x, y) target coordinates from the terminal.
Navigates the robot from its CURRENT position to the target goal using the trained PPO policy.
Validates boundaries & obstacle clearance:
- Rejects coordinates outside the 4m x 4m arena (safe limits [-1.85, 1.85] m).
- Rejects coordinates that fall inside or too close to any active obstacle.
- Automatically prevents the robot from spawning inside/on top of obstacles.
"""

from __future__ import annotations
import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from deploy.ppo_inference_node import PPOInferenceController

# Arena specifications
ARENA_HALF_WIDTH = 2.0  # 4m x 4m arena -> bounds [-2.0, 2.0]
SAFE_MARGIN = 0.15      # Wall thickness + robot clearance margin
MAX_SAFE_COORD = ARENA_HALF_WIDTH - SAFE_MARGIN  # 1.85 meters
ROBOT_FOOTPRINT_RADIUS = 0.105                   # TurtleBot3 Burger footprint

# Known physical dimensions of pool obstacles in arena_4x4.world
OBSTACLE_CATALOG = {
    "obstacle_cyl_1": {"shape": "cylinder", "radius": 0.15},
    "obstacle_cyl_2": {"shape": "cylinder", "radius": 0.12},
    "obstacle_cyl_3": {"shape": "cylinder", "radius": 0.18},
    "obstacle_cyl_4": {"shape": "cylinder", "radius": 0.15},
    "obstacle_box_1": {"shape": "box", "half_size": 0.15},   # 0.30 x 0.30 m
    "obstacle_box_2": {"shape": "box", "half_size": 0.125},  # 0.25 x 0.25 m
    "obstacle_box_3": {"shape": "box", "half_size": 0.175},  # 0.35 x 0.35 m
    "obstacle_box_4": {"shape": "box", "half_size": 0.15},   # 0.30 x 0.30 m
}


class InteractiveNavigator(Node):
    """ROS 2 Node for interactive keyboard-driven PPO waypoint navigation."""

    def __init__(
        self,
        model_path: Path,
        vecnorm_path: Optional[Path],
        active_obstacles: Optional[List[Dict[str, Any]]] = None,
        arena_limit: float = 1.85,
        goal_tolerance: float = 0.20,
        collision_threshold: float = 0.13,
    ):
        super().__init__("interactive_tb3_navigator")
        self.arena_limit = float(arena_limit)
        self.goal_tolerance = float(goal_tolerance)
        self.active_obstacles = active_obstacles or []

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.last_scan_time = 0.0

        # Subscriptions
        qos_scan = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_odom = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, qos_scan)
        self.create_subscription(Odometry, "/odom", self._odom_callback, qos_odom)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Controller
        self.controller = PPOInferenceController(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            goal_x=0.0,
            goal_y=0.0,
            goal_tolerance=self.goal_tolerance,
            collision_threshold=collision_threshold,
            max_steps=500,
            dt=0.20,
        )

    def _scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = time.monotonic()

    def _odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def stop_robot(self):
        """Immediately command zero linear and angular velocity."""
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)

    def get_current_pose(self) -> tuple[float, float, float]:
        """Return (x, y, theta) from current odometry."""
        if self.latest_odom is None:
            return 0.0, 0.0, 0.0
        p = self.latest_odom.pose.pose.position
        q = self.latest_odom.pose.pose.orientation
        theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(p.x), float(p.y), float(theta)

    def check_coordinate_validity(self, x: float, y: float, clearance_margin: float = 0.05) -> Tuple[bool, str]:
        """Validate if target (x, y) is inside the arena and clear of obstacles.
        
        Returns:
            (is_valid, error_message)
        """
        # 1. Arena boundary check
        if abs(x) > self.arena_limit or abs(y) > self.arena_limit:
            msg = (
                f"Target coordinate ({x:.2f}, {y:.2f}) is OUTSIDE the 4m x 4m arena boundary!\n"
                f"  Safe navigable limits are [-{self.arena_limit:.2f}, +{self.arena_limit:.2f}] for both X and Y."
            )
            return False, msg

        # 2. Obstacle collision check
        for obs in self.active_obstacles:
            name = obs.get("name", "unknown_obstacle")
            ox = float(obs["x"])
            oy = float(obs["y"])
            geo = OBSTACLE_CATALOG.get(name, {"shape": "cylinder", "radius": 0.15})
            shape = geo.get("shape", "cylinder")

            # Required safety buffer = obstacle radius/half-size + robot radius (0.105m) + safety margin
            if shape == "cylinder":
                r = float(geo.get("radius", 0.15))
                min_clearance = r + ROBOT_FOOTPRINT_RADIUS + clearance_margin
                dist = math.hypot(x - ox, y - oy)
                if dist <= min_clearance:
                    msg = (
                        f"Target coordinate ({x:.2f}, {y:.2f}) falls INSIDE or TOO CLOSE to obstacle '{name}'!\n"
                        f"  Obstacle Center: ({ox:.2f}, {oy:.2f}) | Radius: {r:.2f} m\n"
                        f"  Distance: {dist:.2f} m <= Minimum Required Clearance: {min_clearance:.2f} m"
                    )
                    return False, msg
            elif shape == "box":
                half_size = float(geo.get("half_size", 0.15))
                min_half = half_size + ROBOT_FOOTPRINT_RADIUS + clearance_margin
                dx = abs(x - ox)
                dy = abs(y - oy)
                if dx <= min_half and dy <= min_half:
                    msg = (
                        f"Target coordinate ({x:.2f}, {y:.2f}) falls INSIDE or TOO CLOSE to obstacle '{name}'!\n"
                        f"  Obstacle Center: ({ox:.2f}, {oy:.2f}) | Box Half-Width: {half_size:.2f} m\n"
                        f"  Offsets: (|dx|={dx:.2f}m, |dy|={dy:.2f}m) <= Minimum Required Clearance: {min_half:.2f} m"
                    )
                    return False, msg

        return True, ""

    def wait_for_sensors(self, timeout_sec: float = 10.0) -> bool:
        """Wait until /scan and /odom topics have published data."""
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_scan is not None and self.latest_odom is not None:
                return True
        return False

    def navigate_to_goal(self, goal_x: float, goal_y: float) -> str:
        """Run closed-loop PPO navigation to the target goal."""
        cur_x, cur_y, cur_theta = self.get_current_pose()
        self.controller.reset(
            init_x=cur_x,
            init_y=cur_y,
            init_theta=cur_theta,
            goal_x=goal_x,
            goal_y=goal_y,
        )

        rate_sec = 0.20
        start_dist = math.hypot(goal_x - cur_x, goal_y - cur_y)
        print(f"\n[Nav] Starting navigation: ({cur_x:.2f}, {cur_y:.2f}) -> Goal ({goal_x:.2f}, {goal_y:.2f}) | Dist: {start_dist:.2f}m")

        try:
            while rclpy.ok() and not self.controller.is_done:
                t0 = time.monotonic()
                rclpy.spin_once(self, timeout_sec=0.01)

                # Check sensor health
                if time.monotonic() - self.last_scan_time > 1.0:
                    print("[Nav] WARNING: Scan timeout! Halting.")
                    self.stop_robot()
                    time.sleep(rate_sec)
                    continue

                ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
                v_meas = float(self.latest_odom.twist.twist.linear.x)
                omega_meas = float(self.latest_odom.twist.twist.angular.z)

                cmd_v, cmd_omega, is_done, outcome, info = self.controller.step(ranges, v_meas, omega_meas)

                # Publish velocity
                cmd = Twist()
                cmd.linear.x = float(cmd_v)
                cmd.angular.z = float(cmd_omega)
                self.cmd_pub.publish(cmd)

                # Telemetry print every 5 steps
                if self.controller.step_count % 5 == 0 or is_done:
                    c_x, c_y, c_th = self.get_current_pose()
                    rem_dist = math.hypot(goal_x - c_x, goal_y - c_y)
                    valid_r = ranges[(~np.isnan(ranges)) & (~np.isinf(ranges)) & (ranges > 0.0)]
                    min_r = float(np.min(valid_r)) if len(valid_r) > 0 else 0.0
                    print(
                        f"  Step {self.controller.step_count:3d} | "
                        f"Pose: ({c_x:5.2f}, {c_y:5.2f}) | "
                        f"Dist: {rem_dist:4.2f}m | "
                        f"Min LiDAR: {min_r:4.2f}m | "
                        f"Cmd: (v={cmd_v:.2f}, w={cmd_omega:+.2f})"
                    )

                if is_done:
                    break

                elapsed = time.monotonic() - t0
                if elapsed < rate_sec:
                    time.sleep(rate_sec - elapsed)

        except KeyboardInterrupt:
            print("\n[Nav] Navigation interrupted by user (Ctrl+C). Halting.")
            self.stop_robot()
            return "CANCELLED"

        self.stop_robot()
        return self.controller.outcome


def setup_gazebo_obstacles(layout_name: str, port: int = 11345, domain_id: int = 30) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    """Spawn visual obstacles in Gazebo and position robot cleanly at safe start."""
    try:
        from scripts.demo_with_obstacles import PRESET_LAYOUTS, teleport_entities_in_gazebo
    except ImportError:
        try:
            from demo_with_obstacles import PRESET_LAYOUTS, teleport_entities_in_gazebo
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "demo_with_obstacles",
                REPO_ROOT / "scripts" / "demo_with_obstacles.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            PRESET_LAYOUTS = mod.PRESET_LAYOUTS
            teleport_entities_in_gazebo = mod.teleport_entities_in_gazebo

    if layout_name in PRESET_LAYOUTS:
        cfg = PRESET_LAYOUTS[layout_name]
        robot_start = cfg.get("robot_start", [0.0, -1.2, math.pi / 2])
        print(f"[Gazebo] Spawning visual obstacle layout: '{layout_name}'...")
        print(f"[Gazebo] Positioning robot at safe layout start pose: ({robot_start[0]:.2f}, {robot_start[1]:.2f})")
        teleport_entities_in_gazebo(
            robot_pose=robot_start,
            active_obstacles=cfg["obstacles"],
            port=port,
            domain_id=domain_id,
        )
        return cfg["obstacles"], robot_start

    return [], None


def main():
    parser = argparse.ArgumentParser(description="Interactive Waypoint Navigation for TurtleBot3 PPO Policy")
    parser.add_argument("--policy", choices=["P4", "P1"], default="P4", help="PPO Policy model to use (P4=Robust, P1=Nominal)")
    parser.add_argument("--obstacles", choices=["center_pillar", "slalom", "hallway_pinch", "none"], default="center_pillar",
                        help="Obstacle layout to position in Gazebo")
    parser.add_argument("--arena-limit", type=float, default=1.85, help="Maximum safe coordinate value (default 1.85 m)")
    parser.add_argument("--collision-thresh", type=float, default=0.13, help="LiDAR safety collision threshold in meters")
    parser.add_argument("--goal-tolerance", type=float, default=0.20, help="Goal arrival tolerance in meters")
    args = parser.parse_args()

    # Model paths
    if args.policy == "P4":
        model_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_vecnormalize.pkl"
    else:
        model_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_vecnormalize.pkl"

    active_obstacles = []
    robot_start = None
    if args.obstacles != "none":
        active_obstacles, robot_start = setup_gazebo_obstacles(args.obstacles)

    if not rclpy.ok():
        rclpy.init()

    navigator = InteractiveNavigator(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        active_obstacles=active_obstacles,
        arena_limit=args.arena_limit,
        goal_tolerance=args.goal_tolerance,
        collision_threshold=args.collision_thresh,
    )

    print("\n[InteractiveNav] Connecting to robot sensor topics (/scan, /odom)...")
    if not navigator.wait_for_sensors(timeout_sec=10.0):
        print("[InteractiveNav] ERROR: Sensor topics not received. Make sure Gazebo is running!")
        navigator.destroy_node()
        rclpy.shutdown()
        return

    # Extra safety: verify current robot pose isn't inside any obstacle
    cur_x, cur_y, _ = navigator.get_current_pose()
    is_safe, obstacle_err = navigator.check_coordinate_validity(cur_x, cur_y)
    if not is_safe:
        print(f"\n[WARNING] Detected robot starting pose ({cur_x:.2f}, {cur_y:.2f}) inside an obstacle!")
        target_reset = robot_start if robot_start else [0.0, -1.2, math.pi / 2]
        print(f"[Gazebo] Teleporting robot to safe start location: ({target_reset[0]:.2f}, {target_reset[1]:.2f})...")
        try:
            from scripts.demo_with_obstacles import teleport_entities_in_gazebo
        except ImportError:
            from demo_with_obstacles import teleport_entities_in_gazebo
        teleport_entities_in_gazebo(robot_pose=target_reset, active_obstacles=active_obstacles)
        time.sleep(0.5)
        # Spin to refresh odometry
        for _ in range(10):
            rclpy.spin_once(navigator, timeout_sec=0.05)

    print("=" * 78)
    print("      TURTLEBOT3 INTERACTIVE WAYPOINT CONTROLLER (PPO POLICY)")
    print("=" * 78)
    print(f"  Selected Policy:         {args.policy} ({model_path.name})")
    print(f"  Arena Physical Limits:   -2.00 m <= X, Y <= +2.00 m")
    print(f"  Safe Navigable Limits:   [-{args.arena_limit:.2f} m, +{args.arena_limit:.2f} m]")
    print(f"  Obstacle Layout:         {args.obstacles}")
    if active_obstacles:
        print("  Active Obstacles in Arena:")
        for obs in active_obstacles:
            geo = OBSTACLE_CATALOG.get(obs['name'], {})
            shape_info = f"radius={geo.get('radius')}m" if geo.get("shape") == "cylinder" else f"size={2*geo.get('half_size', 0.15):.2f}m"
            print(f"    * {obs['name']} ({geo.get('shape', 'obstacle')}, {shape_info}) at ({obs['x']:5.2f}, {obs['y']:5.2f})")
    print("  Type 'q' or 'exit' at any coordinate prompt to stop and quit.")
    print("=" * 78)

    try:
        while rclpy.ok():
            # Get and display current robot pose
            cur_x, cur_y, cur_theta = navigator.get_current_pose()
            deg = math.degrees(cur_theta)
            print(f"\n[CURRENT POSE] -> X: {cur_x:+.2f} m | Y: {cur_y:+.2f} m | Heading: {cur_theta:+.2f} rad ({deg:+.1f}°)")

            # Prompt user for final goal coordinates
            try:
                raw_x = input(f">> Enter target X coordinate ([-{args.arena_limit:.2f}, {args.arena_limit:.2f}]): ").strip()
                if raw_x.lower() in ("q", "quit", "exit"):
                    print("[InteractiveNav] Exiting navigation session.")
                    break
                target_x = float(raw_x)

                raw_y = input(f">> Enter target Y coordinate ([-{args.arena_limit:.2f}, {args.arena_limit:.2f}]): ").strip()
                if raw_y.lower() in ("q", "quit", "exit"):
                    print("[InteractiveNav] Exiting navigation session.")
                    break
                target_y = float(raw_y)
            except ValueError:
                print("\n[ERROR] Invalid number format entered! Bot remains stopped. Please try again.")
                navigator.stop_robot()
                continue

            # Validate arena boundaries and obstacle collision
            is_valid, error_msg = navigator.check_coordinate_validity(target_x, target_y)
            if not is_valid:
                navigator.stop_robot()
                print("\n" + "!" * 78)
                print(f"  [ERROR] INVALID COORDINATE: ({target_x:.2f}, {target_y:.2f})")
                print(f"  {error_msg}")
                print(f"  --> BOT STOPPED IMMEDIATELY (Zero velocity commanded).")
                print("!" * 78)
                continue

            # Check if target is already reached
            dist_to_target = math.hypot(target_x - cur_x, target_y - cur_y)
            if dist_to_target <= args.goal_tolerance:
                print(f"[InteractiveNav] Bot is already at target ({dist_to_target:.2f}m <= {args.goal_tolerance:.2f}m).")
                continue

            # Execute navigation to target
            outcome = navigator.navigate_to_goal(target_x, target_y)

            # Report outcome
            final_x, final_y, _ = navigator.get_current_pose()
            if outcome == "SUCCESS":
                print(f"\n[RESULT] >>> GOAL REACHED SUCCESSFULLY! Final pose: ({final_x:.2f}, {final_y:.2f})")
            elif outcome == "COLLISION":
                print(f"\n[RESULT] >>> SAFETY STOP (Collision avoidance triggered). Final pose: ({final_x:.2f}, {final_y:.2f})")
            elif outcome == "TIMEOUT":
                print(f"\n[RESULT] >>> Step timeout reached. Final pose: ({final_x:.2f}, {final_y:.2f})")
            elif outcome == "CANCELLED":
                print(f"\n[RESULT] >>> Navigation cancelled. Final pose: ({final_x:.2f}, {final_y:.2f})")

    except KeyboardInterrupt:
        print("\n[InteractiveNav] Shutdown requested by user.")
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()
        print("[InteractiveNav] Node stopped cleanly. Robot halted.")


if __name__ == "__main__":
    main()
