#!/usr/bin/env python3
"""Interactive & Visual Demonstration Runner for TB3 PPO Navigation with Obstacles.

Features:
- Teleports realistic obstacles directly into the 4m x 4m arena in Gazebo Classic.
- Sets target goals (e.g. across obstacles) and optionally teleports robot to starting pose.
- Can load pre-defined benchmark test scenarios (from scenarios/test/scenarios.json)
  OR interactive obstacle configurations.
- Runs the trained PPO policy (P4 or P1) with live telemetry and visual tracking.
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Preset Demo Layouts
PRESET_LAYOUTS = {
    "center_pillar": {
        "title": "Central Obstacle Avoidance (Navigate around center column)",
        "robot_start": [0.0, -1.2, math.pi / 2],  # facing north
        "goal": [0.0, 1.2],
        "obstacles": [
            {"name": "obstacle_cyl_1", "x": 0.0, "y": 0.0, "z": 0.25},
            {"name": "obstacle_box_1", "x": 0.7, "y": -0.2, "z": 0.25},
            {"name": "obstacle_box_2", "x": -0.7, "y": 0.2, "z": 0.25},
        ]
    },
    "slalom": {
        "title": "Slalom Corridor (Zig-zag through 4 obstacles)",
        "robot_start": [-1.3, -1.3, math.pi / 4],  # facing north-east
        "goal": [1.3, 1.3],
        "obstacles": [
            {"name": "obstacle_box_1", "x": -0.5, "y": -0.5, "z": 0.25},
            {"name": "obstacle_cyl_2", "x": 0.0, "y": 0.2, "z": 0.25},
            {"name": "obstacle_box_3", "x": 0.6, "y": 0.7, "z": 0.25},
            {"name": "obstacle_cyl_4", "x": -0.2, "y": 0.9, "z": 0.25},
        ]
    },
    "hallway_pinch": {
        "title": "Narrow Constriction (Obstacle pinch-point navigation)",
        "robot_start": [-1.2, 0.0, 0.0],  # facing east
        "goal": [1.2, 0.0],
        "obstacles": [
            {"name": "obstacle_box_1", "x": 0.0, "y": 0.45, "z": 0.25},
            {"name": "obstacle_box_2", "x": 0.0, "y": -0.45, "z": 0.25},
            {"name": "obstacle_cyl_1", "x": 0.6, "y": -0.7, "z": 0.25},
        ]
    }
}

ALL_POOL_OBSTACLES = [
    "obstacle_box_1", "obstacle_box_2", "obstacle_box_3", "obstacle_box_4",
    "obstacle_cyl_1", "obstacle_cyl_2", "obstacle_cyl_3", "obstacle_cyl_4"
]


def teleport_entities_in_gazebo(robot_pose: List[float], active_obstacles: List[Dict[str, Any]], port: int = 11345, domain_id: int = 30):
    """Call Gazebo /gazebo/set_entity_state ROS service to position obstacles & robot."""
    import rclpy
    from gazebo_msgs.srv import SetEntityState
    from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("demo_scenario_teleporter")
    cli = node.create_client(SetEntityState, "/gazebo/set_entity_state")

    print("[Teleporter] Connecting to /gazebo/set_entity_state...")
    if not cli.wait_for_service(timeout_sec=3.0):
        print("[Teleporter] Warning: /gazebo/set_entity_state service not ready. Make sure Gazebo is running!")
        node.destroy_node()
        return False

    # 1. Park all unused pool obstacles far away (x=10.0+)
    active_names = {obs["name"] for obs in active_obstacles}
    for idx, name in enumerate(ALL_POOL_OBSTACLES):
        if name not in active_names:
            req = SetEntityState.Request()
            req.state.name = name
            req.state.pose.position = Point(x=10.0 + (idx * 0.5), y=0.0, z=0.25)
            req.state.reference_frame = "world"
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=0.2)

    # 2. Place active obstacles in arena
    for obs in active_obstacles:
        req = SetEntityState.Request()
        req.state.name = obs["name"]
        req.state.pose.position = Point(x=float(obs["x"]), y=float(obs["y"]), z=float(obs.get("z", 0.25)))
        req.state.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=0.2)
        print(f"  -> Spawned {obs['name']} at ({obs['x']:.2f}, {obs['y']:.2f})")

    # 3. Teleport robot
    if robot_pose:
        rx, ry, rtheta = robot_pose
        req = SetEntityState.Request()
        req.state.name = "burger"
        req.state.pose.position = Point(x=float(rx), y=float(ry), z=0.015)
        half_th = rtheta * 0.5
        req.state.pose.orientation = Quaternion(x=0.0, y=0.0, z=math.sin(half_th), w=math.cos(half_th))
        req.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        req.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        req.state.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=0.5)
        print(f"  -> Teleported TurtleBot3 to ({rx:.2f}, {ry:.2f}, theta={rtheta:.2f} rad)")

    node.destroy_node()
    return True


def run_obstacle_demo():
    parser = argparse.ArgumentParser(description="Demonstrate TurtleBot3 PPO Policy with Visual Obstacles in Gazebo")
    parser.add_argument("--layout", choices=["center_pillar", "slalom", "hallway_pinch", "test_scenario"], default="center_pillar",
                        help="Pre-configured obstacle layout")
    parser.add_argument("--scenario-id", type=int, default=0, help="Scenario index if using test_scenario (0-49)")
    parser.add_argument("--policy", choices=["P4", "P1"], default="P4", help="Policy to deploy (P4=Randomized, P1=Nominal)")
    parser.add_argument("--port", type=int, default=11345, help="Gazebo port")
    parser.add_argument("--domain-id", type=int, default=30, help="ROS Domain ID")
    parser.add_argument("--reset-robot", action="store_true", default=True, help="Teleport robot to layout starting pose")
    args = parser.parse_args()

    # Load Model
    if args.policy == "P4":
        model_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_vecnormalize.pkl"
    else:
        model_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_vecnormalize.pkl"

    # Select Layout
    if args.layout == "test_scenario":
        scen_file = REPO_ROOT / "scenarios" / "test" / "scenarios.json"
        with open(scen_file, "r") as f:
            all_scenarios = json.load(f)
        sc = all_scenarios[args.scenario_id]
        title = f"Test Suite Benchmark Scenario #{args.scenario_id}"
        robot_start = [sc["robot_x"], sc["robot_y"], sc["robot_theta"]]
        goal = [sc["goal_x"], sc["goal_y"]]
        obstacles = [
            {"name": o["name"], "x": o["x"], "y": o["y"]}
            for o in sc["obstacles"] if o.get("active", True)
        ]
    else:
        cfg = PRESET_LAYOUTS[args.layout]
        title = cfg["title"]
        robot_start = cfg["robot_start"]
        goal = cfg["goal"]
        obstacles = cfg["obstacles"]

    print("=" * 78)
    print("  TURTLEBOT3 PPO LIDAR NAVIGATION - OBSTACLE AVOIDANCE LIVE DEMO")
    print("=" * 78)
    print(f"  Demonstration Layout: {title}")
    print(f"  Target Waypoint:       ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"  Active Obstacles:     {len(obstacles)}")
    print(f"  Selected Policy:      {args.policy} ({model_path.name})")
    print("=" * 78)

    # Teleport in Gazebo
    env_vars = os.environ.copy()
    env_vars["GAZEBO_MASTER_URI"] = f"http://localhost:{args.port}"
    env_vars["ROS_DOMAIN_ID"] = str(args.domain_id)
    env_vars["TURTLEBOT3_MODEL"] = "burger"

    print("\n[Demo] Setting up obstacles in Gazebo world...")
    start_pose = robot_start if args.reset_robot else None
    teleport_entities_in_gazebo(start_pose, obstacles, port=args.port, domain_id=args.domain_id)
    time.sleep(1.0)

    # Launch inference node
    print("\n[Demo] Engaging trained policy control...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "deploy" / "ppo_inference_node.py"),
        "--model-path", str(model_path),
        "--goal-x", str(goal[0]),
        "--goal-y", str(goal[1]),
        "--goal-tolerance", "0.20",
    ]
    if vecnorm_path.exists():
        cmd.extend(["--vecnorm-path", str(vecnorm_path)])

    subprocess.run(cmd, env=env_vars, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    run_obstacle_demo()
