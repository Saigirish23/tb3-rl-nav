#!/usr/bin/env python3
"""End-to-end integration test: Live Gazebo Classic navigation with standalone PPO inference."""

from __future__ import annotations
import os
import sys
import time
import math
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.train.watchdog import GazeboWatchdog
from deploy.ppo_inference_node import PPOInferenceController

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def run_smoke_test(
    port: int = 11355,
    domain_id: int = 40,
    goal_x: float = 0.80,
    goal_y: float = 0.0,
    max_duration_sec: float = 40.0,
    output_path: str = "runs/smoke_test_gazebo_inference.json"
):
    print("=" * 70)
    print("STARTING LIVE GAZEBO CLASSIC END-TO-END INFERENCE SMOKE TEST")
    print(f"  Gazebo Port:   {port} (isolated from training)")
    print(f"  ROS Domain:    {domain_id} (isolated from training)")
    print(f"  Target Goal:   ({goal_x:.2f}, {goal_y:.2f})")
    print("=" * 70)

    # 1. Start isolated Gazebo instance
    watchdog = GazeboWatchdog(port=port, domain_id=domain_id)
    print(f"[SmokeTest] Spawning Gazebo Classic simulator on port {port}...")
    watchdog.start_simulator()

    # Configure environment for ROS 2 client
    os.environ['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ['ROS_LOCALHOST_ONLY'] = "1"

    if not rclpy.ok():
        rclpy.init()

    node = Node('live_smoke_test_inference_node')

    state = {
        "latest_scan": None,
        "latest_odom": None,
        "scan_count": 0,
        "odom_count": 0,
        "last_scan_time": time.monotonic(),
        "first_scan_time": None,
    }

    def scan_cb(msg: LaserScan):
        state["latest_scan"] = msg
        state["scan_count"] += 1
        now = time.monotonic()
        if state["first_scan_time"] is None:
            state["first_scan_time"] = now
        state["last_scan_time"] = now

    def odom_cb(msg: Odometry):
        state["latest_odom"] = msg
        state["odom_count"] += 1

    qos_scan = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    qos_odom = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

    node.create_subscription(LaserScan, '/scan', scan_cb, qos_scan)
    node.create_subscription(Odometry, '/odom', odom_cb, qos_odom)
    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)

    # 2. Wait for simulation initialization and topics
    print("[SmokeTest] Waiting up to 25s for Gazebo simulator to publish /scan and /odom...")
    start_wait = time.monotonic()
    while time.monotonic() - start_wait < 25.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state["latest_scan"] is not None and state["latest_odom"] is not None:
            print(f"[SmokeTest] Verified topics active! First scan received in {time.monotonic() - start_wait:.1f}s.")
            break
    else:
        print("[SmokeTest] FAILED: Did not receive /scan and /odom within 25s.")
        watchdog.terminate_simulator()
        node.destroy_node()
        rclpy.shutdown()
        return False

    # 3. Check message characteristics and rates
    time.sleep(2.0)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    scan_msg = state["latest_scan"]
    odom_msg = state["latest_odom"]
    num_beams = len(scan_msg.ranges)
    print(f"[SmokeTest] /scan beams: {num_beams} (angle_min={scan_msg.angle_min:.3f}, inc={scan_msg.angle_increment:.4f})")
    print(f"[SmokeTest] Initial Odometry: x={odom_msg.pose.pose.position.x:.3f}, y={odom_msg.pose.pose.position.y:.3f}")

    assert num_beams == 360, f"Expected 360 beams, got {num_beams}"

    # 4. Initialize PPOInferenceController with checkpoint 700k
    model_path = REPO_ROOT / "runs" / "main_p1_s0" / "checkpoints" / "checkpoint_700000_steps.zip"
    vecnorm_path = REPO_ROOT / "runs" / "main_p1_s0" / "checkpoints" / "checkpoint_vecnormalize_700000_steps.pkl"

    controller = PPOInferenceController(
        model_path=str(model_path),
        vecnorm_path=str(vecnorm_path),
        goal_x=goal_x,
        goal_y=goal_y,
        goal_tolerance=0.20,
        collision_threshold=0.16,
        max_steps=200,
        dt=0.20,
    )

    init_x = odom_msg.pose.pose.position.x
    init_y = odom_msg.pose.pose.position.y
    q = odom_msg.pose.pose.orientation
    init_th = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    controller.reset(init_x=init_x, init_y=init_y, init_theta=init_th)

    # 5. Run live navigation control loop
    rate_hz = 5.0
    dt = 1.0 / rate_hz
    start_run = time.monotonic()
    trajectory_log = []
    success = False

    print(f"[SmokeTest] Running live PPO navigation loop to reach ({goal_x}, {goal_y})...")
    try:
        while rclpy.ok() and not controller.is_done and (time.monotonic() - start_run < max_duration_sec):
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.01)

            ranges = np.array(state["latest_scan"].ranges, dtype=np.float32)
            v_meas = float(state["latest_odom"].twist.twist.linear.x)
            omega_meas = float(state["latest_odom"].twist.twist.angular.z)
            cur_odom_x = float(state["latest_odom"].pose.pose.position.x)
            cur_odom_y = float(state["latest_odom"].pose.pose.position.y)

            cmd_v, cmd_omega, is_done, outcome, info = controller.step(ranges, v_meas, omega_meas)

            # Publish command
            cmd = Twist()
            cmd.linear.x = cmd_v
            cmd.angular.z = cmd_omega
            cmd_pub.publish(cmd)

            trajectory_log.append({
                "step": controller.step_count,
                "odom_x": round(cur_odom_x, 3),
                "odom_y": round(cur_odom_y, 3),
                "cmd_v": round(cmd_v, 3),
                "cmd_omega": round(cmd_omega, 3),
                "outcome": outcome,
            })

            if is_done:
                print(f"[SmokeTest] Episode finished with outcome: {outcome} at step {controller.step_count}!")
                success = (outcome == "SUCCESS")
                break

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))

    finally:
        # Halt robot
        for _ in range(5):
            cmd_pub.publish(Twist())
            time.sleep(0.02)

        # Terminate simulator
        watchdog.terminate_simulator()
        node.destroy_node()
        rclpy.shutdown()

    # 6. Save log
    final_odom_x = trajectory_log[-1]["odom_x"] if trajectory_log else 0.0
    final_odom_y = trajectory_log[-1]["odom_y"] if trajectory_log else 0.0
    displacement = math.hypot(final_odom_x - init_x, final_odom_y - init_y)

    test_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
        "outcome": controller.outcome,
        "steps": controller.step_count,
        "init_pose": [init_x, init_y, init_th],
        "final_pose": [final_odom_x, final_odom_y],
        "displacement_meters": round(displacement, 3),
        "target_goal": [goal_x, goal_y],
        "num_trajectory_points": len(trajectory_log),
        "trajectory_head": trajectory_log[:5],
        "trajectory_tail": trajectory_log[-5:],
    }

    out_file = REPO_ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    print("=" * 70)
    print("LIVE SMOKE TEST RESULTS:")
    print(f"  Outcome:      {controller.outcome}")
    print(f"  Success:      {success}")
    print(f"  Steps:        {controller.step_count}")
    print(f"  Displacement: {displacement:.3f} m (started at {init_x:.2f}, ended at {final_odom_x:.2f})")
    print(f"  Report saved: {out_file}")
    print("=" * 70)

    assert displacement > 0.30, f"Robot did not move sufficiently! displacement={displacement}"
    return success


if __name__ == '__main__':
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
