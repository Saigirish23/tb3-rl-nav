#!/usr/bin/env python3
"""Production ROS 2 Deployment Node for TurtleBot3 LiDAR RL Navigation.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 3, 4, 10)
Design features:
- Scan-synchronous execution: exactly one action per new LaserScan (Zero-Order Hold).
- 43-dim pure NumPy observation preprocessing (tb3_rl_navigation.observation).
- Independent Safety Filter (tb3_rl_navigation.safety) active before publishing /cmd_vel.
- Pure NumPy MLP policy inference engine (tb3_rl_navigation.policy_model) (< 20 us latency).
- Zero PyTorch or Stable-Baselines3 dependency on target hardware (Raspberry Pi).
- Publishes /cmd_vel and /rl_nav/status JSON telemetry.
"""

from __future__ import annotations
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String

from tb3_rl_navigation.observation import ObservationPreprocessor, OBS_DIM, wrap_to_pi
from tb3_rl_navigation.safety import SafetyFilter, SafetyIntervention
from tb3_rl_navigation.policy_model import NumPyPolicy


class TB3DeployNode(Node):
    """ROS 2 Deployment Node for Physical TurtleBot3 Burger and SITL."""

    def __init__(self):
        super().__init__("tb3_deploy_node")

        # Declare parameters
        self.declare_parameter("policy_path", "")
        self.declare_parameter("goal_x", 1.50)
        self.declare_parameter("goal_y", 0.00)
        self.declare_parameter("goal_tolerance", 0.20)
        self.declare_parameter("stale_scan_timeout_sec", 0.40)
        self.declare_parameter("emergency_stop_dist", 0.16)
        self.declare_parameter("control_rate_hz", 5.0)
        self.declare_parameter("max_steps", 350)

        # Retrieve parameters
        self.policy_path_str = self.get_parameter("policy_path").get_parameter_value().string_value
        self.goal_x = float(self.get_parameter("goal_x").get_parameter_value().double_value)
        self.goal_y = float(self.get_parameter("goal_y").get_parameter_value().double_value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").get_parameter_value().double_value)
        stale_timeout = float(self.get_parameter("stale_scan_timeout_sec").get_parameter_value().double_value)
        emergency_dist = float(self.get_parameter("emergency_stop_dist").get_parameter_value().double_value)
        self.dt = 1.0 / float(self.get_parameter("control_rate_hz").get_parameter_value().double_value)
        self.max_steps = int(self.get_parameter("max_steps").get_parameter_value().integer_value)

        # Load Policy Model
        self.policy: Optional[NumPyPolicy] = None
        if self.policy_path_str and Path(self.policy_path_str).exists():
            self.policy = NumPyPolicy.load(self.policy_path_str)
            self.get_logger().info(f"Loaded policy from: {self.policy_path_str}")
        else:
            self.get_logger().warn(f"No valid policy_path provided ('{self.policy_path_str}'). Standing by.")

        # Observation preprocessor and Safety supervisor
        self.preprocessor = ObservationPreprocessor()
        self.safety = SafetyFilter(
            v_min=0.0,
            v_max=0.15,
            omega_max=0.60,
            stale_scan_timeout_sec=stale_timeout,
            emergency_stop_dist=emergency_dist,
        )

        # Runtime state
        self.step_count = 0
        self.latest_v = 0.0
        self.latest_omega = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.is_active = True
        self.is_goal_reached = False

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/rl_nav/status", 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_callback, 10
        )

        # Watchdog timer to ensure safety stop even if scans freeze completely
        self.watchdog_timer = self.create_timer(0.10, self._watchdog_check)

        self.get_logger().info(
            f"TB3 RL Deployment Node Initialized. Target Goal: ({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"Tolerance: {self.goal_tolerance:.2f} m, Timeout: {stale_timeout:.2f} s"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Cache latest measured robot velocities from wheel odometry."""
        self.latest_v = float(msg.twist.twist.linear.x)
        self.latest_omega = float(msg.twist.twist.angular.z)

    def _goal_callback(self, msg: PoseStamped) -> None:
        """Update navigation goal interactively."""
        self.goal_x = float(msg.pose.position.x)
        self.goal_y = float(msg.pose.position.y)
        self.is_goal_reached = False
        self.is_active = True
        self.get_logger().info(f"New navigation goal received: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _scan_callback(self, scan_msg: LaserScan) -> None:
        """Scan-synchronous step loop executed upon arrival of each new scan."""
        if not self.is_active:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.safety.notify_scan_received(now_sec)

        # 1. Check goal condition from estimated pose
        cur_x = self.preprocessor.estimated_pose.x
        cur_y = self.preprocessor.estimated_pose.y
        dist_to_goal = math.hypot(self.goal_x - cur_x, self.goal_y - cur_y)

        if dist_to_goal <= self.goal_tolerance:
            self.is_goal_reached = True
            self.is_active = False
            self._publish_stop("GOAL_REACHED")
            self.get_logger().info(f"SUCCESS: Goal reached (dist: {dist_to_goal:.3f} m <= {self.goal_tolerance:.3f} m)")
            return

        # 2. Check step timeout
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.is_active = False
            self._publish_stop("TIMEOUT")
            self.get_logger().warn(f"TIMEOUT: Reached {self.max_steps} steps without reaching goal.")
            return

        # 3. Preprocess 43-dim observation
        ranges_arr = np.asarray(scan_msg.ranges, dtype=np.float32)
        obs = self.preprocessor.step(
            ranges=ranges_arr,
            v_meas=self.latest_v,
            omega_meas=self.latest_omega,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            last_action=self.prev_action,
            dt=self.dt,
            angle_min=scan_msg.angle_min,
            angle_increment=scan_msg.angle_increment,
        )

        # 4. Policy inference
        if self.policy is None:
            self._publish_stop("NO_POLICY")
            return

        action, _ = self.policy.predict(obs, deterministic=True)
        u = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.prev_action = u.copy()

        # 5. Action mapping: v in [0, 0.15], omega in [-0.60, 0.60]
        raw_cmd_v = float(0.075 * (u[0] + 1.0))
        raw_cmd_omega = float(0.60 * u[1])

        # 6. Safety Filter
        safe_v, safe_omega, intervention, safety_info = self.safety.filter_command(
            cmd_v=raw_cmd_v,
            cmd_omega=raw_cmd_omega,
            raw_ranges=ranges_arr,
            current_time=now_sec,
        )

        # 7. Publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = safe_v
        cmd_msg.angular.z = safe_omega
        self.cmd_vel_pub.publish(cmd_msg)

        # 8. Publish telemetry
        status_data = {
            "step": self.step_count,
            "dist_to_goal": round(dist_to_goal, 3),
            "est_x": round(cur_x, 3),
            "est_y": round(cur_y, 3),
            "raw_cmd_v": round(raw_cmd_v, 3),
            "raw_cmd_omega": round(raw_cmd_omega, 3),
            "safe_v": round(safe_v, 3),
            "safe_omega": round(safe_omega, 3),
            "intervention": intervention.value,
            "total_interventions": self.safety.total_interventions,
        }
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    def _watchdog_check(self) -> None:
        """Periodic watchdog check to stop robot if LiDAR stops streaming."""
        if not self.is_active:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.safety.last_scan_time is not None:
            elapsed = now_sec - self.safety.last_scan_time
            if elapsed > self.safety.stale_scan_timeout_sec:
                self._publish_stop("STALE_SCAN_WATCHDOG")

    def _publish_stop(self, reason: str = "") -> None:
        """Command immediate zero velocity to robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        if reason:
            self.get_logger().debug(f"Commanded stop: {reason}")


def main(args=None):
    rclpy.init(args=args)
    node = TB3DeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, stopping robot.")
    finally:
        node._publish_stop("NODE_SHUTDOWN")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

