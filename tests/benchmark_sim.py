#!/usr/bin/env python3
"""Single-worker Headless Gazebo Simulation Throughput Benchmark.

Measures:
- Wall clock time vs Simulation clock time (Real-Time Factor: RTF)
- Scan message rate and latency
- Control step throughput (steps/sec)
- System resource utilization (CPU, Memory)
"""

import time
import psutil
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist


class SimBenchmark(Node):
    def __init__(self, target_steps=100):
        super().__init__('sim_benchmark')
        self.target_steps = target_steps
        self.step_count = 0

        self.sim_start_time = None
        self.sim_end_time = None
        self.wall_start_time = None
        self.wall_end_time = None

        self.last_scan_stamp = None
        self.scan_count = 0
        self.clock_count = 0

        # Subscriptions
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_clock = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_scan)
        self.sub_clock = self.create_subscription(Clock, '/clock', self.clock_cb, qos_clock)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Process monitor
        self.process = psutil.Process()

    def clock_cb(self, msg: Clock):
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.sim_start_time is None:
            self.sim_start_time = t
        self.sim_end_time = t
        self.clock_count += 1

    def scan_cb(self, msg: LaserScan):
        if self.wall_start_time is None:
            self.wall_start_time = time.perf_counter()

        self.scan_count += 1
        self.step_count += 1

        # Publish a nominal forward twist command
        cmd = Twist()
        cmd.linear.x = 0.10
        cmd.angular.z = 0.05
        self.pub_cmd.publish(cmd)

        if self.step_count >= self.target_steps:
            self.wall_end_time = time.perf_counter()
            self.report()
            raise SystemExit

    def report(self):
        wall_duration = self.wall_end_time - self.wall_start_time
        sim_duration = (self.sim_end_time - self.sim_start_time) if self.sim_start_time and self.sim_end_time else 0.0
        rtf = sim_duration / wall_duration if wall_duration > 0 else 0.0
        steps_per_sec = self.step_count / wall_duration if wall_duration > 0 else 0.0
        scans_per_sec = self.scan_count / wall_duration if wall_duration > 0 else 0.0

        cpu_pct = psutil.cpu_percent(interval=None)
        mem_info = psutil.virtual_memory()

        print("\n" + "=" * 50)
        print(" SINGLE-WORKER HEADLESS BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Total Steps Evaluated  : {self.step_count}")
        print(f"Wall Clock Duration    : {wall_duration:.2f} s")
        print(f"Sim Time Advanced      : {sim_duration:.2f} s")
        print(f"Real-Time Factor (RTF) : {rtf:.2f}x")
        print(f"Control Step Rate      : {steps_per_sec:.2f} steps/s")
        print(f"Scan Publish Rate      : {scans_per_sec:.2f} Hz")
        print(f"Host CPU Usage         : {cpu_pct:.1f}%")
        print(f"Available Host RAM     : {mem_info.available / (1024**3):.2f} GiB / {mem_info.total / (1024**3):.2f} GiB")
        print("=" * 50 + "\n")


def main():
    rclpy.init()
    node = SimBenchmark(target_steps=50)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
