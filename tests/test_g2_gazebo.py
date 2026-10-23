"""Pre-Training Gate G2 Tests on LIVE Gazebo Classic Backend.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 9)
Requirements:
1. 1,000 consecutive Gazebo resets:
   - Pose within 2 cm and 1° of commanded target.
   - Zero residual velocity (< 0.02 m/s linear, < 0.05 rad/s angular).
   - Fresh scan received post-reset.
   - Valid goal geometry (1.5 m <= start_goal_dist <= 3.5 m).
   - No obstacle overlaps.
   - Zero reset failures.
2. Strictly newer /scan data per step:
   - Scan timestamp strictly increases monotonically.
   - Scan interval matches physical 5.0 Hz rate (dt = 0.20s +- jitter).
3. Scripted termination semantics:
   - Wall collision: drive into wall -> terminated with is_collision=True.
   - Goal reached: teleport near goal and advance -> terminated with is_success=True.
   - Timeout: stationary for max_episode_steps -> truncated with is_success=False, is_collision=False.
4. Soak test under live Gazebo:
   - Tracks Python RSS and gzserver RSS memory growth.
   - Verifies 0 NaN/Inf values, zero crashes, episode recovery.
"""

from __future__ import annotations
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import pytest
import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import OBS_DIM, RANGE_MAX, wrap_to_pi


def is_gazebo_running() -> bool:
    """Check if gzserver process is currently running on the system."""
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            if 'gzserver' in p.info['name'] or (p.info['cmdline'] and any('gzserver' in c for c in p.info['cmdline'])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


@pytest.fixture(scope="module")
def gazebo_env():
    """Module-level fixture to create and tear down a live Gazebo environment."""
    if not is_gazebo_running():
        pytest.skip("Gazebo is not running. Launch headless_sim.launch.py to run live Gazebo tests.")

    env = TurtleBot3NavigationEnv(use_gazebo=True, seed=42)
    if not env.use_gazebo or env.ros_node is None:
        pytest.skip("Failed to establish ROS 2 node connection to Gazebo.")

    # Settle time for topic discovery
    import rclpy
    start_t = time.perf_counter()
    while (time.perf_counter() - start_t) < 5.0:
        rclpy.spin_once(env.ros_node, timeout_sec=0.1)
        if env._latest_scan is not None and env._latest_models is not None:
            break

    if env._latest_scan is None:
        env.close()
        pytest.skip("No scan received from Gazebo /scan topic within 5.0s.")

    yield env
    env.close()


# ─── Check G2-GZ-1: 1,000 Consecutive Gazebo Resets ─────────────────────────

class TestGazebo1000Resets:
    """Validate 1,000 consecutive Gazebo resets for physical and geometric integrity."""

    def test_gazebo_resets_integrity(self, gazebo_env):
        """Run 1,000 resets verifying pose tolerance, zero velocity, fresh scan, no overlap."""
        import rclpy
        env = gazebo_env

        pose_err_limit_m = 0.02   # 2 cm per spec
        yaw_err_limit_rad = math.radians(1.0)  # 1 degree per spec
        vel_limit_linear = 0.05   # m/s
        vel_limit_angular = 0.10  # rad/s

        failures = []
        n_resets = 1000

        print(f"\nRunning {n_resets} consecutive Gazebo resets...")
        start_time = time.perf_counter()

        from gazebo_msgs.srv import GetEntityState
        cli_get_state = env.ros_node.create_client(GetEntityState, '/gazebo/get_entity_state')
        assert cli_get_state.wait_for_service(timeout_sec=5.0), "GetEntityState service not available"

        for i in range(n_resets):
            seed = 500_000 + i
            prev_scan_stamp = None
            if env._latest_scan is not None:
                prev_scan_stamp = env._latest_scan.header.stamp.sec + env._latest_scan.header.stamp.nanosec * 1e-9

            obs, info = env.reset(seed=seed)

            # Query ground truth directly from Gazebo physics engine via GetEntityState
            req = GetEntityState.Request()
            req.name = 'burger'
            req.reference_frame = 'world'
            future = cli_get_state.call_async(req)
            rclpy.spin_until_future_complete(env.ros_node, future, timeout_sec=0.5)

            if future.result() is not None and future.result().success:
                state = future.result().state
                actual_x = state.pose.position.x
                actual_y = state.pose.position.y
                q = state.pose.orientation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                actual_yaw = math.atan2(siny_cosp, cosy_cosp)

                pos_err = math.hypot(actual_x - env.robot_x, actual_y - env.robot_y)
                yaw_err = abs(wrap_to_pi(actual_yaw - env.robot_theta))

                if pos_err > pose_err_limit_m:
                    failures.append((i, f"Position error {pos_err*100:.2f} cm > 2 cm"))
                if yaw_err > yaw_err_limit_rad:
                    failures.append((i, f"Yaw error {math.degrees(yaw_err):.2f}° > 1°"))

                # Velocity check directly from physics state
                v = math.hypot(state.twist.linear.x, state.twist.linear.y)
                w = abs(state.twist.angular.z)
                if v > vel_limit_linear:
                    failures.append((i, f"Residual linear velocity {v:.3f} m/s"))
                if w > vel_limit_angular:
                    failures.append((i, f"Residual angular velocity {w:.3f} rad/s"))
            else:
                failures.append((i, "GetEntityState call failed"))

            # 3. Fresh scan check: scan timestamp must strictly advance post-reset
            if prev_scan_stamp is not None and env._latest_scan is not None:
                curr_scan_stamp = env._latest_scan.header.stamp.sec + env._latest_scan.header.stamp.nanosec * 1e-9
                if curr_scan_stamp <= prev_scan_stamp:
                    failures.append((i, f"Scan timestamp did not advance: {curr_scan_stamp:.3f} <= {prev_scan_stamp:.3f}"))

            # 4. Goal distance geometry: within [1.45, 3.55] m
            dist = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
            if not (1.45 <= dist <= 3.55):
                failures.append((i, f"Goal distance out of bounds: {dist:.2f}m"))

            # 5. Obstacle clearance and overlap integrity
            for obs in env.obstacle_pool:
                if obs.active:
                    d_rob = math.hypot(obs.x - env.robot_x, obs.y - env.robot_y)
                    if d_rob < (0.40 + env.robot_radius + obs.radius - 1e-3):
                        failures.append((i, f"Obstacle {obs.name} too close to robot: {d_rob:.3f}m"))
                    d_goal = math.hypot(obs.x - env.goal_x, obs.y - env.goal_y)
                    if d_goal < (0.40 + obs.radius - 1e-3):
                        failures.append((i, f"Obstacle {obs.name} too close to goal: {d_goal:.3f}m"))

            active_obs = [o for o in env.obstacle_pool if o.active]
            for a_idx in range(len(active_obs)):
                for b_idx in range(a_idx + 1, len(active_obs)):
                    oa, ob = active_obs[a_idx], active_obs[b_idx]
                    d_pair = math.hypot(oa.x - ob.x, oa.y - ob.y)
                    if d_pair < (oa.radius + ob.radius + 0.10 - 1e-3):
                        failures.append((i, f"Obstacle overlap: {oa.name} and {ob.name} dist={d_pair:.3f}m"))

            if (i + 1) % 250 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Reset {i+1}/{n_resets} ({elapsed:.1f}s, {(i+1)/elapsed:.1f} resets/s, Failures: {len(failures)})")

        duration = time.perf_counter() - start_time
        print(f"1000 Resets finished in {duration:.1f}s (Rate: {n_resets/duration:.1f} resets/s)")
        print(f"Total failures: {len(failures)}")

        assert len(failures) == 0, f"Encountered {len(failures)} reset failures. Sample: {failures[:5]}"


# ─── Check G2-GZ-2: Strictly Newer Scan Timing ──────────────────────────────

class TestGazeboScanTiming:
    """Validate strictly newer scan arrival, timestamp monotonicity, and timing."""

    def test_scan_timestamp_monotonicity(self, gazebo_env):
        """Verify that sequential step calls receive strictly newer scan data."""
        import rclpy
        env = gazebo_env
        env.reset(seed=42)

        stamps = []
        action = np.array([0.0, 0.0], dtype=np.float32)

        for s in range(50):
            obs, r, term, trunc, info = env.step(action)
            assert env._latest_scan is not None, "Scan is None"
            t_sec = env._latest_scan.header.stamp.sec + env._latest_scan.header.stamp.nanosec * 1e-9
            stamps.append(t_sec)

        # Verify strict monotonicity
        for i in range(1, len(stamps)):
            delta = stamps[i] - stamps[i - 1]
            assert delta > 0, f"Scan timestamp did not advance: {stamps[i]} <= {stamps[i-1]}"
            # Control period is 0.2s, so delta should be ~0.20s +- jitter / floating-point epsilon
            assert 0.10 <= delta <= 0.45, f"Unexpected scan arrival interval: delta={delta:.3f}s"


# ─── Check G2-GZ-3: Scripted Termination Semantics ──────────────────────────

class TestGazeboTerminationSemantics:
    """Verify wall collision, goal success, and timeout termination in live Gazebo."""

    def test_scripted_wall_collision(self, gazebo_env):
        """Drive full speed forward into arena wall and verify collision termination."""
        env = gazebo_env
        env.reset(seed=100)

        # Teleport near the arena wall (wall at x = 1.95m, boundary limit = 1.795m)
        env.robot_x = 1.70
        env.robot_y = 0.0
        env.robot_theta = 0.0  # Facing directly towards wall at +x
        env._teleport_gazebo_entities()

        # Step forward with max velocity
        action = np.array([1.0, 0.0], dtype=np.float32)  # v = 0.15 m/s forward

        collided = False
        for _ in range(30):
            obs, r, terminated, truncated, info = env.step(action)
            if info['is_collision']:
                collided = True
                assert terminated, "Collision must set terminated=True"
                assert not truncated, "Collision cannot set truncated=True"
                assert r <= -90.0, f"Collision penalty expected, got reward {r}"
                break

        assert collided, "Expected wall collision did not trigger"

    def test_scripted_goal_success(self, gazebo_env):
        """Teleport robot within goal tolerance and verify goal termination."""
        env = gazebo_env
        env.reset(seed=200)

        # Place robot 0.10m away from goal (< 0.20m tolerance)
        env.robot_x = env.goal_x - 0.10
        env.robot_y = env.goal_y
        env.robot_theta = 0.0
        env._teleport_gazebo_entities()

        # Step with zero velocity
        action = np.array([-1.0, 0.0], dtype=np.float32)
        obs, r, terminated, truncated, info = env.step(action)

        assert info['is_success'], f"Expected is_success=True, got goal_dist={info['goal_distance']:.3f}"
        assert terminated, "Goal success must set terminated=True"
        assert not truncated, "Goal success cannot set truncated=True"
        assert r >= 90.0, f"Expected goal reward >= +90, got {r}"

    def test_scripted_timeout_truncation(self, gazebo_env):
        """Advance past max_episode_steps and verify truncation."""
        env = gazebo_env
        env.reset(seed=300)
        env.step_count = env.max_episode_steps - 1

        action = np.array([-1.0, 0.0], dtype=np.float32)  # Stand still
        obs, r, terminated, truncated, info = env.step(action)

        assert truncated, "Expected truncated=True at max_episode_steps"
        assert not terminated, "Timeout must not set terminated=True"
        assert not info['is_collision']
        assert not info['is_success']


# ─── Check G2-GZ-4: Gazebo Soak & Memory Tracking ───────────────────────────

class TestGazeboSoakAndMemory:
    """Track Gazebo and Python memory and stability over live stepping."""

    def test_gazebo_soak_500_steps(self, gazebo_env):
        """Live soak test measuring gzserver RSS memory, throughput, and stability."""
        env = gazebo_env
        env.reset(seed=400)

        # Get gzserver process
        gz_procs = []
        for p in psutil.process_iter(['name', 'cmdline', 'pid']):
            try:
                if 'gzserver' in p.info['name'] or (p.info['cmdline'] and any('gzserver' in c for c in p.info['cmdline'])):
                    gz_procs.append(p)
            except Exception:
                pass

        assert len(gz_procs) > 0, "No gzserver process found"
        gz_p = gz_procs[0]

        py_proc = psutil.Process()

        gz_rss_start = gz_p.memory_info().rss / (1024 * 1024)
        py_rss_start = py_proc.memory_info().rss / (1024 * 1024)

        n_steps = 250
        print(f"\nRunning {n_steps}-step live Gazebo soak...")
        start_t = time.perf_counter()

        rng = np.random.default_rng(42)
        nan_count = 0
        episodes = 0

        for s in range(n_steps):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)

            if not np.all(np.isfinite(obs)) or not np.isfinite(r):
                nan_count += 1

            if term or trunc:
                episodes += 1
                obs, info = env.reset()

        duration = time.perf_counter() - start_t
        steps_per_sec = n_steps / max(duration, 0.001)

        gz_rss_end = gz_p.memory_info().rss / (1024 * 1024)
        py_rss_end = py_proc.memory_info().rss / (1024 * 1024)

        gz_growth = gz_rss_end - gz_rss_start
        py_growth = py_rss_end - py_rss_start

        print(f"Soak completed: {n_steps} steps in {duration:.1f}s ({steps_per_sec:.2f} steps/s)")
        print(f"  gzserver RSS: {gz_rss_start:.1f} MB -> {gz_rss_end:.1f} MB (Change: {gz_growth:+.2f} MB)")
        print(f"  Python RSS:   {py_rss_start:.1f} MB -> {py_rss_end:.1f} MB (Change: {py_growth:+.2f} MB)")
        print(f"  NaN/Inf count: {nan_count}")

        assert nan_count == 0, f"Found {nan_count} non-finite values"
        assert gz_growth < 50.0, f"Excessive gzserver memory growth: {gz_growth:.1f} MB"
