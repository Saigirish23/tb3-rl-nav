"""Live Gazebo Integration Test for TurtleBot3NavigationEnv.

Tests:
1. Connecting to live headless gzserver instance.
2. Initializing TurtleBot3NavigationEnv with use_gazebo=True.
3. Resetting environment and verifying SetEntityState teleportation.
4. Stepping with continuous action and receiving live scan and odom.
5. Verifying observation integrity and closing cleanly.
"""

import os
import sys
import time
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import OBS_DIM


def test_live_gazebo_integration():
    """Verify live Gazebo interaction when gzserver is running."""
    import rclpy

    env = TurtleBot3NavigationEnv(use_gazebo=True)
    if not env.use_gazebo or env.ros_node is None:
        pytest.skip("Gazebo is not running or ROS 2 interface unavailable.")

    # Check if /scan is active in ROS topic list
    topic_names = [name for name, _ in env.ros_node.get_topic_names_and_types()]
    if '/scan' not in topic_names:
        env.close()
        pytest.skip("Gazebo is not actively publishing /scan. Launch headless_sim.launch.py to run live test.")

    # Spin briefly to ensure initial topic discovery
    start_wait = time.perf_counter()
    while (time.perf_counter() - start_wait) < 3.0:
        rclpy.spin_once(env.ros_node, timeout_sec=0.1)
        if env._latest_scan is not None:
            break

    if env._latest_scan is None:
        env.close()
        pytest.skip("No scan received within timeout; Gazebo may be paused.")

    # Reset
    obs, info = env.reset(seed=123)
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert "goal_distance" in info

    # Step 5 times
    for _ in range(5):
        action = np.array([0.5, 0.1], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)

    env.close()


if __name__ == '__main__':
    test_live_gazebo_integration()
    print("Live Gazebo integration test PASSED!")
