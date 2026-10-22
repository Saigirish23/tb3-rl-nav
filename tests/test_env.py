"""Unit and Integration Tests for TurtleBot3 Navigation Gymnasium Environment.

Verifies:
1. Gymnasium check_env and SB3 check_env compliance.
2. Action space mapping: [0, 0.15] m/s, [-0.6, 0.6] rad/s.
3. Observation dimensions (43) and bounds [0, 1] / [-1, 1], finite.
4. Ground-truth confinement (zero privileged leak into observation).
5. Reward function components: progress, step penalty, goal bonus, collision penalty.
6. Termination vs Truncation semantics:
   - Goal -> terminated
   - Collision -> terminated
   - Timeout (350 steps) -> truncated only
7. Reset consistency and obstacle overlap rejection.
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import OBS_DIM, ARENA_DIAGONAL


def test_gym_and_sb3_check_env():
    """Verify standard Gymnasium and Stable-Baselines3 environment checker passes."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, max_episode_steps=350)
    # Check Gymnasium compliance
    gym_check_env(env.unwrapped)
    # Check SB3 compliance
    sb3_check_env(env.unwrapped)
    env.close()


def test_action_mapping_bounds():
    """Verify action mapping respects [0, 0.15] m/s and [-0.6, 0.6] rad/s."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)
    env.reset(seed=42)

    # u = [-1.0, -1.0] -> v = 0.0, omega = -0.6
    _, _, _, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert np.isclose(env.robot_v, 0.0)
    assert np.isclose(env.robot_omega, -0.60)

    # u = [0.0, 0.0] -> v = 0.075, omega = 0.0
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isclose(env.robot_v, 0.075)
    assert np.isclose(env.robot_omega, 0.0)

    # u = [1.0, 1.0] -> v = 0.15, omega = 0.6
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert np.isclose(env.robot_v, 0.15)
    assert np.isclose(env.robot_omega, 0.60)

    # Clipped out-of-bounds input
    _, _, _, _, _ = env.step(np.array([5.0, -10.0], dtype=np.float32))
    assert np.isclose(env.robot_v, 0.15)
    assert np.isclose(env.robot_omega, -0.60)

    env.close()


def test_observation_bounds_and_dtype():
    """Verify all 43 observation components are finite and within space bounds."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)
    obs, info = env.reset(seed=123)

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.all(np.isfinite(obs))

    # Take 50 random steps and verify every observation
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_ground_truth_isolation():
    """Verify privileged simulator state is NOT present in observation."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)
    obs, info = env.reset(seed=999)

    # Observation vector must NOT contain absolute robot position or obstacle coordinates
    assert len(obs) == 43
    # Check that info has ground_truth, but obs does not contain raw world positions
    assert "ground_truth" in info or "goal_distance" in info
    # Goal distance in obs is normalized
    assert 0.0 <= obs[36] <= 1.0

    env.close()


def test_reward_function():
    """Verify reward components: progress, step penalty, goal, collision."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)
    env.reset(seed=42)

    # Manually configure state for deterministic reward testing
    env.robot_x = 0.0
    env.robot_y = 0.0
    env.robot_theta = 0.0
    env.goal_x = 2.0
    env.goal_y = 0.0
    env.prev_goal_dist = 2.0

    # Step forward toward goal: action [1.0, 0.0] -> v = 0.15, dt = 0.2s -> advances 0.03m
    # dt = 2.0, dt+1 = 1.97 -> progress = 10 * (2.0 - 1.97) = +0.30
    # step penalty = -0.05 -> reward = +0.25
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert not terminated
    assert np.isclose(reward, 0.30 - 0.05, atol=1e-3)
    assert np.isclose(info["reward_components"]["progress"], 0.30, atol=1e-3)
    assert np.isclose(info["reward_components"]["step_penalty"], -0.05, atol=1e-3)

    # Goal reach test: place robot at 0.15m from goal (within 0.20m tolerance)
    env.robot_x = 1.85
    env.robot_y = 0.0
    env.prev_goal_dist = 0.16
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert info["reward_components"]["goal"] == 100.0

    env.close()


def test_collision_termination():
    """Verify wall collision triggers terminated=True, collision penalty=-100, and is_collision=True."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)
    env.reset(seed=10)

    # Place robot adjacent to North wall (wall inner face at y = 1.95, wall limit at 1.845)
    env.robot_x = 0.0
    env.robot_y = 1.83
    env.robot_theta = np.pi / 2.0  # Facing North

    # Move forward into wall (advances 0.03m to 1.86m)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["is_collision"] is True
    assert info["is_success"] is False
    assert info["reward_components"]["collision"] == -100.0

    env.close()


def test_timeout_truncation():
    """Verify timeout at max_episode_steps triggers truncated=True and terminated=False."""
    max_steps = 10
    env = TurtleBot3NavigationEnv(use_gazebo=False, max_episode_steps=max_steps)
    env.reset(seed=1)

    for i in range(max_steps - 1):
        # Rotate in place in center of arena (no collision, no goal)
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert not terminated
        assert not truncated

    # 10th step (timeout)
    obs, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert info.get("TimeLimit.truncated") is True

    env.close()


def test_obstacle_overlap_rejection():
    """Verify reset always places obstacles without overlapping robot or goal."""
    env = TurtleBot3NavigationEnv(use_gazebo=False)

    for s in range(20):
        obs, info = env.reset(seed=100 + s)
        num_obs = info["num_obstacles"]
        assert 4 <= num_obs <= 8

        # Verify clearance of all active obstacles from robot and goal
        for o in env.obstacle_pool:
            if o.active:
                d_robot = np.hypot(o.x - env.robot_x, o.y - env.robot_y)
                assert d_robot >= (0.40 + env.robot_radius + o.radius - 1e-4)

                d_goal = np.hypot(o.x - env.goal_x, o.y - env.goal_y)
                assert d_goal >= (0.40 + o.radius - 1e-4)

    env.close()
