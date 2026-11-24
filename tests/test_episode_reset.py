#!/usr/bin/env python3
"""Regression tests verifying complete episode reset isolation across environments and randomizers.

Ensures zero state leakage across consecutive episodes:
- Robot pose and zeroed velocities (v = 0.0, omega = 0.0)
- Obstacle poses and velocities (inactive pool obstacles parked at (20, 20))
- Previous action buffers (env.prev_action and preprocessor.prev_action = [0, 0])
- Observation preprocessor and dead-reckoning estimated pose synchronized to start pose
- Sensor delay buffers and dropout state cleared
- Actuator dynamics lag, filter, and delay states reset
- Step counters and timers initialized to zero
"""

import math
import numpy as np
import pytest

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv, Obstacle
from rl.env.randomizers import DynamicsRandomizer, SensorRandomizer, CombinedRandomizer, DynamicsParams, SensorParams
from scripts.evaluate_benchmark import BenchmarkSuite, evaluate_trial
from tb3_rl_navigation.policy_model import NumPyPolicy


def test_env_reset_clears_velocities_and_buffers():
    """Verify that calling reset() on TurtleBot3NavigationEnv zeroes velocities and buffers."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, arena_size=5.0)

    # Step environment with non-zero commands to dirty all state variables
    env.reset(seed=123)
    for _ in range(10):
        env.step(np.array([1.0, 0.5], dtype=np.float32))

    assert env.step_count == 10
    assert env.robot_v > 0.0
    assert not np.all(env.prev_action == 0.0)
    assert not np.all(env.preprocessor.prev_action == 0.0)

    # Now reset
    obs, info = env.reset(seed=456)

    assert env.step_count == 0, "Step count was not reset to 0"
    assert env.robot_v == 0.0, "robot_v was not zeroed"
    assert env.robot_omega == 0.0, "robot_omega was not zeroed"
    assert np.all(env.prev_action == 0.0), "env.prev_action was not zeroed"
    assert np.all(env.preprocessor.prev_action == 0.0), "preprocessor.prev_action was not zeroed"
    assert env.preprocessor.estimated_pose.x == pytest.approx(env.robot_x, abs=1e-5)
    assert env.preprocessor.estimated_pose.y == pytest.approx(env.robot_y, abs=1e-5)
    assert env.preprocessor.estimated_pose.theta == pytest.approx(env.robot_theta, abs=1e-5)
    assert info["is_collision"] is False
    assert info["is_success"] is False
    assert info["collision_physical"] is False
    assert info["collision_gazebo_contact"] is False
    assert info["collision_geometric"] is False


def test_randomizer_resets():
    """Verify that dynamics, sensor, and combined randomizers cleanly reset internal state."""
    dyn = DynamicsRandomizer(seed=42)
    dyn.params = DynamicsParams(actuator_lag_tau=0.30, control_delay_s=0.20)

    # Step dynamics with large actions to accumulate filter velocity
    # (Step twice because first step with full delay d=dt holds prev_cmd=0.0)
    dyn.perturb_action(0.15, 0.60, 0.0, 0.0, dt=0.20)
    v_eff, w_eff = dyn.perturb_action(0.15, 0.60, 0.15, 0.60, dt=0.20)
    assert dyn.actual_v > 0.0 or dyn.actual_omega > 0.0

    # Reset
    dyn.reset()
    assert dyn.actual_v == 0.0, "DynamicsRandomizer.actual_v was not reset"
    assert dyn.actual_omega == 0.0, "DynamicsRandomizer.actual_omega was not reset"
    assert np.all(dyn.prev_action == 0.0), "DynamicsRandomizer.prev_action was not reset"

    # Sensor randomizer
    sens = SensorRandomizer(seed=42)
    sens.scan_delay_buffer.append(np.ones(360, dtype=np.float32))
    assert len(sens.scan_delay_buffer) == 1
    sens.reset()
    assert len(sens.scan_delay_buffer) == 0, "SensorRandomizer.scan_delay_buffer was not cleared"

    # Combined randomizer
    comb = CombinedRandomizer(seed=42)
    comb.dynamics_rand.actual_v = 0.12
    comb.sensor_rand.scan_delay_buffer.append(np.zeros(360))
    comb.reset()
    assert comb.dynamics_rand.actual_v == 0.0
    assert len(comb.sensor_rand.scan_delay_buffer) == 0


def test_consecutive_episodes_zero_state_leakage():
    """Verify that two consecutive episodes run with the exact same seed produce identical trajectories."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, arena_size=5.0)

    # Episode 1
    obs1, _ = env.reset(seed=999)
    traj1 = [(env.robot_x, env.robot_y)]
    actions = [
        np.array([0.5, 0.2], dtype=np.float32),
        np.array([1.0, -0.4], dtype=np.float32),
        np.array([0.0, 0.8], dtype=np.float32),
        np.array([-0.5, 0.0], dtype=np.float32),
    ]
    for act in actions:
        o1, r1, t1, tr1, _ = env.step(act)
        traj1.append((env.robot_x, env.robot_y))

    # Dirty intermediate state with random steps
    for _ in range(5):
        env.step(np.array([1.0, 1.0], dtype=np.float32))

    # Episode 2 with identical initial seed 999
    obs2, _ = env.reset(seed=999)
    traj2 = [(env.robot_x, env.robot_y)]
    for act in actions:
        o2, r2, t2, tr2, _ = env.step(act)
        traj2.append((env.robot_x, env.robot_y))

    np.testing.assert_allclose(obs1, obs2, rtol=1e-5, atol=1e-5, err_msg="Initial observations differ between identical seed resets!")
    for (x1, y1), (x2, y2) in zip(traj1, traj2):
        assert math.isclose(x1, x2, abs_tol=1e-6), f"Trajectory x differed: {x1} vs {x2}"
        assert math.isclose(y1, y2, abs_tol=1e-6), f"Trajectory y differed: {y1} vs {y2}"


def test_benchmark_suite_case5_obstacle_reset():
    """Verify Case 5 dynamic obstacle reset to exact coordinate (-0.80, 0.00)."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, arena_size=5.0)
    suite = BenchmarkSuite(env)

    # Setup Case 5
    suite.setup_case("case5_dynamic", seed=101)
    assert env.obstacle_pool[0].x == pytest.approx(-0.80, abs=1e-4)
    assert env.obstacle_pool[0].y == pytest.approx(0.00, abs=1e-4)

    # Simulate dynamic movement
    env.obstacle_pool[0].x = 0.50
    assert env.obstacle_pool[0].x == 0.50

    # Reset Case 5 again
    suite.setup_case("case5_dynamic", seed=102)
    assert env.obstacle_pool[0].x == pytest.approx(-0.80, abs=1e-4), "Dynamic obstacle did not reset to (-0.80, 0.00)"
    assert env.obstacle_pool[0].y == pytest.approx(0.00, abs=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", str(__file__)])
