"""Unit tests for Observation Preprocessing and Estimated-Pose Model.

Verifies:
- Zero ROS imports (pure NumPy).
- Exactly 43 observation dimensions.
- LiDAR 10° min-pooling with bin 0 centered on forward.
- Handling of invalid readings (0, inf, NaN, negative).
- Bounded and finite observation values.
- Estimated-pose dead-reckoning integration and goal feature calculations.
- Motion distortion correctness.
"""

import sys
import numpy as np
import pytest

from rl.env.observation import (
    process_lidar_ranges,
    apply_motion_distortion,
    EstimatedPose,
    ObservationPreprocessor,
    wrap_to_pi,
    OBS_DIM,
    NUM_LIDAR_BINS,
    RANGE_MAX,
    ARENA_DIAGONAL,
    V_MAX,
    OMEGA_MAX,
)


def test_zero_ros_imports():
    """Verify that rl.env.observation has zero ROS dependencies."""
    import rl.env.observation as obs_mod
    # Verify no rclpy, ros*, etc. in module globals
    for key, val in vars(obs_mod).items():
        if hasattr(val, '__name__'):
            assert 'rclpy' not in val.__name__, f"Forbidden ROS import found: {val.__name__}"
            assert 'ros' not in val.__name__.lower() or 'ros' == val.__name__, f"Forbidden ROS import: {val.__name__}"
    assert 'rclpy' not in sys.modules or 'rclpy' not in str(obs_mod.__file__)


def test_observation_dimensions():
    """Verify exact 43 dimensions and float32 dtype."""
    preprocessor = ObservationPreprocessor()
    preprocessor.reset(0.0, 0.0, 0.0)

    dummy_ranges = np.full(360, 2.0, dtype=np.float32)
    obs = preprocessor.step(
        ranges=dummy_ranges,
        v_meas=0.10,
        omega_meas=0.20,
        goal_x=2.0,
        goal_y=1.0,
        last_action=np.array([0.5, -0.5], dtype=np.float32),
    )

    assert obs.shape == (43,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_lidar_binning_forward_centering():
    """Verify that bin 0 is centered on forward (spanning [-5°, +5°))."""
    # Create 360 beams, each set to max range except specific beams
    ranges = np.full(360, RANGE_MAX, dtype=np.float32)

    # Beam 0 is at 0° (forward). Set beam 0 to 1.0m.
    ranges[0] = 1.0
    binned = process_lidar_ranges(ranges, angle_min=0.0, angle_increment=np.deg2rad(1.0))
    # Bin 0 should reflect min range 1.0 / 3.5 = 0.2857
    assert np.isclose(binned[0], 1.0 / RANGE_MAX)
    # Other bins should be 1.0
    assert np.all(binned[1:] == 1.0)

    # Test beam at 358° (-2°, within bin 0's [-5°, +5°) span)
    ranges_wrap = np.full(360, RANGE_MAX, dtype=np.float32)
    ranges_wrap[358] = 0.7
    binned_wrap = process_lidar_ranges(ranges_wrap, angle_min=0.0, angle_increment=np.deg2rad(1.0))
    assert np.isclose(binned_wrap[0], 0.7 / RANGE_MAX)

    # Test beam at 10° (should fall into bin 1, which covers [5°, 15°))
    ranges_b1 = np.full(360, RANGE_MAX, dtype=np.float32)
    ranges_b1[10] = 0.5
    binned_b1 = process_lidar_ranges(ranges_b1, angle_min=0.0, angle_increment=np.deg2rad(1.0))
    assert np.isclose(binned_b1[1], 0.5 / RANGE_MAX)
    assert binned_b1[0] == 1.0


def test_lidar_invalid_readings_handling():
    """Verify that invalid readings (0, inf, NaN, negative) map to range_max."""
    ranges = np.array([0.0, np.inf, np.nan, -1.0, 10.0] * 72, dtype=np.float32)
    binned = process_lidar_ranges(ranges)

    assert binned.shape == (36,)
    assert np.all(binned == 1.0)
    assert np.all(np.isfinite(binned))


def test_estimated_pose_integration():
    """Verify dead-reckoning integration and goal distance/bearing features."""
    pose = EstimatedPose(x=0.0, y=0.0, theta=0.0)

    # 1. Forward motion: v=0.10 m/s for 1 second (5 steps of 0.2s)
    for _ in range(5):
        pose.update(v_meas=0.10, omega_meas=0.0, dt=0.20)

    assert np.isclose(pose.x, 0.10, atol=1e-4)
    assert np.isclose(pose.y, 0.0, atol=1e-4)
    assert np.isclose(pose.theta, 0.0, atol=1e-4)

    # 2. Goal straight ahead at (1.10, 0.0): distance should be 1.0m, bearing = 0.0
    dist_norm, sin_b, cos_b, raw_dist = pose.compute_goal_features(goal_x=1.10, goal_y=0.0)
    assert np.isclose(raw_dist, 1.0, atol=1e-4)
    assert np.isclose(dist_norm, 1.0 / ARENA_DIAGONAL, atol=1e-4)
    assert np.isclose(sin_b, 0.0, atol=1e-4)
    assert np.isclose(cos_b, 1.0, atol=1e-4)

    # 3. Pure rotation: rotate 90° (pi/2) counter-clockwise
    # omega = pi / 2 rad/s for 1.0s (5 steps of 0.2s)
    pose.reset(0.0, 0.0, 0.0)
    omega = (np.pi / 2.0)
    for _ in range(5):
        pose.update(v_meas=0.0, omega_meas=omega, dt=0.20)

    assert np.isclose(pose.theta, np.pi / 2.0, atol=1e-3)
    # Goal at (1.0, 0.0) is now to the robot's right (-90° bearing)
    dist_norm, sin_b, cos_b, raw_dist = pose.compute_goal_features(goal_x=1.0, goal_y=0.0)
    assert np.isclose(raw_dist, 1.0, atol=1e-4)
    assert np.isclose(sin_b, -1.0, atol=1e-3)
    assert np.isclose(cos_b, 0.0, atol=1e-3)


def test_motion_distortion():
    """Verify motion distortion skews ranges when moving and is identity when stationary."""
    nominal_ranges = np.full(360, 2.0, dtype=np.float32)

    # Stationary robot: distortion should be identity
    distorted_stat = apply_motion_distortion(nominal_ranges, v=0.0, omega=0.0)
    assert np.allclose(distorted_stat, nominal_ranges)

    # Moving robot: distortion must modify ranges
    distorted_mov = apply_motion_distortion(nominal_ranges, v=0.15, omega=0.30)
    assert not np.allclose(distorted_mov, nominal_ranges)
    assert np.all(np.isfinite(distorted_mov))
    assert np.all(distorted_mov >= 0.0)
    assert np.all(distorted_mov <= RANGE_MAX)


def test_previous_action_buffer():
    """Verify previous action resets to zero and stores last normalized action."""
    prep = ObservationPreprocessor()
    prep.reset()

    # At reset: previous action in obs[41:43] must be [0.0, 0.0]
    dummy_ranges = np.full(360, 2.0, dtype=np.float32)
    obs0 = prep.step(dummy_ranges, 0.0, 0.0, 1.0, 1.0, last_action=None)
    assert np.all(obs0[41:43] == 0.0)

    # After step with action [0.8, -0.4]
    act = np.array([0.8, -0.4], dtype=np.float32)
    obs1 = prep.step(dummy_ranges, 0.05, 0.1, 1.0, 1.0, last_action=act)
    assert np.allclose(obs1[41:43], act)
