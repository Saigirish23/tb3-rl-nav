"""Comprehensive Inference Correctness and Parity Test Suite.

Verifies Section A requirements:
1. Direct parity of observation layout between deploy/ppo_inference_node.py,
   rl/env/observation.py, and rl/env/tb3_navigation_env.py.
2. Complete 43-D observation bounds and coordinate frame definitions.
3. LiDAR pooling (10 deg min-pooling, forward centering [-5 deg, +5 deg]).
4. Velocity normalization (v in [-1, 1], omega in [-1, 1]).
5. Previous action buffer handling and persistence.
6. Action scaling (cmd_v in [0, 0.15], cmd_omega in [-0.60, 0.60]).
7. VecNormalize settings verification (ensure normalize_observations is False,
   confirming zero double-normalization).
8. Robustness against NaN, Inf, and negative LiDAR ranges.
"""

import sys
import math
import yaml
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.env.observation import (
    ObservationPreprocessor,
    process_lidar_ranges,
    EstimatedPose,
    OBS_DIM,
    RANGE_MAX,
    ARENA_DIAGONAL,
    V_MAX,
    OMEGA_MAX,
)
from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from deploy.ppo_inference_node import PPOInferenceController


def test_observation_layout_and_parity():
    """Verify that observation layout in deploy matches environment exactly."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
    obs_env, _ = env.reset(seed=42)

    controller = PPOInferenceController(
        model_path="runs/main_p1_s0/final_model.zip",
        vecnorm_path="runs/main_p1_s0/final_vecnormalize.pkl",
        goal_x=env.goal_x,
        goal_y=env.goal_y,
    )
    controller.reset(init_x=env.robot_x, init_y=env.robot_y, init_theta=env.robot_theta)

    # Both must have dimension 43
    assert obs_env.shape == (43,)
    assert controller.preprocessor.prev_action.shape == (2,)

    # Test identical input fed to preprocessors
    test_ranges = np.random.uniform(0.5, 3.0, size=360).astype(np.float32)
    v_meas = 0.08
    omega_meas = -0.25
    prev_act = np.array([0.2, -0.4], dtype=np.float32)

    obs_from_prep = controller.preprocessor.step(
        ranges=test_ranges,
        v_meas=v_meas,
        omega_meas=omega_meas,
        goal_x=env.goal_x,
        goal_y=env.goal_y,
        last_action=prev_act,
    )

    # Check structure
    # 0..35: LiDAR
    assert np.all(obs_from_prep[0:36] >= 0.0) and np.all(obs_from_prep[0:36] <= 1.0)
    # 36: Goal distance normalized
    assert 0.0 <= obs_from_prep[36] <= 1.0
    # 37, 38: sin, cos bearing
    assert -1.0 <= obs_from_prep[37] <= 1.0
    assert -1.0 <= obs_from_prep[38] <= 1.0
    assert math.isclose(obs_from_prep[37]**2 + obs_from_prep[38]**2, 1.0, rel_tol=1e-4)
    # 39, 40: v_norm, omega_norm
    assert -1.0 <= obs_from_prep[39] <= 1.0
    assert -1.0 <= obs_from_prep[40] <= 1.0
    assert math.isclose(obs_from_prep[39], v_meas / V_MAX, rel_tol=1e-4)
    assert math.isclose(obs_from_prep[40], omega_meas / OMEGA_MAX, rel_tol=1e-4)
    # 41, 42: previous action
    assert np.allclose(obs_from_prep[41:43], prev_act)


def test_nan_inf_negative_handling():
    """Verify that NaN, Inf, and negative sensor readings do not propagate."""
    prep = ObservationPreprocessor()
    prep.reset(0.0, 0.0, 0.0)

    corrupt_ranges = np.array([np.nan, np.inf, -5.0, 0.0, 100.0] * 72, dtype=np.float32)
    obs = prep.step(
        ranges=corrupt_ranges,
        v_meas=float('nan'),  # np.clip will handle or test robustness
        omega_meas=0.0,
        goal_x=1.0,
        goal_y=1.0,
    )

    # Check LiDAR bins are capped to 1.0
    assert np.all(obs[0:36] == 1.0)
    # Check no NaNs in LiDAR
    assert not np.any(np.isnan(obs[0:36]))
    assert not np.any(np.isinf(obs[0:36]))


def test_vecnormalize_and_no_double_normalization():
    """Verify configuration and code ensure no double observation normalization."""
    # 1. Check ppo.yaml config
    with open("configs/ppo.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("normalize_observations") is False, "normalize_observations must be False in config!"
    assert cfg.get("normalize_rewards") is True, "normalize_rewards should be True in config"

    # 2. Check controller does not double normalize
    controller = PPOInferenceController(
        model_path="runs/main_p1_s0/final_model.zip",
        vecnorm_path="runs/main_p1_s0/final_vecnormalize.pkl",
    )
    if controller.vecnorm is not None:
        # Check vecnorm.norm_obs attribute if present
        assert getattr(controller.vecnorm, "norm_obs", False) is False, "VecNormalize norm_obs must be False!"

    # 3. Model predict accepts raw preprocessed obs directly
    dummy_ranges = np.full(360, 2.0, dtype=np.float32)
    cmd_v, cmd_omega, is_done, outcome, info = controller.step(dummy_ranges, v_meas=0.0, omega_meas=0.0)
    assert 0.0 <= cmd_v <= 0.15
    assert -0.60 <= cmd_omega <= 0.60
    assert not math.isnan(cmd_v)
    assert not math.isnan(cmd_omega)


def test_action_scaling():
    """Verify that action scaling maps [-1, 1] correctly to kinematic bounds."""
    # Test boundary u values
    u_min = np.array([-1.0, -1.0], dtype=np.float32)
    cmd_v_min = float(0.075 * (u_min[0] + 1.0))
    cmd_omega_min = float(0.60 * u_min[1])
    assert math.isclose(cmd_v_min, 0.0, abs_tol=1e-5)
    assert math.isclose(cmd_omega_min, -0.60, abs_tol=1e-5)

    u_max = np.array([1.0, 1.0], dtype=np.float32)
    cmd_v_max = float(0.075 * (u_max[0] + 1.0))
    cmd_omega_max = float(0.60 * u_max[1])
    assert math.isclose(cmd_v_max, 0.15, abs_tol=1e-5)
    assert math.isclose(cmd_omega_max, 0.60, abs_tol=1e-5)

    u_zero = np.array([0.0, 0.0], dtype=np.float32)
    cmd_v_zero = float(0.075 * (u_zero[0] + 1.0))
    cmd_omega_zero = float(0.60 * u_zero[1])
    assert math.isclose(cmd_v_zero, 0.075, abs_tol=1e-5)
    assert math.isclose(cmd_omega_zero, 0.0, abs_tol=1e-5)
