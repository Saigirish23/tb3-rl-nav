#!/usr/bin/env python3
"""Unit and Integration Tests for Phase 2 Benchmark Evaluation Suite.

Validates:
- All four policy checkpoint paths resolve and load correctly.
- 43-D observation and 2-D action inference dimensions match.
- All 9 benchmark conditions configure fixtures deterministically.
- Identical scenario seeds produce identical layouts for all policies.
- Full reset state handling between trials.
- Output CSV and JSON serialization schemas.
"""

import sys
import tempfile
import json
import csv
import math
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "tb3_rl_navigation"))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from tb3_rl_navigation.policy_model import NumPyPolicy
from scripts.evaluate_benchmark import BenchmarkSuite, evaluate_trial, run_benchmark, BENCHMARK_CONDITIONS, POLICY_NAMES


def test_all_four_policy_checkpoints_resolve():
    """Verify that P1, P2, P3, P4 checkpoints exist and load without error."""
    for p_name in POLICY_NAMES:
        npz_candidate = REPO_ROOT / f"deploy/{p_name.lower()}_best_policy.npz"
        zip_candidate = REPO_ROOT / f"runs/main_{p_name.lower()}_s0/best_model/best_model.zip"
        target_path = npz_candidate if npz_candidate.exists() else zip_candidate

        assert target_path.exists(), f"Checkpoint path for {p_name} does not exist: {target_path}"
        policy = NumPyPolicy.load(target_path)
        assert policy is not None, f"Failed to load {p_name}"
        assert policy.w0.shape == (128, 43), f"Unexpected W0 shape for {p_name}: {policy.w0.shape}"
        assert policy.w1.shape == (128, 128), f"Unexpected W1 shape for {p_name}: {policy.w1.shape}"
        assert policy.w2.shape == (2, 128), f"Unexpected W2 shape for {p_name}: {policy.w2.shape}"


def test_policy_inference_dimensions():
    """Verify policy inference given arbitrary 43-D observations."""
    pol = NumPyPolicy.load(REPO_ROOT / "deploy/p1_best_policy.npz")
    obs = np.random.uniform(-1.0, 1.0, size=(43,)).astype(np.float32)

    action, _ = pol.predict(obs, deterministic=True)
    assert action.shape == (2,), f"Expected 2-D action, got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of [-1, 1] bounds: {action}"


def test_nine_benchmark_conditions_setup():
    """Verify that all 9 conditions configure valid start/goal poses and active obstacles."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, arena_size=5.0)
    suite = BenchmarkSuite(env)

    for cond in BENCHMARK_CONDITIONS:
        start_x, start_y, start_th, goal_x, goal_y = suite.setup_case(cond, seed=2000000)

        # Start and goal must be within arena boundaries
        assert abs(start_x) < 2.40, f"{cond}: start_x out of bounds"
        assert abs(start_y) < 2.40, f"{cond}: start_y out of bounds"
        assert abs(goal_x) < 2.40, f"{cond}: goal_x out of bounds"
        assert abs(goal_y) < 2.40, f"{cond}: goal_y out of bounds"

        # Start and goal distance must be non-trivial
        sg_dist = math.hypot(goal_x - start_x, goal_y - start_y)
        assert sg_dist >= 1.0, f"{cond}: start-goal distance too small ({sg_dist:.2f}m)"

        # Check fixture specific properties
        if cond == "case1_corridor":
            active_pool = [o for o in env.obstacle_pool if o.active]
            assert len(active_pool) == 2, f"{cond}: Expected 2 corridor walls"
            assert np.isclose(abs(active_pool[0].x - active_pool[1].x), 0.55), f"{cond}: Center-to-center wall spacing mismatch"
        elif cond == "case2_utrap":
            active_pool = [o for o in env.obstacle_pool if o.active]
            assert len(active_pool) == 3, f"{cond}: Expected 3 U-trap walls"
        elif cond == "case3_chicane":
            # Racks 1 & 2 must be inactive
            assert not env.static_obstacles[0].active, f"{cond}: Rack 1 should be inactive"
            assert not env.static_obstacles[1].active, f"{cond}: Rack 2 should be inactive"
        elif cond == "case5_dynamic":
            assert env.obstacle_pool[0].kind == 'cylinder', f"{cond}: Expected cylinder dynamic obstacle"
            assert np.isclose(env.obstacle_pool[0].x, -0.80), f"{cond}: Expected initial x = -0.80"

    env.close()


def test_matched_seed_determinism():
    """Verify that for any seed, all policies encounter identical scenario initialization."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, arena_size=5.0)
    suite = BenchmarkSuite(env)

    seed = 2000042
    for cond in BENCHMARK_CONDITIONS:
        p1_setup = suite.setup_case(cond, seed=seed)
        p1_pool = [(o.name, o.x, o.y, o.active) for o in env.obstacle_pool]

        p4_setup = suite.setup_case(cond, seed=seed)
        p4_pool = [(o.name, o.x, o.y, o.active) for o in env.obstacle_pool]

        assert p1_setup == p4_setup, f"{cond}: Start/goal mismatch across policies for seed {seed}"
        assert p1_pool == p4_pool, f"{cond}: Obstacle layout mismatch across policies for seed {seed}"

    env.close()


def test_output_csv_and_json_schemas():
    """Test full execution of a small multi-policy mock evaluation and verify output schemas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = run_benchmark(
            mode="pilot",
            selected_policies=["P1", "P4"],
            selected_conditions=["case1_corridor", "case5_dynamic"],
            base_seed=2000000,
            out_dir=tmp_dir,
            use_gazebo=False,
        )

        csv_path = Path(tmp_dir) / "raw_episodes.csv"
        json_path = Path(tmp_dir) / "benchmark_summary.json"

        assert csv_path.exists(), "raw_episodes.csv not generated"
        assert json_path.exists(), "benchmark_summary.json not generated"

        # Verify CSV fields
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 4, f"Expected 4 rows (2 cond x 2 pol x 1 seed), got {len(rows)}"
            for r in rows:
                assert "policy" in r
                assert "condition" in r
                assert "seed" in r
                assert "outcome" in r
                assert "completion_time_sec" in r
                assert "penalized_time_sec" in r
                assert "smoothness_tv" in r
                assert "min_clearance_m" in r
                assert "collision_geometric" in r
                assert "collision_gazebo_contact" in r
                assert "min_lidar_range_m" in r
                assert "lidar_proximity_warning" in r
                assert "collision_disagreement" in r

        # Verify JSON structure
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "benchmark_metadata" in data
            assert "per_policy" in data
            assert "P1" in data["per_policy"]
            assert "P4" in data["per_policy"]
            assert "per_condition" in data
            assert "case1_corridor" in data["per_condition"]
            assert "case5_dynamic" in data["per_condition"]


if __name__ == "__main__":
    pytest.main(["-v", str(__file__)])

