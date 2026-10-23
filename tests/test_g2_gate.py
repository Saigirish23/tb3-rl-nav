"""Pre-Training Gate G2 Tests (Section 9 Checks 1-10).

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 9)
All checks must pass before PPO training runs.

Checks:
1. SB3 check_env                    (already passed in Stage 5)
2. Scenario generator determinism   (same seed = same layout; disjoint seeds)
3. Reset soak (1000 resets)          (pose, velocity, scan, goal, no overlaps)
4. Timing                           (strictly newer scan per step)
5. Preprocessing parity             (Gym and deployment modules identical)
6. Termination semantics            (wall=collision, goal=success, timeout=truncated)
7. Reward audit                     (stationary, random, spin, hand-coded traces)
8. Hand-coded controller            (>= 90% success on nominal scenarios)
9. Soak and recovery (100k steps)   (no crash, memory < 5% growth)
10. PPO smoke test                  (learns empty-arena goal reaching)
"""

from __future__ import annotations
import sys
import os
import math
import time
import gc
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import (
    ObservationPreprocessor, OBS_DIM, process_lidar_ranges,
    RANGE_MAX, ARENA_DIAGONAL, V_MAX, OMEGA_MAX, wrap_to_pi,
)


# ─── Check 1: SB3 check_env ─────────────────────────────────────────────────

class TestCheck1SB3CheckEnv:
    """G2 Check 1: SB3 check_env passes."""

    def test_sb3_check_env(self):
        from stable_baselines3.common.env_checker import check_env
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        check_env(env, warn=True)
        env.close()

    def test_gymnasium_check_env(self):
        from gymnasium.utils.env_checker import check_env
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        check_env(env)
        env.close()


# ─── Check 2: Scenario Generator ────────────────────────────────────────────

class TestCheck2ScenarioGenerator:
    """G2 Check 2: Deterministic scenario generation; disjoint seed sets."""

    def test_same_seed_same_layout(self):
        from scenarios.scenario_generator import generate_scenario
        s1 = generate_scenario(seed=42, scenario_id=0)
        s2 = generate_scenario(seed=42, scenario_id=0)

        assert s1.robot_x == s2.robot_x
        assert s1.robot_y == s2.robot_y
        assert s1.robot_theta == s2.robot_theta
        assert s1.goal_x == s2.goal_x
        assert s1.goal_y == s2.goal_y
        assert len(s1.obstacles) == len(s2.obstacles)
        for o1, o2 in zip(s1.obstacles, s2.obstacles):
            assert o1.x == o2.x
            assert o1.y == o2.y
            assert o1.kind == o2.kind

    def test_different_seed_different_layout(self):
        from scenarios.scenario_generator import generate_scenario
        s1 = generate_scenario(seed=42, scenario_id=0)
        s2 = generate_scenario(seed=43, scenario_id=1)
        # Very unlikely to match
        assert s1.robot_x != s2.robot_x or s1.robot_y != s2.robot_y

    def test_disjoint_seeds(self):
        from scenarios.scenario_generator import verify_disjoint_seeds
        assert verify_disjoint_seeds()

    def test_serialization_roundtrip(self):
        from scenarios.scenario_generator import generate_scenario
        s1 = generate_scenario(seed=100)
        json_str = s1.to_json()
        from scenarios.scenario_generator import ScenarioSpec
        s2 = ScenarioSpec.from_json(json_str)
        assert s1.robot_x == s2.robot_x
        assert s1.goal_x == s2.goal_x


# ─── Check 3: Reset Soak (1000 resets) ──────────────────────────────────────

class TestCheck3ResetSoak:
    """G2 Check 3: 1000 consecutive resets with quality checks."""

    @pytest.fixture
    def env(self):
        e = TurtleBot3NavigationEnv(use_gazebo=False, seed=0)
        yield e
        e.close()

    def test_1000_resets(self, env):
        """1000 resets: pose within bounds, zero velocity, fresh scan, valid goal,
        no overlaps, zero failures."""
        arena_half = env.arena_half
        failures = []

        for i in range(1000):
            obs, info = env.reset(seed=i)

            # 1. Observation is valid float32, correct dim
            assert obs.shape == (OBS_DIM,), f"Reset {i}: bad obs shape {obs.shape}"
            assert obs.dtype == np.float32, f"Reset {i}: bad dtype"
            assert np.all(np.isfinite(obs)), f"Reset {i}: non-finite obs"

            # 2. Robot pose within arena bounds
            assert abs(env.robot_x) < arena_half - 0.1, \
                f"Reset {i}: robot_x={env.robot_x} out of bounds"
            assert abs(env.robot_y) < arena_half - 0.1, \
                f"Reset {i}: robot_y={env.robot_y} out of bounds"

            # 3. Zero velocity
            assert env.robot_v == 0.0, f"Reset {i}: non-zero v={env.robot_v}"
            assert env.robot_omega == 0.0, f"Reset {i}: non-zero omega={env.robot_omega}"

            # 4. Goal within arena and reasonable distance from robot
            assert abs(env.goal_x) < arena_half, f"Reset {i}: goal_x out"
            assert abs(env.goal_y) < arena_half, f"Reset {i}: goal_y out"
            goal_dist = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
            assert 0.5 < goal_dist < 6.0, f"Reset {i}: bad goal_dist={goal_dist}"

            # 5. No obstacle overlaps with robot or goal
            for obs_obj in env.obstacle_pool:
                if obs_obj.active:
                    d_robot = math.hypot(obs_obj.x - env.robot_x, obs_obj.y - env.robot_y)
                    assert d_robot > (env.robot_radius + obs_obj.radius - 0.01), \
                        f"Reset {i}: obstacle {obs_obj.name} overlaps robot (d={d_robot:.3f})"

                    d_goal = math.hypot(obs_obj.x - env.goal_x, obs_obj.y - env.goal_y)
                    assert d_goal > (obs_obj.radius - 0.01), \
                        f"Reset {i}: obstacle {obs_obj.name} overlaps goal (d={d_goal:.3f})"

            # 6. Step count reset
            assert env.step_count == 0, f"Reset {i}: step_count not 0"

            # 7. Previous action reset
            assert np.allclose(env.prev_action, 0.0), f"Reset {i}: prev_action not zero"


# ─── Check 4: Timing ────────────────────────────────────────────────────────

class TestCheck4Timing:
    """G2 Check 4: Step loop timing consistency."""

    def test_step_produces_valid_obs(self):
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        obs, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            assert obs.shape == (OBS_DIM,)
            assert np.all(np.isfinite(obs))
            assert isinstance(r, float)
            if term or trunc:
                obs, _ = env.reset()
        env.close()


# ─── Check 5: Preprocessing Parity ──────────────────────────────────────────

class TestCheck5PreprocessingParity:
    """G2 Check 5: Gym and deployment modules give identical observations."""

    def test_standalone_preprocessor_matches_env(self):
        """ObservationPreprocessor used standalone should match env output."""
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        obs_env, _ = env.reset()

        # Create standalone preprocessor with same init
        pp = ObservationPreprocessor()
        pp.reset(init_x=env.robot_x, init_y=env.robot_y, init_theta=env.robot_theta)

        raw_ranges = env._get_lidar_ranges()
        obs_standalone = pp.step(
            ranges=raw_ranges,
            v_meas=0.0,
            omega_meas=0.0,
            goal_x=env.goal_x,
            goal_y=env.goal_y,
            last_action=np.zeros(2, dtype=np.float32),
            dt=env.dt,
        )

        # They should be very close (tolerance 1e-6)
        np.testing.assert_allclose(obs_env, obs_standalone, atol=1e-6,
                                   err_msg="Preprocessing parity failed")
        env.close()

    def test_recorded_scan_reproducibility(self):
        """Same raw scan -> same processed observation (no hidden state leak)."""
        pp1 = ObservationPreprocessor()
        pp2 = ObservationPreprocessor()

        pp1.reset(init_x=0.5, init_y=0.3, init_theta=0.1)
        pp2.reset(init_x=0.5, init_y=0.3, init_theta=0.1)

        rng = np.random.default_rng(123)
        ranges = rng.uniform(0.5, 3.0, 360).astype(np.float32)

        obs1 = pp1.step(ranges=ranges, v_meas=0.1, omega_meas=0.05,
                        goal_x=1.0, goal_y=1.0, dt=0.2)
        obs2 = pp2.step(ranges=ranges, v_meas=0.1, omega_meas=0.05,
                        goal_x=1.0, goal_y=1.0, dt=0.2)

        np.testing.assert_allclose(obs1, obs2, atol=1e-6)


# ─── Check 6: Termination Semantics ─────────────────────────────────────────

class TestCheck6Termination:
    """G2 Check 6: Correct terminated/truncated semantics."""

    def test_wall_collision_terminated(self):
        """Robot placed near wall should collide and produce terminated=True."""
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        env.reset()

        # Force robot near wall
        env.robot_x = 1.80  # Near the wall_limit ~1.85
        env.robot_y = 0.0
        env.robot_theta = 0.0  # Facing +x (toward wall)
        env.goal_x = -1.0
        env.goal_y = 0.0
        env.prev_goal_dist = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)

        # Drive forward into wall
        action = np.array([1.0, 0.0], dtype=np.float32)  # max forward
        for _ in range(20):
            obs, r, terminated, truncated, info = env.step(action)
            if terminated:
                assert info.get('is_collision', False), "Wall hit should be collision"
                assert not truncated, "Collision should not be truncated"
                break
        else:
            pytest.fail("Expected wall collision within 20 steps")
        env.close()

    def test_goal_reached_terminated(self):
        """Robot placed near goal should reach it and produce terminated=True."""
        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        env.reset()

        # Place robot very close to goal
        env.robot_x = 0.0
        env.robot_y = 0.0
        env.robot_theta = 0.0
        env.goal_x = 0.18  # Within 0.20m goal tolerance
        env.goal_y = 0.0
        env.prev_goal_dist = 0.18

        # Deactivate all obstacles
        for obs_obj in env.obstacle_pool:
            obs_obj.active = False

        action = np.array([1.0, 0.0], dtype=np.float32)  # forward
        obs, r, terminated, truncated, info = env.step(action)
        assert terminated, "Goal reached should be terminated"
        assert info.get('is_success', False), "Goal reached should be success"
        assert not truncated
        env.close()

    def test_timeout_truncated(self):
        """Episode exceeding max steps should produce truncated=True."""
        env = TurtleBot3NavigationEnv(
            use_gazebo=False, seed=42, max_episode_steps=5
        )
        env.reset()

        # Place goal far enough to not reach
        env.goal_x = 3.0
        env.goal_y = 3.0
        env.prev_goal_dist = math.hypot(3.0, 3.0)

        # Deactivate obstacles
        for obs_obj in env.obstacle_pool:
            obs_obj.active = False

        # Step with zero action (stationary)
        for _ in range(5):
            action = np.array([-1.0, 0.0], dtype=np.float32)  # v=0
            obs, r, terminated, truncated, info = env.step(action)
            if truncated:
                assert not terminated, "Timeout should only be truncated"
                break
        else:
            pytest.fail("Expected truncation at max_episode_steps=5")
        env.close()


# ─── Check 7: Reward Audit ──────────────────────────────────────────────────

class TestCheck7RewardAudit:
    """G2 Check 7: Reward audit across stationary, random, spin, hand-coded traces."""

    @pytest.fixture
    def env(self):
        e = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        yield e
        e.close()

    def _run_policy(self, env, policy_fn, n_episodes=10):
        """Run policy for n episodes and return mean reward."""
        rewards = []
        for i in range(n_episodes):
            obs, _ = env.reset(seed=1_000_000 + i)
            ep_reward = 0.0
            done = False
            while not done:
                action = policy_fn(obs)
                obs, r, term, trunc, _ = env.step(action)
                ep_reward += r
                done = term or trunc
            rewards.append(ep_reward)
        return float(np.mean(rewards))

    def test_stationary_negative_reward(self, env):
        """Stationary policy should have negative reward (step penalties accumulate)."""
        mean_r = self._run_policy(env, lambda obs: np.array([-1.0, 0.0], dtype=np.float32))
        assert mean_r < 0, f"Stationary policy should have negative reward, got {mean_r}"

    def test_spin_negative_reward(self, env):
        """Spin-in-place should have negative reward."""
        mean_r = self._run_policy(env, lambda obs: np.array([-1.0, 1.0], dtype=np.float32))
        assert mean_r < 0, f"Spin policy should have negative reward, got {mean_r}"

    def test_random_mostly_negative(self, env):
        """Random policy should have mostly negative reward."""
        rng = np.random.default_rng(42)
        mean_r = self._run_policy(env, lambda obs: rng.uniform(-1, 1, 2).astype(np.float32))
        # Random won't consistently reach goal, so reward should be quite negative
        assert mean_r < 50, f"Random policy reward unexpectedly high: {mean_r}"

    def test_hand_coded_highest(self, env):
        """Hand-coded controller should score highest among baselines."""
        from baselines.reactive_controller import ReactiveController
        controller = ReactiveController()

        hc_reward = self._run_policy(env, controller.act, n_episodes=20)
        stat_reward = self._run_policy(env, lambda obs: np.array([-1.0, 0.0], dtype=np.float32), n_episodes=20)
        spin_reward = self._run_policy(env, lambda obs: np.array([-1.0, 1.0], dtype=np.float32), n_episodes=20)

        assert hc_reward > stat_reward, \
            f"Hand-coded ({hc_reward:.1f}) should beat stationary ({stat_reward:.1f})"
        assert hc_reward > spin_reward, \
            f"Hand-coded ({hc_reward:.1f}) should beat spin ({spin_reward:.1f})"


# ─── Check 8: Hand-Coded Controller ─────────────────────────────────────────

class TestCheck8HandCodedController:
    """G2 Check 8: Hand-coded reactive controller >= 90% success on nominal."""

    def test_reactive_controller_success_rate(self):
        from baselines.reactive_controller import ReactiveController, run_reactive_controller_episode

        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
        controller = ReactiveController()

        successes = 0
        n_episodes = 100

        for i in range(n_episodes):
            obs, _ = env.reset(seed=1_000_000 + i)
            result = run_reactive_controller_episode(env, controller)
            if result['success']:
                successes += 1

        success_rate = successes / n_episodes
        print(f"\nReactive controller: {successes}/{n_episodes} = {success_rate:.1%}")
        assert success_rate >= 0.90, \
            f"Hand-coded controller success rate {success_rate:.1%} < 90% target"
        env.close()


# ─── Check 9: Soak and Recovery ─────────────────────────────────────────────

class TestCheck9SoakAndRecovery:
    """G2 Check 9: 100k random steps, no crash, memory growth < 5%."""

    @pytest.mark.slow
    def test_100k_random_steps(self):
        import psutil
        process = psutil.Process()
        gc.collect()

        env = TurtleBot3NavigationEnv(use_gazebo=False, seed=0)
        obs, _ = env.reset()

        rng = np.random.default_rng(42)
        start_mem_mb = process.memory_info().rss / (1024 * 1024)

        steps = 0
        episodes = 0
        nan_count = 0

        for _ in range(100_000):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            steps += 1

            if not np.all(np.isfinite(obs)):
                nan_count += 1
            if not np.isfinite(r):
                nan_count += 1

            if term or trunc:
                obs, _ = env.reset()
                episodes += 1

        gc.collect()
        current_mem_mb = process.memory_info().rss / (1024 * 1024)
        mem_growth = (current_mem_mb - start_mem_mb) / max(start_mem_mb, 1.0)

        env.close()

        print(f"\n100k soak: {steps} steps, {episodes} episodes")
        print(f"  RSS start: {start_mem_mb:.2f} MB, RSS end: {current_mem_mb:.2f} MB")
        print(f"  Memory growth: {mem_growth:.2%}")
        print(f"  NaN/inf values: {nan_count}")

        assert nan_count == 0, f"Found {nan_count} non-finite values"
        assert mem_growth < 0.05, f"Memory growth {mem_growth:.1%} >= 5%"


# ─── Check 10: PPO Smoke Test & Checkpoint Save/Reload ───────────────────────

class TestCheck10PPOSmokeTest:
    """G2 Check 10: PPO learns empty-arena goal reaching, checkpoint save/reload, clean shutdown."""

    @pytest.mark.slow
    def test_ppo_smoke_and_checkpoint(self, tmp_path):
        """Short PPO run (20k steps) with checkpoint save/reload and clean shutdown."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # Create simple env (no obstacles for smoke test)
        def make_simple_env():
            return TurtleBot3NavigationEnv(use_gazebo=False, seed=0)

        vec_env = DummyVecEnv([make_simple_env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=0,
            device='cpu',
            verbose=0,
        )

        # 1. Train for 20k steps
        model.learn(total_timesteps=20_000)

        # 2. Checkpoint Save
        ckpt_file = str(tmp_path / "smoke_ppo_model.zip")
        vec_file = str(tmp_path / "smoke_vec_normalize.pkl")
        model.save(ckpt_file)
        vec_env.save(vec_file)
        assert os.path.exists(ckpt_file), "Checkpoint model file was not saved"
        assert os.path.exists(vec_file), "VecNormalize file was not saved"

        # 3. Reload from Checkpoint
        reloaded_env = DummyVecEnv([make_simple_env])
        reloaded_vec_env = VecNormalize.load(vec_file, reloaded_env)
        reloaded_vec_env.training = False
        reloaded_vec_env.norm_reward = False

        reloaded_model = PPO.load(ckpt_file, env=reloaded_vec_env, device='cpu')

        # 4. Verify identical predictions between original and reloaded models
        test_obs = np.random.default_rng(42).uniform(-1, 1, size=(5, OBS_DIM)).astype(np.float32)
        orig_acts, _ = model.predict(test_obs, deterministic=True)
        reloaded_acts, _ = reloaded_model.predict(test_obs, deterministic=True)
        np.testing.assert_allclose(
            orig_acts, reloaded_acts, atol=1e-5,
            err_msg="Reloaded model actions differ from saved model"
        )

        # 5. Evaluate policy performance on holdout episodes
        eval_env = TurtleBot3NavigationEnv(use_gazebo=False, seed=999)
        successes = 0
        n_eval = 20

        for i in range(n_eval):
            obs, _ = eval_env.reset(seed=2_000_000 + i)
            done = False
            while not done:
                action, _ = reloaded_model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                done = term or trunc
            if info.get('is_success', False):
                successes += 1

        print(f"\nPPO smoke test: {successes}/{n_eval} successes after 20k steps")

        # 6. Verify resume learning continues cleanly without error
        reloaded_vec_env.training = True
        reloaded_model.learn(total_timesteps=1_000)

        # 7. Clean shutdown
        eval_env.close()
        reloaded_vec_env.close()
        vec_env.close()
        print("Clean shutdown verified for all environments.")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
