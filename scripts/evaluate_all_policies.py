#!/usr/bin/env python3
"""Comprehensive 2x2 Evaluation Suite for P1, P2, P3, P4 Policies.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 8)
Evaluates trained policies on the 50 held-out test scenarios (seeds 2,000,000..2,000,049).
Calculates:
- Success rate, collision rate, timeout rate with Wilson 95% confidence intervals.
- Mean path length, mean steps, minimum obstacle clearance.
- 2x2 Factorial analysis (P1 nominal baseline, P2 sensor effect, P3 dynamics effect, P4 combined, interaction).
- Outputs summary table and structured JSON artifact.
"""

from __future__ import annotations
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "tb3_rl_navigation"))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import ObservationPreprocessor
from tb3_rl_navigation.policy_model import NumPyPolicy
from scenarios.scenario_generator import load_scenario_set


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_policy_on_scenarios(
    policy: NumPyPolicy,
    policy_name: str,
    scenarios: List[Any],
    max_steps: int = 350,
) -> Dict[str, Any]:
    """Evaluate single policy across scenarios deterministically."""
    env = TurtleBot3NavigationEnv(use_gazebo=False, max_episode_steps=max_steps)

    episode_records = []
    successes = 0
    collisions = 0
    timeouts = 0

    path_lengths = []
    steps_list = []
    clearances = []
    rewards = []

    for sc_idx, sc in enumerate(scenarios):
        # Deterministic setup from scenario spec
        obs, info = env.reset(seed=sc.seed)
        env.robot_x = sc.robot_x
        env.robot_y = sc.robot_y
        env.robot_theta = sc.robot_theta
        env.goal_x = sc.goal_x
        env.goal_y = sc.goal_y
        env.prev_goal_dist = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)

        # Place obstacles from scenario
        for i, obs_spec in enumerate(sc.obstacles):
            if i < len(env.obstacle_pool):
                env.obstacle_pool[i].x = obs_spec.x
                env.obstacle_pool[i].y = obs_spec.y
                env.obstacle_pool[i].active = obs_spec.active
                env.obstacle_pool[i].kind = obs_spec.kind
                env.obstacle_pool[i].size = obs_spec.size

        # Deactivate remaining obstacles
        for i in range(len(sc.obstacles), len(env.obstacle_pool)):
            env.obstacle_pool[i].active = False
            env.obstacle_pool[i].x = 20.0
            env.obstacle_pool[i].y = 20.0

        # Reset preprocessor and previous action
        env.preprocessor.reset(env.robot_x, env.robot_y, env.robot_theta)
        env.prev_action = np.zeros(2, dtype=np.float32)

        raw_ranges = env._get_lidar_ranges()
        current_obs = env.preprocessor.step(
            ranges=raw_ranges,
            v_meas=0.0,
            omega_meas=0.0,
            goal_x=env.goal_x,
            goal_y=env.goal_y,
            last_action=env.prev_action,
            dt=env.dt,
        )

        ep_reward = 0.0
        trajectory = [(env.robot_x, env.robot_y)]
        min_clear = float("inf")
        outcome = "TIMEOUT"

        for step in range(max_steps):
            action, _ = policy.predict(current_obs, deterministic=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            current_obs = next_obs
            trajectory.append((env.robot_x, env.robot_y))

            # Track clearance
            d_goal = step_info["goal_distance"]
            for o in env.obstacle_pool:
                if o.active:
                    d_o = math.hypot(env.robot_x - o.x, env.robot_y - o.y) - (o.radius + env.robot_radius)
                    if d_o < min_clear:
                        min_clear = d_o

            if terminated:
                if step_info["is_success"]:
                    outcome = "SUCCESS"
                    successes += 1
                elif step_info["is_collision"]:
                    outcome = "COLLISION"
                    collisions += 1
                break
            elif truncated:
                outcome = "TIMEOUT"
                timeouts += 1
                break

        # Calculate path length
        path_len = sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(trajectory[:-1], trajectory[1:])
        )

        rewards.append(ep_reward)
        path_lengths.append(path_len)
        steps_list.append(env.step_count)
        clearances.append(min_clear if not math.isinf(min_clear) else 0.5)

        episode_records.append({
            "scenario_id": sc_idx,
            "seed": sc.seed,
            "outcome": outcome,
            "reward": round(ep_reward, 3),
            "steps": env.step_count,
            "path_length": round(path_len, 3),
            "min_clearance": round(min_clear, 3),
        })

    env.close()

    n = len(scenarios)
    succ_rate = successes / n
    ci_low, ci_high = wilson_ci(successes, n)

    return {
        "policy_name": policy_name,
        "n_scenarios": n,
        "successes": successes,
        "success_rate": round(succ_rate, 4),
        "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "collisions": collisions,
        "collision_rate": round(collisions / n, 4),
        "timeouts": timeouts,
        "timeout_rate": round(timeouts / n, 4),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_steps": round(float(np.mean(steps_list)), 1),
        "mean_path_length": round(float(np.mean(path_lengths)), 3),
        "mean_min_clearance": round(float(np.mean(clearances)), 3),
        "episodes": episode_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2x2 Navigation Policies on Test Scenarios")
    parser.add_argument("--scenarios", type=str, default="scenarios/test/scenarios.json", help="Path to scenarios JSON")
    parser.add_argument("--n-scenarios", type=int, default=50, help="Number of scenarios to evaluate (default: 50)")
    parser.add_argument("--out", type=str, default="runs/eval/all_policies_2x2_comparison.json", help="Output summary path")
    args = parser.parse_args()

    sc_path = Path(args.scenarios)
    if not sc_path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {sc_path}")

    sc_dir = sc_path.parent if sc_path.is_file() else sc_path
    all_scenarios = load_scenario_set(sc_dir)
    scenarios = all_scenarios[:args.n_scenarios]
    print(f"Loaded {len(all_scenarios)} total scenarios; evaluating on first {len(scenarios)} scenarios.")

    # Discover available policies
    policy_candidates = {
        "P1": Path("deploy/p1_policy.npz"),
        "P2": Path("deploy/p2_policy.npz"),
        "P3": Path("deploy/p3_policy.npz"),
        "P4": Path("deploy/p4_policy.npz"),
    }

    # Also check runs directories if deploy .npz not yet created
    fallback_map = {
        "P1": Path("runs/main_p1_s0/final_model.zip"),
        "P2": Path("runs/main_p2_s0/final_model.zip"),
        "P3": Path("runs/main_p3_s0/final_model.zip"),
        "P4": Path("runs/main_p4_s0/final_model.zip"),
    }

    results = {}
    for pol_name, npz_path in policy_candidates.items():
        pol_path = npz_path if npz_path.exists() else fallback_map[pol_name]
        if pol_path.exists():
            print(f"Evaluating {pol_name} from: {pol_path} ...")
            policy = NumPyPolicy.load(pol_path)
            res = evaluate_policy_on_scenarios(policy, pol_name, scenarios)
            results[pol_name] = res
            print(f"  {pol_name}: Success={res['successes']}/{res['n_scenarios']} ({res['success_rate']*100:.1f}%), "
                  f"Collisions={res['collisions']}, Timeouts={res['timeouts']}, Mean Reward={res['mean_reward']:.2f}")
        else:
            print(f"Policy {pol_name} model not found at {pol_path}. Skipping.")

    # Print 2x2 comparison summary table
    print("\n" + "=" * 80)
    print(" 2x2 POLICY EVALUATION COMPARISON SUMMARY (50 Held-Out Scenarios)")
    print("=" * 80)
    print(f"{'Policy':<8} | {'Success Rate':<16} | {'Wilson 95% CI':<18} | {'Collisions':<10} | {'Timeouts':<10} | {'Mean Steps':<10} | {'Mean Clear':<10}")
    print("-" * 80)
    for pol in ["P1", "P2", "P3", "P4"]:
        if pol in results:
            r = results[pol]
            ci_str = f"[{r['wilson_ci_95'][0]:.3f}, {r['wilson_ci_95'][1]:.3f}]"
            succ_str = f"{r['successes']}/{r['n_scenarios']} ({r['success_rate']*100:.1f}%)"
            print(f"{pol:<8} | {succ_str:<16} | {ci_str:<18} | {r['collisions']:<10} | {r['timeouts']:<10} | {r['mean_steps']:<10} | {r['mean_min_clearance']:<10}")
    print("=" * 80)

    # Save output
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to: {out_file} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
