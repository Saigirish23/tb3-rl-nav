"""Reward audit script to produce empirical reward traces across 4 baseline behaviors.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 5, 9 Check 7)
Behaviors:
1. Hand-coded reactive controller
2. Random uniform action policy
3. Spin-in-place policy (v=0, omega=0.6)
4. Stationary policy (v=0, omega=0)

Records per-episode total reward, term-by-term breakdown (progress, step penalty,
success reward, collision penalty), episode length, and termination reason.
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from baselines.reactive_controller import ReactiveController


def audit_policies(n_episodes: int = 20):
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    policies = {
        "hand_coded": "ReactiveController",
        "random": "UniformRandomAction",
        "spinning": "SpinInPlace_v0_w06",
        "stationary": "Stationary_v0_w0",
    }

    env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
    controller = ReactiveController()
    rng = np.random.default_rng(42)

    summary = {}
    detailed_traces = {}
    all_step_traces = []

    for pol_name in policies.keys():
        ep_rewards = []
        ep_lengths = []
        outcomes = {"success": 0, "collision": 0, "timeout": 0}
        component_totals = {
            "progress_reward": [],
            "step_penalty": [],
            "goal_reward": [],
            "collision_penalty": [],
        }
        episode_records = []
        step_traces = []

        for ep_idx in range(n_episodes):
            seed = 1_000_000 + ep_idx
            obs, info = env.reset(seed=seed)
            if pol_name == "hand_coded":
                controller.reset()

            total_r = 0.0
            prog_r = 0.0
            step_pen = 0.0
            goal_r = 0.0
            col_pen = 0.0
            steps = 0
            done = False

            while not done:
                if pol_name == "hand_coded":
                    action = controller.act(obs)
                elif pol_name == "random":
                    action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                elif pol_name == "spinning":
                    action = np.array([-1.0, 1.0], dtype=np.float32)  # v=0, omega=+0.6
                elif pol_name == "stationary":
                    action = np.array([-1.0, 0.0], dtype=np.float32)  # v=0, omega=0
                else:
                    raise ValueError(f"Unknown policy {pol_name}")

                # Step env
                # Measure components:
                prev_dist = env.prev_goal_dist
                obs, r, term, trunc, info = env.step(action)
                steps += 1
                done = term or trunc

                # Record the authoritative reward components from the environment.
                components = info.get("reward_components")
                if not isinstance(components, dict):
                    raise RuntimeError("Missing info['reward_components']")

                progress = float(components["progress"])
                step_penalty = float(components["step_penalty"])
                goal_reward = float(components["goal"])
                collision_penalty = float(components["collision"])

                component_sum = (
                    progress + step_penalty + goal_reward + collision_penalty
                )
                reward_error = float(r) - component_sum

                if not np.isclose(float(r), component_sum, rtol=1e-7, atol=1e-6):
                    raise AssertionError(
                        f"Reward mismatch: policy={pol_name}, episode={ep_idx}, "
                        f"step={steps}, reward={r}, component_sum={component_sum}"
                    )

                prog_r += progress
                step_pen += step_penalty
                goal_r += goal_reward
                col_pen += collision_penalty
                total_r += float(r)

                step_traces.append({
                    "policy": pol_name,
                    "episode": ep_idx,
                    "seed": seed,
                    "step": steps,
                    "reward": float(r),
                    "progress_reward": progress,
                    "step_penalty": step_penalty,
                    "goal_reward": goal_reward,
                    "collision_penalty": collision_penalty,
                    "component_sum": float(component_sum),
                    "reward_error": reward_error,
                    "is_success": bool(info.get("is_success", False)),
                    "is_collision": bool(info.get("is_collision", False)),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                })

            ep_rewards.append(total_r)
            ep_lengths.append(steps)
            if info.get("is_success", False):
                outcomes["success"] += 1
                term_reason = "success"
            elif info.get("is_collision", False):
                outcomes["collision"] += 1
                term_reason = "collision"
            else:
                outcomes["timeout"] += 1
                term_reason = "timeout"

            component_totals["progress_reward"].append(prog_r)
            component_totals["step_penalty"].append(step_pen)
            component_totals["goal_reward"].append(goal_r)
            component_totals["collision_penalty"].append(col_pen)

            episode_records.append({
                "episode": ep_idx,
                "seed": seed,
                "steps": steps,
                "reward": float(total_r),
                "progress_reward": float(prog_r),
                "step_penalty": float(step_pen),
                "goal_reward": float(goal_r),
                "collision_penalty": float(col_pen),
                "outcome": term_reason,
            })

        summary[pol_name] = {
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "min_reward": float(np.min(ep_rewards)),
            "max_reward": float(np.max(ep_rewards)),
            "mean_length": float(np.mean(ep_lengths)),
            "outcomes": outcomes,
            "success_rate": outcomes["success"] / n_episodes,
            "collision_rate": outcomes["collision"] / n_episodes,
            "timeout_rate": outcomes["timeout"] / n_episodes,
            "mean_components": {
                "progress_reward": float(np.mean(component_totals["progress_reward"])),
                "step_penalty": float(np.mean(component_totals["step_penalty"])),
                "goal_reward": float(np.mean(component_totals["goal_reward"])),
                "collision_penalty": float(np.mean(component_totals["collision_penalty"])),
            }
        }
        detailed_traces[pol_name] = episode_records
        all_step_traces.extend(step_traces)

    env.close()

    output = {
        "summary": summary,
        "n_episodes": n_episodes,
        "specs": {
            "k_progress": 10.0,
            "c_step": 0.05,
            "r_goal": 100.0,
            "r_col": 100.0,
            "max_episode_steps": 350,
        },
        "traces": detailed_traces,
        "step_traces": all_step_traces,
    }

    out_file = results_dir / "reward_audit_traces.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Audit traces written to {out_file}\n")
    print(f"{'Policy':<15} | {'Mean Reward':<12} | {'Std':<8} | {'Success':<8} | {'Collision':<10} | {'Timeout':<8}")
    print("-" * 75)
    for pol_name, stats in summary.items():
        print(f"{pol_name:<15} | {stats['mean_reward']:>12.2f} | {stats['std_reward']:>8.2f} | {stats['success_rate']:>7.1%} | {stats['collision_rate']:>9.1%} | {stats['timeout_rate']:>7.1%}")

    return output

if __name__ == "__main__":
    audit_policies(n_episodes=20)
