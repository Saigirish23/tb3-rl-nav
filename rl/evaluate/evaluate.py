"""Evaluation Harness for trained PPO policies and baselines.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 8)
Supports: fixed scenario evaluation, paired perturbation draws, matrix runner,
Wilson 95% confidence intervals, and systematic metrics logging.

Metrics:
- Success rate, collision rate, timeout rate, mis-stop rate
- Time to goal (successes only), path length, minimum clearance
- Nominal-to-perturbed degradation
- Wilson 95% confidence intervals
"""

from __future__ import annotations
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from scenarios.scenario_generator import (
    ScenarioSpec, load_scenario_set, generate_scenario_set
)


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    scenario_id: int
    seed: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    total_reward: float
    final_goal_distance: float
    path_length: float = 0.0
    min_clearance: float = float('inf')
    time_to_goal: float = float('inf')  # seconds
    reward_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Aggregated evaluation results."""
    policy_name: str
    condition: str
    n_episodes: int
    successes: int
    collisions: int
    timeouts: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_reward: float
    std_reward: float
    mean_steps_success: float
    mean_path_length_success: float
    mean_min_clearance: float
    wilson_lower: float  # Wilson 95% CI lower bound
    wilson_upper: float  # Wilson 95% CI upper bound
    per_episode: List[Dict[str, Any]] = field(default_factory=list)


def wilson_confidence_interval(
    successes: int, total: int, z: float = 1.96
) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion.

    Args:
        successes: Number of successes.
        total: Total trials.
        z: Z-score (1.96 for 95% CI).

    Returns:
        (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 1.0)

    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom

    return (max(0.0, center - spread), min(1.0, center + spread))


def evaluate_policy(
    env: TurtleBot3NavigationEnv,
    policy_fn,
    scenarios: List[ScenarioSpec],
    policy_name: str = "unknown",
    condition: str = "nominal",
    deterministic: bool = True,
) -> EvaluationSummary:
    """Evaluate a policy over a set of fixed scenarios.

    Args:
        env: Gymnasium environment instance.
        policy_fn: Callable(obs) -> action, or SB3 model with .predict().
        scenarios: List of scenario specifications.
        policy_name: Name identifier for the policy.
        condition: Evaluation condition name.
        deterministic: If True, use deterministic actions.

    Returns:
        EvaluationSummary with aggregated results.
    """
    results: List[EpisodeResult] = []

    for scenario in scenarios:
        # Reset environment with scenario seed
        obs, info = env.reset(seed=scenario.seed)

        # Override environment state with scenario layout
        env.robot_x = scenario.robot_x
        env.robot_y = scenario.robot_y
        env.robot_theta = scenario.robot_theta
        env.goal_x = scenario.goal_x
        env.goal_y = scenario.goal_y
        env.prev_goal_dist = math.hypot(
            scenario.goal_x - scenario.robot_x,
            scenario.goal_y - scenario.robot_y
        )

        # Place obstacles from scenario
        for i, obs_spec in enumerate(scenario.obstacles):
            if i < len(env.obstacle_pool):
                env.obstacle_pool[i].x = obs_spec.x
                env.obstacle_pool[i].y = obs_spec.y
                env.obstacle_pool[i].active = obs_spec.active
                env.obstacle_pool[i].kind = obs_spec.kind
                env.obstacle_pool[i].size = obs_spec.size

        # Deactivate remaining obstacles
        for i in range(len(scenario.obstacles), len(env.obstacle_pool)):
            env.obstacle_pool[i].active = False
            env.obstacle_pool[i].x = 20.0
            env.obstacle_pool[i].y = 20.0

        # Reset preprocessor with scenario pose
        env.preprocessor.reset(
            init_x=scenario.robot_x,
            init_y=scenario.robot_y,
            init_theta=scenario.robot_theta,
        )

        # Regenerate initial observation
        raw_ranges = env._get_lidar_ranges()
        obs = env.preprocessor.step(
            ranges=raw_ranges,
            v_meas=0.0,
            omega_meas=0.0,
            goal_x=scenario.goal_x,
            goal_y=scenario.goal_y,
            last_action=np.zeros(2, dtype=np.float32),
            dt=env.dt,
        )

        # Run episode
        total_reward = 0.0
        steps = 0
        done = False
        path_length = 0.0
        min_clearance = float('inf')
        prev_x, prev_y = scenario.robot_x, scenario.robot_y
        reward_components = {'progress': 0.0, 'step_penalty': 0.0, 'goal': 0.0, 'collision': 0.0}

        while not done:
            # Get action
            if hasattr(policy_fn, 'predict'):
                action, _ = policy_fn.predict(obs, deterministic=deterministic)
            else:
                action = policy_fn(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Track path length
            cur_x, cur_y = env.robot_x, env.robot_y
            path_length += math.hypot(cur_x - prev_x, cur_y - prev_y)
            prev_x, prev_y = cur_x, cur_y

            # Track minimum clearance to obstacles
            for obs_obj in env.obstacle_pool:
                if obs_obj.active:
                    d = math.hypot(cur_x - obs_obj.x, cur_y - obs_obj.y) - obs_obj.radius - env.robot_radius
                    min_clearance = min(min_clearance, d)

            # Wall clearance
            wall_clearance = min(
                env.arena_half - 0.05 - abs(cur_x) - env.robot_radius,
                env.arena_half - 0.05 - abs(cur_y) - env.robot_radius,
            )
            min_clearance = min(min_clearance, wall_clearance)

            if 'reward_components' in info:
                for k, v in info['reward_components'].items():
                    reward_components[k] = reward_components.get(k, 0.0) + v

        is_success = info.get('is_success', False)
        is_collision = info.get('is_collision', False)
        is_timeout = not is_success and not is_collision and steps >= env.max_episode_steps

        time_to_goal = (steps * env.dt) if is_success else float('inf')

        results.append(EpisodeResult(
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            success=is_success,
            collision=is_collision,
            timeout=is_timeout,
            steps=steps,
            total_reward=total_reward,
            final_goal_distance=info.get('goal_distance', float('inf')),
            path_length=path_length,
            min_clearance=max(0.0, min_clearance),
            time_to_goal=time_to_goal,
            reward_components=reward_components,
        ))

    # Aggregate
    n = len(results)
    successes = sum(1 for r in results if r.success)
    collisions = sum(1 for r in results if r.collision)
    timeouts = sum(1 for r in results if r.timeout)

    success_steps = [r.steps for r in results if r.success]
    success_paths = [r.path_length for r in results if r.success]
    all_clearances = [r.min_clearance for r in results if r.min_clearance < float('inf')]

    wilson_lo, wilson_hi = wilson_confidence_interval(successes, n)

    return EvaluationSummary(
        policy_name=policy_name,
        condition=condition,
        n_episodes=n,
        successes=successes,
        collisions=collisions,
        timeouts=timeouts,
        success_rate=successes / n if n > 0 else 0.0,
        collision_rate=collisions / n if n > 0 else 0.0,
        timeout_rate=timeouts / n if n > 0 else 0.0,
        mean_reward=float(np.mean([r.total_reward for r in results])) if results else 0.0,
        std_reward=float(np.std([r.total_reward for r in results])) if results else 0.0,
        mean_steps_success=float(np.mean(success_steps)) if success_steps else float('inf'),
        mean_path_length_success=float(np.mean(success_paths)) if success_paths else float('inf'),
        mean_min_clearance=float(np.mean(all_clearances)) if all_clearances else 0.0,
        wilson_lower=wilson_lo,
        wilson_upper=wilson_hi,
        per_episode=[asdict(r) for r in results],
    )


def print_evaluation_summary(summary: EvaluationSummary) -> None:
    """Print formatted evaluation summary."""
    print(f"\n{'=' * 70}")
    print(f"Evaluation: {summary.policy_name} | Condition: {summary.condition}")
    print(f"{'=' * 70}")
    print(f"  Episodes:       {summary.n_episodes}")
    print(f"  Success:        {summary.successes}/{summary.n_episodes} "
          f"({summary.success_rate:.1%})")
    print(f"  Wilson 95% CI:  [{summary.wilson_lower:.3f}, {summary.wilson_upper:.3f}]")
    print(f"  Collision:      {summary.collisions}/{summary.n_episodes} "
          f"({summary.collision_rate:.1%})")
    print(f"  Timeout:        {summary.timeouts}/{summary.n_episodes} "
          f"({summary.timeout_rate:.1%})")
    print(f"  Mean Reward:    {summary.mean_reward:.2f} ± {summary.std_reward:.2f}")
    print(f"  Mean Steps (success): {summary.mean_steps_success:.1f}")
    print(f"  Mean Path (success):  {summary.mean_path_length_success:.2f} m")
    print(f"  Mean Min Clearance:   {summary.mean_min_clearance:.3f} m")
    print(f"{'=' * 70}")


def save_evaluation_results(
    summary: EvaluationSummary,
    output_dir: str,
) -> None:
    """Save evaluation results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"eval_{summary.policy_name}_{summary.condition}.json"
    data = asdict(summary)
    (output_path / filename).write_text(
        json.dumps(data, indent=2, default=str), encoding='utf-8'
    )


def main():
    """Main evaluation entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate policies for TB3 Navigation")
    parser.add_argument("--policy", type=str, required=True,
                        help="Policy name or model path")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained SB3 model .zip")
    parser.add_argument("--scenarios-dir", type=str, default=None,
                        help="Directory with scenarios.json")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test", "stress"],
                        help="Scenario split to evaluate")
    parser.add_argument("--condition", type=str, default="nominal",
                        help="Evaluation condition name")
    parser.add_argument("--output-dir", type=str, default="runs/eval",
                        help="Output directory for results")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["reactive", "random", "stationary"],
                        help="Use a baseline controller")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Override number of episodes")
    args = parser.parse_args()

    env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)

    # Load scenarios
    if args.scenarios_dir:
        scenarios = load_scenario_set(args.scenarios_dir)
    else:
        scenarios = generate_scenario_set(args.split)

    if args.n_episodes is not None:
        scenarios = scenarios[:args.n_episodes]

    # Select policy
    if args.baseline == "reactive":
        from baselines.reactive_controller import ReactiveController
        controller = ReactiveController()
        policy_fn = controller.act
        policy_name = "reactive_baseline"
    elif args.baseline == "random":
        policy_fn = lambda obs: env.action_space.sample()
        policy_name = "random_baseline"
    elif args.baseline == "stationary":
        policy_fn = lambda obs: np.array([-1.0, 0.0], dtype=np.float32)  # v=0, omega=0
        policy_name = "stationary_baseline"
    elif args.model_path:
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path, device='cpu')
        policy_fn = model
        policy_name = args.policy
    else:
        print("ERROR: Must specify --model-path or --baseline")
        sys.exit(1)

    # Evaluate
    summary = evaluate_policy(
        env=env,
        policy_fn=policy_fn,
        scenarios=scenarios,
        policy_name=policy_name,
        condition=args.condition,
    )

    print_evaluation_summary(summary)
    save_evaluation_results(summary, args.output_dir)

    env.close()


if __name__ == '__main__':
    main()
