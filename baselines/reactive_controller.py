"""Hand-Coded Reactive Controller for TurtleBot3 LiDAR Navigation.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 5)
Uses the same 43-dim observation vector as the RL policy.

Algorithm:
- Turn proportionally to goal bearing, scale forward speed by cos(bearing_error).
- If min range in ±30° front sector < 0.5 m: slow down, turn toward side with larger mean range.
- If min range in front sector < 0.25 m: stop forward motion, rotate in place.
- Target: >= 90% success on nominal scenarios.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from rl.env.observation import V_MAX, OMEGA_MAX, RANGE_MAX, OBS_DIM


class ReactiveController:
    """Pure reactive hand-coded navigation controller.

    Works on the standard 43-dim observation vector:
      obs[0:36]  - 36 min-pooled LiDAR bins (10° each, normalized by range_max)
      obs[36]    - goal distance (normalized by arena diagonal)
      obs[37]    - sin(goal bearing)
      obs[38]    - cos(goal bearing)
      obs[39]    - measured v / V_MAX
      obs[40]    - measured omega / OMEGA_MAX
      obs[41:43] - previous action u in [-1, 1]^2

    Outputs action u in [-1, 1]^2 matching the Gymnasium env mapping:
      v     = 0.075 * (u[0] + 1.0) in [0.0, 0.15] m/s
      omega = 0.60  * u[1]          in [-0.6, 0.6] rad/s
    """

    def __init__(
        self,
        front_sector_deg: float = 30.0,
        obstacle_slow_thresh: float = 0.50,   # meters
        obstacle_stop_thresh: float = 0.25,   # meters
        side_stop_thresh: float = 0.22,       # meters (diagonal safety buffer)
        goal_bearing_gain: float = 2.5,
        max_forward_speed_norm: float = 0.95,
        range_max: float = RANGE_MAX,
    ):
        self.front_sector_deg = front_sector_deg
        self.obstacle_slow_thresh = obstacle_slow_thresh / range_max  # normalized
        self.obstacle_stop_thresh = obstacle_stop_thresh / range_max  # normalized
        self.obstacle_slow_thresh_norm = self.obstacle_slow_thresh
        self.obstacle_stop_thresh_norm = self.obstacle_stop_thresh
        self.side_stop_thresh = side_stop_thresh / range_max          # normalized
        self.goal_bearing_gain = goal_bearing_gain
        self.max_forward_speed_norm = max_forward_speed_norm
        self.range_max = range_max

        # Bin indices for ±30° front sector (Bin 0 = forward, each bin = 10°)
        front_bins = int(front_sector_deg / 10.0)
        self.front_bin_indices_right = list(range(0, front_bins + 1))  # [0, 1, 2, 3]
        self.front_bin_indices_left = list(range(36 - front_bins, 36))  # [33, 34, 35]
        self.front_bin_indices = self.front_bin_indices_right + self.front_bin_indices_left

        # Front diagonal/side sectors (40°-60° left, 300°-320° right)
        self.left_front_bins = [4, 5, 6]
        self.right_front_bins = [30, 31, 32]

        # Full side sectors
        self.left_bin_indices = list(range(1, 18))
        self.right_bin_indices = list(range(19, 36))

        # Stuck & escape recovery state
        self.stuck_steps = 0
        self.escape_steps = 0
        self.escape_dir = 1.0

    def reset(self) -> None:
        """Reset internal controller state between episodes."""
        self.stuck_steps = 0
        self.escape_steps = 0
        self.escape_dir = 1.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute action from 43-dim observation.

        Args:
            obs: (43,) np.ndarray observation vector.

        Returns:
            (2,) np.ndarray action u in [-1, 1]^2.
        """
        assert obs.shape == (OBS_DIM,), f"Expected obs dim {OBS_DIM}, got {obs.shape}"

        lidar_bins = obs[0:36]  # normalized [0, 1]
        sin_bearing = obs[37]
        cos_bearing = obs[38]

        # Goal bearing in radians [-pi, pi]
        bearing = math.atan2(sin_bearing, cos_bearing)

        # Active escape mode
        if self.escape_steps > 0:
            self.escape_steps -= 1
            return np.array([-1.0, 0.9 * self.escape_dir], dtype=np.float32)

        # Obstacle proximity analysis
        min_front = float(np.min(lidar_bins[self.front_bin_indices]))
        min_left_front = float(np.min(lidar_bins[self.left_front_bins]))
        min_right_front = float(np.min(lidar_bins[self.right_front_bins]))

        left_mean = float(np.mean(lidar_bins[self.left_bin_indices]))
        right_mean = float(np.mean(lidar_bins[self.right_bin_indices]))

        # 1. EMERGENCY STOP: obstacle in front sector < 0.25 m
        if min_front < self.obstacle_stop_thresh:
            self.stuck_steps += 1
            if self.stuck_steps > 12:
                # Trigger multi-step escape turn
                self.escape_steps = 20
                self.escape_dir = 1.0 if left_mean >= right_mean else -1.0
                self.stuck_steps = 0
                return np.array([-1.0, 0.9 * self.escape_dir], dtype=np.float32)

            u_v = -1.0  # Stop forward motion
            turn_dir = 1.0 if left_mean >= right_mean else -1.0
            u_omega = 0.8 * turn_dir
            return np.array([u_v, u_omega], dtype=np.float32)

        # 2. DIAGONAL PROXIMITY BUFFER: prevent robot width from grazing obstacles
        if min_left_front < self.side_stop_thresh:
            return np.array([-0.5, -0.8], dtype=np.float32)
        elif min_right_front < self.side_stop_thresh:
            return np.array([-0.5, 0.8], dtype=np.float32)

        # 3. CAUTION ZONE: 0.25 m <= min_front < 0.50 m
        if min_front < self.obstacle_slow_thresh:
            self.stuck_steps = 0
            slow_factor = (min_front - self.obstacle_stop_thresh) / (
                self.obstacle_slow_thresh - self.obstacle_stop_thresh
            )
            slow_factor = max(0.0, min(1.0, slow_factor))

            cos_scale = max(0.0, math.cos(bearing))
            u_v = -1.0 + 2.0 * 0.90 * cos_scale * slow_factor
            if u_v < -0.9:
                u_v = -1.0

            # Decide avoidance direction decisively
            avoid_dir = 1.0 if left_mean >= right_mean else -1.0
            avoid_bias = avoid_dir * (0.4 + 0.6 * (1.0 - slow_factor))

            goal_turn = self.goal_bearing_gain * bearing / np.pi
            avoid_weight = 1.0 - slow_factor

            u_omega = (1.0 - avoid_weight) * goal_turn + avoid_weight * avoid_bias
            if abs(u_omega) < 0.3 and slow_factor < 0.3:
                u_omega = 0.5 * avoid_dir

            return np.array([
                float(np.clip(u_v, -1.0, 1.0)),
                float(np.clip(u_omega, -1.0, 1.0))
            ], dtype=np.float32)

        # 4. CLEAR ZONE: Navigate toward goal
        self.stuck_steps = 0
        cos_scale = max(0.0, math.cos(bearing))
        u_v = -1.0 + 2.0 * self.max_forward_speed_norm * cos_scale
        u_omega = float(np.clip(self.goal_bearing_gain * bearing / np.pi, -1.0, 1.0))
        return np.array([
            float(np.clip(u_v, -1.0, 1.0)),
            u_omega
        ], dtype=np.float32)


def run_reactive_controller_episode(
    env,
    controller: Optional[ReactiveController] = None,
    render: bool = False,
) -> dict:
    """Run one episode with the reactive controller.

    Args:
        env: Gymnasium TurtleBot3NavigationEnv instance.
        controller: ReactiveController instance (default: create new).
        render: Whether to render.

    Returns:
        Dict with episode metrics.
    """
    if controller is None:
        controller = ReactiveController()
    else:
        controller.reset()

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    reward_components = {'progress': 0.0, 'step_penalty': 0.0, 'goal': 0.0, 'collision': 0.0}

    while not done:
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if 'reward_components' in info:
            for k, v in info['reward_components'].items():
                reward_components[k] = reward_components.get(k, 0.0) + v

    return {
        'success': info.get('is_success', False),
        'collision': info.get('is_collision', False),
        'timeout': (not info.get('is_success', False) and not info.get('is_collision', False) and steps >= 350),
        'steps': steps,
        'total_reward': total_reward,
        'final_goal_distance': info.get('goal_distance', float('inf')),
        'reward_components': reward_components,
    }


def benchmark_reactive_controller(
    env,
    n_episodes: int = 50,
    seed_start: int = 1_000_000,
) -> dict:
    """Benchmark the reactive controller over multiple episodes.

    Args:
        env: Gymnasium environment.
        n_episodes: Number of episodes to run.
        seed_start: Starting seed for reproducibility.

    Returns:
        Aggregated benchmark results.
    """
    controller = ReactiveController()
    results = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_start + i)
        episode_result = run_reactive_controller_episode(env, controller)
        episode_result['episode'] = i
        episode_result['seed'] = seed_start + i
        results.append(episode_result)

    successes = sum(1 for r in results if r['success'])
    collisions = sum(1 for r in results if r['collision'])
    timeouts = sum(1 for r in results if r['timeout'])
    success_rate = successes / n_episodes if n_episodes > 0 else 0.0

    success_steps = [r['steps'] for r in results if r['success']]
    mean_success_steps = float(np.mean(success_steps)) if success_steps else float('inf')

    return {
        'n_episodes': n_episodes,
        'successes': successes,
        'collisions': collisions,
        'timeouts': timeouts,
        'success_rate': success_rate,
        'mean_success_steps': mean_success_steps,
        'mean_reward': float(np.mean([r['total_reward'] for r in results])),
        'per_episode': results,
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

    from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv

    env = TurtleBot3NavigationEnv(use_gazebo=False, seed=42)
    result = benchmark_reactive_controller(env, n_episodes=100)

    print(f"\n{'=' * 60}")
    print(f"Reactive Controller Benchmark ({result['n_episodes']} episodes)")
    print(f"{'=' * 60}")
    print(f"  Success Rate: {result['success_rate']:.1%} ({result['successes']}/{result['n_episodes']})")
    print(f"  Collisions:   {result['collisions']}")
    print(f"  Timeouts:     {result['timeouts']}")
    print(f"  Mean Steps (success): {result['mean_success_steps']:.1f}")
    print(f"  Mean Reward:  {result['mean_reward']:.2f}")
    print(f"  Target: >= 90% success")
    print(f"  Status: {'PASSED' if result['success_rate'] >= 0.90 else 'NEEDS TUNING'}")
