#!/usr/bin/env python3
"""Unified Phase 2 Benchmark Evaluation Engine for TurtleBot3 PPO Navigation.

Evaluates trained Gen1 policies (P1, P2, P3, P4) across the 9 defined stress conditions:
- Case 1: Ultra-Narrow Corridor (0.45m width)
- Case 2: Deep Concave U-Trap Escape
- Case 3: Overlapping Baffle Chicane (S-Curve)
- Case 4: Blind 90-Degree L-Junction
- Case 5: Dynamic Cross-Traffic Obstacle (0.20m/s crossing at ~10s, stops before Rack 2)
- Case 6: Stepped Funnel with Bottleneck Throat (0.35m throat)
- Case 7A: Out-of-Distribution Distance-Dependent LiDAR Noise
- Case 7B: Forward 30-Degree Blind Sector (Bins 35, 0, 1 masked)
- Case 8: Out-of-Distribution Actuation Lag, Delay & Gains

Evaluation Modes:
- Pilot Mode: 9 conditions x 4 policies x 1 seed = 36 episodes (Default).
- Benchmark Mode: 9 conditions x 4 policies x 20 matched seeds = 720 episodes.

Outputs:
- Machine-readable raw CSV: <output_dir>/raw_episodes.csv
- Structured summary JSON: <output_dir>/benchmark_summary.json
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

def find_repo_root() -> Path:
    """Find repository root by looking for 'rl' and 'deploy' directories."""
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "rl").exists() and (parent / "deploy").exists():
            return parent
    return Path("/home/saigirish/sim-real-rl")

REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "tb3_rl_navigation"))

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv, Obstacle, DEFAULT_STATIC_WAREHOUSE_OBSTACLES
from rl.env.observation import ObservationPreprocessor, RANGE_MAX
from rl.env.randomizers import SensorRandomizer, DynamicsRandomizer, SensorParams, DynamicsParams, RANGE_MIN
from tb3_rl_navigation.policy_model import NumPyPolicy
from scenarios.scenario_generator import generate_scenario


BENCHMARK_CONDITIONS = [
    "case1_corridor",
    "case2_utrap",
    "case3_chicane",
    "case4_ljunction",
    "case5_dynamic",
    "case6_funnel",
    "case7a_sensor_noise",
    "case7b_blind_sector",
    "case8_actuation_lag",
]

POLICY_NAMES = ["P1", "P2", "P3", "P4"]


@dataclass
class EpisodeResult:
    """Telemetry and outcome data for a single evaluation trial."""
    policy: str
    condition: str
    seed: int
    outcome: str               # 'SUCCESS', 'COLLISION', 'TIMEOUT'
    success: int               # 1 or 0
    collision: int             # 1 or 0 (Primary physical collision outcome)
    timeout: int               # 1 or 0
    collision_gazebo_contact: int   # 1 or 0 (Gazebo physical contact sensor)
    collision_geometric: int        # 1 or 0 (Geometric footprint penetration)
    min_lidar_range_m: float        # Minimum LiDAR range observed across episode
    lidar_proximity_warning: int    # 1 if min LiDAR range <= RANGE_MIN (0.120m), else 0
    collision_disagreement: int     # 1 if Gazebo contact != geometric collision, else 0
    completion_time_sec: float
    penalized_time_sec: float
    steps: int
    path_length_m: float
    smoothness_tv: float       # Time-normalized total variation of angular velocity
    min_clearance_m: float
    final_goal_dist_m: float
    reward: float
    failure_reason: str


class BenchmarkSuite:
    """Manages scenario generation and environment configuration for the 9 stress cases."""

    def __init__(self, env: TurtleBot3NavigationEnv):
        self.env = env
        self.arena_size = env.arena_size
        self.arena_half = env.arena_half
        self.robot_radius = env.robot_radius

    def setup_case(self, condition: str, seed: int) -> Tuple[float, float, float, float, float]:
        """Configure the environment fixtures and return (start_x, start_y, start_th, goal_x, goal_y)."""
        # Default reset of static racks to nominal SDF state
        self.env.static_obstacles = [
            Obstacle('box', 'warehouse_rack_1', (0.40, 1.50), x=-1.15, y=0.65, active=True),
            Obstacle('box', 'warehouse_rack_2', (0.40, 1.50), x=1.15, y=-0.65, active=True),
            Obstacle('box', 'warehouse_shelf_north', (1.40, 0.30), x=0.00, y=2.25, active=True),
            Obstacle('box', 'pallet_stack', (0.50, 0.50), x=-1.95, y=1.95, active=True),
            Obstacle('cylinder', 'storage_container', (0.22,), x=1.95, y=-1.95, active=True),
        ]

        # Reset all movable pool obstacles to inactive/parked
        for obs in self.env.obstacle_pool:
            obs.active = False
            obs.x = 20.0
            obs.y = 20.0

        if condition == "case1_corridor":
            # Case 1: Ultra-Narrow Corridor (0.45m clear width along y in [-1.0, 1.0])
            start_x, start_y, start_th = 0.0, -1.4, math.pi / 2.0
            goal_x, goal_y = 0.0, 1.4

            # Two corridor boundary boxes
            self.env.obstacle_pool[0].kind = 'box'
            self.env.obstacle_pool[0].size = (0.10, 2.00)
            self.env.obstacle_pool[0].x = -0.275  # Inner face at -0.225
            self.env.obstacle_pool[0].y = 0.0
            self.env.obstacle_pool[0].active = True

            self.env.obstacle_pool[1].kind = 'box'
            self.env.obstacle_pool[1].size = (0.10, 2.00)
            self.env.obstacle_pool[1].x = +0.275  # Inner face at +0.225
            self.env.obstacle_pool[1].y = 0.0
            self.env.obstacle_pool[1].active = True

        elif condition == "case2_utrap":
            # Case 2: Deep Concave U-Trap Escape
            start_x, start_y, start_th = 0.0, 0.0, -math.pi / 2.0  # Inside trap facing South
            goal_x, goal_y = 0.0, -1.8                            # Target behind South wall

            # South back wall
            self.env.obstacle_pool[0].kind = 'box'
            self.env.obstacle_pool[0].size = (1.00, 0.20)
            self.env.obstacle_pool[0].x = 0.0
            self.env.obstacle_pool[0].y = -0.50
            self.env.obstacle_pool[0].active = True

            # West side wall
            self.env.obstacle_pool[1].kind = 'box'
            self.env.obstacle_pool[1].size = (0.20, 0.90)
            self.env.obstacle_pool[1].x = -0.40
            self.env.obstacle_pool[1].y = 0.05
            self.env.obstacle_pool[1].active = True

            # East side wall
            self.env.obstacle_pool[2].kind = 'box'
            self.env.obstacle_pool[2].size = (0.20, 0.90)
            self.env.obstacle_pool[2].x = +0.40
            self.env.obstacle_pool[2].y = 0.05
            self.env.obstacle_pool[2].active = True

        elif condition == "case3_chicane":
            # Case 3: Overlapping Baffle Chicane (Isolated layout: Racks 1 & 2 removed)
            self.env.static_obstacles[0].active = False  # Deactivate Rack 1
            self.env.static_obstacles[1].active = False  # Deactivate Rack 2

            start_x, start_y, start_th = 0.0, -1.8, math.pi / 2.0
            goal_x, goal_y = 0.0, 1.8

            # Barrier 1 (South): x in [-2.50, +0.25], y in [-0.50, -0.30]
            self.env.obstacle_pool[0].kind = 'box'
            self.env.obstacle_pool[0].size = (2.70, 0.20)
            self.env.obstacle_pool[0].x = -1.10
            self.env.obstacle_pool[0].y = -0.40
            self.env.obstacle_pool[0].active = True

            # Barrier 2 (North): x in [-0.25, +2.50], y in [+0.30, +0.50]
            self.env.obstacle_pool[1].kind = 'box'
            self.env.obstacle_pool[1].size = (2.70, 0.20)
            self.env.obstacle_pool[1].x = +1.10
            self.env.obstacle_pool[1].y = +0.40
            self.env.obstacle_pool[1].active = True

        elif condition == "case4_ljunction":
            # Case 4: Blind 90-Degree L-Junction (Isolated layout: Racks 1 & 2 removed)
            self.env.static_obstacles[0].active = False  # Deactivate Rack 1
            self.env.static_obstacles[1].active = False  # Deactivate Rack 2

            start_x, start_y, start_th = +0.125, -1.000, math.pi / 2.0
            goal_x, goal_y = -1.000, +0.575

            # Inner Wall 1 (vertical leg): x in [-0.35, -0.15], y in [-1.50, 0.30]
            self.env.obstacle_pool[0].kind = 'box'
            self.env.obstacle_pool[0].size = (0.20, 1.80)
            self.env.obstacle_pool[0].x = -0.25
            self.env.obstacle_pool[0].y = -0.60
            self.env.obstacle_pool[0].active = True

            # Inner Wall 2 (horizontal leg): x in [-0.15, 1.50], y in [0.30, 0.50]
            self.env.obstacle_pool[1].kind = 'box'
            self.env.obstacle_pool[1].size = (1.65, 0.20)
            self.env.obstacle_pool[1].x = 0.675
            self.env.obstacle_pool[1].y = 0.40
            self.env.obstacle_pool[1].active = True

            # Outer boundary channel wall 1 (South approach East face at x = 0.40)
            self.env.obstacle_pool[2].kind = 'box'
            self.env.obstacle_pool[2].size = (0.10, 1.85)
            self.env.obstacle_pool[2].x = 0.45
            self.env.obstacle_pool[2].y = -0.075
            self.env.obstacle_pool[2].active = True

            # Outer boundary channel wall 2 (West exit North face at y = 0.85)
            self.env.obstacle_pool[3].kind = 'box'
            self.env.obstacle_pool[3].size = (1.60, 0.10)
            self.env.obstacle_pool[3].x = -0.40
            self.env.obstacle_pool[3].y = 0.90
            self.env.obstacle_pool[3].active = True

        elif condition == "case5_dynamic":
            # Case 5: Dynamic Obstacle Crossing
            start_x, start_y, start_th = 0.0, -1.5, math.pi / 2.0
            goal_x, goal_y = 0.0, 1.5

            # Dynamic obstacle: cylinder radius 0.15m initially at (-0.80, 0.0)
            # Tangent to Rack 1 East face (x = -0.95m), 0 cm initial clearance
            self.env.obstacle_pool[0].kind = 'cylinder'
            self.env.obstacle_pool[0].size = (0.15,)
            self.env.obstacle_pool[0].x = -0.80
            self.env.obstacle_pool[0].y = 0.00
            self.env.obstacle_pool[0].active = True

        elif condition == "case6_funnel":
            # Case 6: Stepped Funnel with Bottleneck Throat
            start_x, start_y, start_th = 0.0, -1.2, math.pi / 2.0
            goal_x, goal_y = 0.0, 1.2

            # Step 1: y in [-0.50, -0.25], width 0.90m (half-width 0.45)
            self.env.obstacle_pool[0].kind = 'box'
            self.env.obstacle_pool[0].size = (0.20, 0.25)
            self.env.obstacle_pool[0].x = -0.55
            self.env.obstacle_pool[0].y = -0.375
            self.env.obstacle_pool[0].active = True

            self.env.obstacle_pool[1].kind = 'box'
            self.env.obstacle_pool[1].size = (0.20, 0.25)
            self.env.obstacle_pool[1].x = +0.55
            self.env.obstacle_pool[1].y = -0.375
            self.env.obstacle_pool[1].active = True

            # Step 2: y in [-0.25, 0.00], width 0.70m (half-width 0.35)
            self.env.obstacle_pool[2].kind = 'box'
            self.env.obstacle_pool[2].size = (0.20, 0.25)
            self.env.obstacle_pool[2].x = -0.45
            self.env.obstacle_pool[2].y = -0.125
            self.env.obstacle_pool[2].active = True

            self.env.obstacle_pool[3].kind = 'box'
            self.env.obstacle_pool[3].size = (0.20, 0.25)
            self.env.obstacle_pool[3].x = +0.45
            self.env.obstacle_pool[3].y = -0.125
            self.env.obstacle_pool[3].active = True

            # Step 3: y in [0.00, +0.25], width 0.50m (half-width 0.25)
            self.env.obstacle_pool[4].kind = 'box'
            self.env.obstacle_pool[4].size = (0.20, 0.25)
            self.env.obstacle_pool[4].x = -0.35
            self.env.obstacle_pool[4].y = +0.125
            self.env.obstacle_pool[4].active = True

            self.env.obstacle_pool[5].kind = 'box'
            self.env.obstacle_pool[5].size = (0.20, 0.25)
            self.env.obstacle_pool[5].x = +0.35
            self.env.obstacle_pool[5].y = +0.125
            self.env.obstacle_pool[5].active = True

            # Step 4 Throat: y in [+0.25, +0.50], width 0.35m (half-width 0.175)
            self.env.obstacle_pool[6].kind = 'box'
            self.env.obstacle_pool[6].size = (0.20, 0.25)
            self.env.obstacle_pool[6].x = -0.275
            self.env.obstacle_pool[6].y = +0.375
            self.env.obstacle_pool[6].active = True

            self.env.obstacle_pool[7].kind = 'box'
            self.env.obstacle_pool[7].size = (0.20, 0.25)
            self.env.obstacle_pool[7].x = +0.275
            self.env.obstacle_pool[7].y = +0.375
            self.env.obstacle_pool[7].active = True

        else:
            # Cases 7A, 7B, 8: Nominal warehouse scenario generated from seed
            sc = generate_scenario(seed=seed, arena_size=5.0)
            start_x, start_y, start_th = sc.robot_x, sc.robot_y, sc.robot_theta
            goal_x, goal_y = sc.goal_x, sc.goal_y

            for i, obs_spec in enumerate(sc.obstacles):
                if i < len(self.env.obstacle_pool):
                    self.env.obstacle_pool[i].kind = obs_spec.kind
                    self.env.obstacle_pool[i].size = obs_spec.size
                    self.env.obstacle_pool[i].x = obs_spec.x
                    self.env.obstacle_pool[i].y = obs_spec.y
                    self.env.obstacle_pool[i].active = obs_spec.active

        return start_x, start_y, start_th, goal_x, goal_y


def evaluate_trial(
    env: TurtleBot3NavigationEnv,
    suite: BenchmarkSuite,
    policy: NumPyPolicy,
    policy_name: str,
    condition: str,
    seed: int,
    max_steps: int = 500,
    goal_eval_tolerance: float = 0.25,
) -> EpisodeResult:
    """Execute a single evaluation trial under matched conditions with full telemetry."""
    # 1. Reset base environment
    env.reset(seed=seed)

    # 2. Configure benchmark fixture
    start_x, start_y, start_th, goal_x, goal_y = suite.setup_case(condition, seed)
    env.robot_x = start_x
    env.robot_y = start_y
    env.robot_theta = start_th
    env.goal_x = goal_x
    env.goal_y = goal_y
    env.prev_goal_dist = math.hypot(goal_x - start_x, goal_y - start_y)

    # 3. Reset preprocessor and action buffer
    env.preprocessor.reset(init_x=start_x, init_y=start_y, init_theta=start_th)
    env.prev_action = np.zeros(2, dtype=np.float32)
    env.robot_v = 0.0
    env.robot_omega = 0.0

    # 4. Configure condition-specific randomizers
    sens_rand = None
    dyn_rand = None
    env.randomizer = None

    if condition == "case7a_sensor_noise":
        sens_rand = SensorRandomizer(seed=seed)
        # OOD noise exceeding training parameters: intercept 0.025, slope 0.015
        sens_rand.params = SensorParams(range_noise_a=0.025, range_noise_b=0.015)
        sens_rand.reset()
        env.randomizer = sens_rand
    elif condition == "case8_actuation_lag":
        dyn_rand = DynamicsRandomizer(seed=seed)
        # OOD actuation: delay 0.20s (1 step), lag tau 0.30s (alpha=0.40), gains 0.80/1.20
        dyn_rand.params = DynamicsParams(
            control_delay_s=0.20,
            actuator_lag_tau=0.30,
            linear_acc_limit=0.20,
            angular_acc_limit=1.00,
            v_command_gain=0.80,
            omega_command_gain=1.20,
        )
        dyn_rand.reset()
        env.randomizer = dyn_rand

    # 5. Gazebo physical state synchronization
    if env.use_gazebo and env.ros_node is not None:
        env._teleport_gazebo_entities()

    # Initial observation
    raw_ranges = env._get_lidar_ranges()
    if condition == "case7a_sensor_noise" and sens_rand is not None:
        raw_ranges = sens_rand.perturb_scan(raw_ranges)
    elif condition == "case7b_blind_sector":
        # Mask forward 30-degree sector (angles in [-15 deg, +15 deg]) to RANGE_MAX
        angles_deg = np.linspace(0.0, 360.0, 360, endpoint=False)
        blind_mask = (angles_deg <= 15.0) | (angles_deg >= 345.0)
        raw_ranges[blind_mask] = RANGE_MAX

    current_obs = env.preprocessor.step(
        ranges=raw_ranges,
        v_meas=env.robot_v,
        omega_meas=env.robot_omega,
        goal_x=env.goal_x,
        goal_y=env.goal_y,
        last_action=env.prev_action,
        dt=env.dt,
    )

    trajectory = [(env.robot_x, env.robot_y)]
    omegas = []
    min_clearance = float("inf")
    min_lidar_range_episode = float(np.min(raw_ranges)) if len(raw_ranges) > 0 else RANGE_MAX
    lidar_proximity_event = bool(min_lidar_range_episode <= RANGE_MIN)
    had_geometric_col = False
    had_gazebo_col = False
    total_reward = 0.0
    outcome = "TIMEOUT"
    failure_reason = "TIMEOUT"

    for step_idx in range(max_steps):
        # Case 5 dynamic obstacle update:
        # Stationary for t < 6.0s at (-0.80, 0.0).
        # At t >= 6.0s, moves East at 0.20m/s until x >= +0.70m, then halts.
        if condition == "case5_dynamic":
            sim_time = step_idx * env.dt
            obs5 = env.obstacle_pool[0]
            if sim_time >= 6.0:
                t_move = sim_time - 6.0
                obs5.x = min(0.70, -0.80 + 0.20 * t_move)
            else:
                obs5.x = -0.80
            obs5.y = 0.00
            # If in Gazebo, update obstacle entity
            if env.use_gazebo and hasattr(env, 'cli_set_state'):
                try:
                    from gazebo_msgs.srv import SetEntityState
                    from geometry_msgs.msg import Point, Quaternion, Vector3
                    req_o = SetEntityState.Request()
                    req_o.state.name = obs5.name
                    req_o.state.pose.position = Point(x=obs5.x, y=obs5.y, z=0.25)
                    req_o.state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    req_o.state.reference_frame = 'world'
                    env.cli_set_state.call_async(req_o)
                except Exception:
                    pass

        # Inference
        action, _ = policy.predict(current_obs, deterministic=True)
        omegas.append(float(0.60 * action[1]))

        # Step environment (applies dynamics randomizer perturb_action internally if attached)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        current_obs = next_obs
        trajectory.append((env.robot_x, env.robot_y))

        # Check clearances
        d_goal = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
        for s in env.static_obstacles:
            if s.active:
                d = s.distance_to_point(env.robot_x, env.robot_y) - env.robot_radius
                min_clearance = min(min_clearance, d)
        for o in env.obstacle_pool:
            if o.active:
                d = o.distance_to_point(env.robot_x, env.robot_y) - env.robot_radius
                min_clearance = min(min_clearance, d)

        # Track minimum LiDAR range across the episode
        ranges_step = env._get_lidar_ranges()
        step_min_range = float(np.min(ranges_step))
        min_lidar_range_episode = min(min_lidar_range_episode, step_min_range)
        if step_min_range <= RANGE_MIN:
            lidar_proximity_event = True

        # Evaluate physical vs geometric collision
        step_geom_col = env.check_geometric_collision() or (min_clearance <= 0.0)
        step_gazebo_col = bool(getattr(env, '_gazebo_contact_occurred', False))

        if step_geom_col:
            had_geometric_col = True
        if step_gazebo_col:
            had_gazebo_col = True

        # Primary physical collision: Gazebo contact if active; otherwise geometric penetration
        if env.use_gazebo and getattr(env, '_bumper_subscribed_and_active', False):
            is_physical_col = step_gazebo_col
        else:
            is_physical_col = step_geom_col or step_gazebo_col

        # Goal arrival check using evaluation threshold (0.25m)
        is_goal_reached = (d_goal <= goal_eval_tolerance)

        if is_physical_col:
            outcome = "COLLISION"
            failure_reason = "PHYSICAL_CONTACT" if step_gazebo_col else "GEOMETRIC_PENETRATION"
            break
        elif is_goal_reached:
            outcome = "SUCCESS"
            failure_reason = "NONE"
            break
        elif truncated or (step_idx >= max_steps - 1):
            outcome = "TIMEOUT"
            failure_reason = "MAX_STEPS_EXCEEDED"
            break

    # Clean up trial randomizer
    env.randomizer = None

    # Calculate trajectory metrics
    path_len = sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(trajectory[:-1], trajectory[1:])
    )

    steps_taken = len(trajectory) - 1
    duration_sec = steps_taken * env.dt

    # Normalized angular velocity total variation: TV_norm(w) = (1 / ((N-1)*dt)) * sum(|w_{t+1} - w_t|)
    if len(omegas) > 1:
        diffs = [abs(omegas[i+1] - omegas[i]) for i in range(len(omegas) - 1)]
        tv_norm = sum(diffs) / ((len(omegas) - 1) * env.dt)
    else:
        tv_norm = 0.0

    is_success = 1 if outcome == "SUCCESS" else 0
    is_col = 1 if outcome == "COLLISION" else 0
    is_to = 1 if outcome == "TIMEOUT" else 0
    penalized_time = duration_sec if is_success else 100.0

    return EpisodeResult(
        policy=policy_name,
        condition=condition,
        seed=seed,
        outcome=outcome,
        success=is_success,
        collision=is_col,
        timeout=is_to,
        collision_gazebo_contact=1 if had_gazebo_col else 0,
        collision_geometric=1 if had_geometric_col else 0,
        min_lidar_range_m=round(min_lidar_range_episode, 3),
        lidar_proximity_warning=1 if lidar_proximity_event else 0,
        collision_disagreement=1 if (had_gazebo_col != had_geometric_col) else 0,
        completion_time_sec=round(duration_sec, 3),
        penalized_time_sec=round(penalized_time, 3),
        steps=steps_taken,
        path_length_m=round(path_len, 3),
        smoothness_tv=round(tv_norm, 3),
        min_clearance_m=round(min_clearance, 3) if not math.isinf(min_clearance) else 0.50,
        final_goal_dist_m=round(math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y), 3),
        reward=round(total_reward, 3),
        failure_reason=failure_reason,
    )


def run_benchmark(
    mode: str = "pilot",
    selected_policies: Optional[List[str]] = None,
    selected_conditions: Optional[List[str]] = None,
    base_seed: int = 2000000,
    out_dir: str = "runs/benchmark_results",
    use_gazebo: bool = False,
    use_final: bool = False,
    watchdog_timeout: float = 5.0,
) -> Dict[str, Any]:
    """Execute evaluation across selected policies and conditions."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    policies_to_run = selected_policies or POLICY_NAMES
    conditions_to_run = selected_conditions or BENCHMARK_CONDITIONS

    n_seeds = 1 if mode == "pilot" else 20
    seeds = [base_seed + i for i in range(n_seeds)]

    print("=" * 80)
    print(f" SIM-TO-REAL LIDAR NAVIGATION PHASE 2 BENCHMARK")
    print(f" Mode: {mode.upper()} ({len(conditions_to_run)} conditions x {len(policies_to_run)} policies x {len(seeds)} seeds = {len(conditions_to_run)*len(policies_to_run)*len(seeds)} episodes)")
    print(f" Checkpoint Tier: {'FINAL CHECKPOINTS' if use_final else 'BEST CHECKPOINTS (PRIMARY)'}")
    print(f" Gazebo Simulation: {'ACTIVE' if use_gazebo else 'FAST ANALYTICAL MOCK'}")
    print(f" Output Directory: {out_path}")
    print("=" * 80)

    # Load policies
    loaded_policies: Dict[str, NumPyPolicy] = {}
    for p_name in policies_to_run:
        if use_final:
            npz_candidate = REPO_ROOT / f"deploy/{p_name.lower()}_policy.npz"
            zip_candidate = REPO_ROOT / f"runs/main_{p_name.lower()}_s0/final_model.zip"
            tier_name = "FINAL"
        else:
            npz_candidate = REPO_ROOT / f"deploy/{p_name.lower()}_best_policy.npz"
            zip_candidate = REPO_ROOT / f"runs/main_{p_name.lower()}_s0/best_model/best_model.zip"
            tier_name = "BEST"

        target_path = npz_candidate if npz_candidate.exists() else zip_candidate
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint for {p_name} ({tier_name}) not found at {npz_candidate} or {zip_candidate}")
        loaded_policies[p_name] = NumPyPolicy.load(target_path)
        print(f"Loaded policy {p_name} [{tier_name}] from: {target_path}")

    # Initialize environment
    env = TurtleBot3NavigationEnv(
        use_gazebo=use_gazebo,
        arena_size=5.0,
        max_episode_steps=500,
        goal_tolerance=0.20,
    )

    if use_gazebo and env.ros_node is not None:
        import rclpy
        print("Waiting for Gazebo simulator to become ready (/scan and set_entity_state)...")
        t_wait_gz = time.time()
        gz_ready = False
        while time.time() - t_wait_gz < 30.0:
            rclpy.spin_once(env.ros_node, timeout_sec=0.2)
            if env._latest_scan is not None and env.cli_set_state.service_is_ready():
                gz_ready = True
                break
            time.sleep(0.3)
        if gz_ready:
            print("Gazebo simulator is ready and connected successfully!")
        else:
            print("WARNING: Gazebo simulator did not report fresh scan within 30s timeout.")

    suite = BenchmarkSuite(env)

    results: List[EpisodeResult] = []

    total_episodes = len(conditions_to_run) * len(policies_to_run) * len(seeds)
    ep_count = 0

    t_benchmark_start = time.time()

    for cond in conditions_to_run:
        print(f"\n--- Condition: {cond} ---")
        for s in seeds:
            for p_name in policies_to_run:
                ep_count += 1
                pol = loaded_policies[p_name]
                res = evaluate_trial(
                    env=env,
                    suite=suite,
                    policy=pol,
                    policy_name=p_name,
                    condition=cond,
                    seed=s,
                    max_steps=500,
                    goal_eval_tolerance=0.25,
                )
                results.append(res)
                print(f"[{ep_count:3d}/{total_episodes:3d}] {p_name} | {cond:<20s} | Seed {s} -> {res.outcome:<9s} | Steps: {res.steps:3d} | Len: {res.path_length_m:.2f}m | TV: {res.smoothness_tv:.2f} | Time: {res.completion_time_sec:.1f}s")

    env.close()
    elapsed_total = time.time() - t_benchmark_start

    # Export machine-readable CSV
    csv_file = out_path / "raw_episodes.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Compute aggregate summary statistics
    summary: Dict[str, Any] = {
        "benchmark_metadata": {
            "mode": mode,
            "total_episodes": len(results),
            "n_conditions": len(conditions_to_run),
            "n_policies": len(policies_to_run),
            "n_seeds_per_condition": len(seeds),
            "use_gazebo": use_gazebo,
            "elapsed_seconds": round(elapsed_total, 2),
        },
        "per_policy": {},
        "per_condition": {},
        "per_policy_condition": {},
    }

    for p_name in policies_to_run:
        p_res = [r for r in results if r.policy == p_name]
        n_p = len(p_res)
        succ = [r for r in p_res if r.success == 1]
        summary["per_policy"][p_name] = {
            "total": n_p,
            "successes": len(succ),
            "success_rate": round(len(succ) / n_p, 4),
            "collisions": sum(r.collision for r in p_res),
            "collision_rate": round(sum(r.collision for r in p_res) / n_p, 4),
            "timeouts": sum(r.timeout for r in p_res),
            "timeout_rate": round(sum(r.timeout for r in p_res) / n_p, 4),
            "mean_penalized_time_sec": round(float(np.mean([r.penalized_time_sec for r in p_res])), 2),
            "conditional_successful_episodes": {
                "n": len(succ),
                "mean_path_length_m": round(float(np.mean([r.path_length_m for r in succ])), 3) if succ else None,
                "mean_smoothness_tv": round(float(np.mean([r.smoothness_tv for r in succ])), 3) if succ else None,
                "mean_completion_time_sec": round(float(np.mean([r.completion_time_sec for r in succ])), 2) if succ else None,
            }
        }

    for cond in conditions_to_run:
        c_res = [r for r in results if r.condition == cond]
        n_c = len(c_res)
        succ_c = [r for r in c_res if r.success == 1]
        summary["per_condition"][cond] = {
            "total": n_c,
            "success_rate": round(len(succ_c) / n_c, 4),
            "collision_rate": round(sum(r.collision for r in c_res) / n_c, 4),
            "timeout_rate": round(sum(r.timeout for r in c_res) / n_c, 4),
        }

    json_file = out_path / "benchmark_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 88)
    print(f" BENCHMARK EVALUATION SUMMARY ({mode.upper()} MODE: {len(results)} EPISODES)")
    print("=" * 88)
    print(f"{'Policy':<8} | {'Success Rate':<16} | {'Collisions':<12} | {'Timeouts':<10} | {'Pen. Time':<10} | {'Succ. Len':<10} | {'Succ. TV':<10}")
    print("-" * 88)
    for p_name in policies_to_run:
        p_stat = summary["per_policy"][p_name]
        succ_info = p_stat["conditional_successful_episodes"]
        succ_str = f"{p_stat['successes']}/{p_stat['total']} ({p_stat['success_rate']*100:.1f}%)"
        col_str = f"{p_stat['collisions']} ({p_stat['collision_rate']*100:.1f}%)"
        to_str = f"{p_stat['timeouts']} ({p_stat['timeout_rate']*100:.1f}%)"
        pen_time = f"{p_stat['mean_penalized_time_sec']:.1f}s"
        succ_len = f"{succ_info['mean_path_length_m']:.2f}m" if succ_info['mean_path_length_m'] is not None else "N/A"
        succ_tv = f"{succ_info['mean_smoothness_tv']:.2f}" if succ_info['mean_smoothness_tv'] is not None else "N/A"
        print(f"{p_name:<8} | {succ_str:<16} | {col_str:<12} | {to_str:<10} | {pen_time:<10} | {succ_len:<10} | {succ_tv:<10}")
    print("=" * 88)
    print(f"Raw CSV Export: {csv_file}")
    print(f"Summary JSON:   {json_file}")
    print("=" * 88)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified Phase 2 Benchmark Evaluation")
    parser.add_argument("--mode", type=str, default="pilot", choices=["pilot", "benchmark"], help="Evaluation mode (pilot=36 eps, benchmark=720 eps)")
    parser.add_argument("--policy", type=str, default="all", help="Comma-separated policy selection (e.g. 'all' or 'P1,P4')")
    parser.add_argument("--condition", type=str, default="all", help="Comma-separated condition selection (e.g. 'all' or 'case1_corridor,case5_dynamic')")
    parser.add_argument("--base-seed", type=int, default=2000000, help="Base seed for evaluation")
    parser.add_argument("--out-dir", type=str, default="runs/benchmark_results", help="Output directory")
    parser.add_argument("--use-gazebo", type=str, default="false", help="Use live Gazebo simulation (true/false)")
    parser.add_argument("--use-final", action="store_true", default=False, help="Evaluate final model checkpoints instead of best checkpoints")
    args = parser.parse_args()

    use_gz = args.use_gazebo.lower() in ["true", "1", "yes"]

    selected_pols = None
    if args.policy.lower() != "all":
        selected_pols = [p.strip().upper() for p in args.policy.split(",")]

    selected_conds = None
    if args.condition.lower() != "all":
        selected_conds = [c.strip().lower() for c in args.condition.split(",")]

    run_benchmark(
        mode=args.mode,
        selected_policies=selected_pols,
        selected_conditions=selected_conds,
        base_seed=args.base_seed,
        out_dir=args.out_dir,
        use_gazebo=use_gz,
        use_final=args.use_final,
    )


if __name__ == "__main__":
    main()
