"""Scenario Generator for reproducible evaluation and training layouts.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 8, 9, 10)
Requirements:
- Same seed gives same layout (deterministic).
- Train, validation, test, and stress seed sets are disjoint.
- Independent random-number streams via SeedSequence.spawn.
- Scenario files are hashable for reproducibility verification.
"""

from __future__ import annotations
import json
import hashlib
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class ObstacleSpec:
    """Obstacle placement specification."""
    kind: str           # 'box' or 'cylinder'
    name: str           # e.g., 'obstacle_box_1'
    size: Tuple[float, ...]  # (w, h) for box, (radius,) for cylinder
    x: float = 0.0
    y: float = 0.0
    active: bool = True


@dataclass
class ScenarioSpec:
    """Complete specification for one episode layout."""
    scenario_id: int
    seed: int
    robot_x: float
    robot_y: float
    robot_theta: float
    goal_x: float
    goal_y: float
    start_goal_distance: float
    obstacles: List[ObstacleSpec] = field(default_factory=list)
    num_active_obstacles: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ScenarioSpec':
        obs_list = [ObstacleSpec(**o) for o in d.pop('obstacles', [])]
        return cls(**d, obstacles=obs_list)

    @classmethod
    def from_json(cls, s: str) -> 'ScenarioSpec':
        return cls.from_dict(json.loads(s))


# The default obstacle pool matching arena_4x4.world and tb3_navigation_env.py
DEFAULT_OBSTACLE_POOL = [
    ('box', 'obstacle_box_1', (0.30, 0.30)),
    ('box', 'obstacle_box_2', (0.25, 0.25)),
    ('box', 'obstacle_box_3', (0.35, 0.35)),
    ('box', 'obstacle_box_4', (0.30, 0.30)),
    ('cylinder', 'obstacle_cyl_1', (0.15,)),
    ('cylinder', 'obstacle_cyl_2', (0.12,)),
    ('cylinder', 'obstacle_cyl_3', (0.18,)),
    ('cylinder', 'obstacle_cyl_4', (0.15,)),
]


# Disjoint seed ranges for each split (never overlap)
SEED_RANGES = {
    'train':      (0,        999_999),      # Procedural stream
    'validation': (1_000_000, 1_000_049),   # 50 fixed scenarios
    'test':       (2_000_000, 2_000_099),   # 100 fixed scenarios
    'stress':     (3_000_000, 3_000_009),   # ~10 stress scenarios
}


def _obs_bounding_radius(kind: str, size: Tuple[float, ...]) -> float:
    """Compute bounding radius for an obstacle."""
    if kind == 'cylinder':
        return size[0]
    else:
        w, h = size
        return 0.5 * math.hypot(w, h)


def generate_scenario(
    seed: int,
    scenario_id: int = 0,
    arena_size: float = 4.0,
    robot_radius: float = 0.105,
    min_start_goal_dist: float = 1.5,
    max_start_goal_dist: float = 3.5,
    min_obstacles: int = 4,
    max_obstacles: int = 8,
    wall_clearance: float = 0.40,
    obstacle_pool: Optional[List[Tuple[str, str, Tuple[float, ...]]]] = None,
) -> ScenarioSpec:
    """Generate a single deterministic scenario from a seed.

    Args:
        seed: Integer seed for full reproducibility.
        scenario_id: Numeric ID for this scenario.
        arena_size: Square arena side length (m).
        robot_radius: Robot collision radius (m).
        min_start_goal_dist: Minimum start-to-goal distance (m).
        max_start_goal_dist: Maximum start-to-goal distance (m).
        min_obstacles: Minimum number of active obstacles.
        max_obstacles: Maximum number of active obstacles.
        wall_clearance: Minimum clearance from walls for robot/goal placement (m).
        obstacle_pool: List of (kind, name, size) tuples.

    Returns:
        ScenarioSpec with all placement details.
    """
    if obstacle_pool is None:
        obstacle_pool = DEFAULT_OBSTACLE_POOL

    rng = np.random.default_rng(seed)
    arena_half = arena_size / 2.0

    # 1. Sample robot start pose
    spawn_bound = arena_half - wall_clearance - robot_radius
    robot_x = float(rng.uniform(-spawn_bound, spawn_bound))
    robot_y = float(rng.uniform(-spawn_bound, spawn_bound))
    robot_theta = float(rng.uniform(-np.pi, np.pi))

    # 2. Sample goal with distance constraint
    goal_x, goal_y = 0.0, 0.0
    for _ in range(500):
        dist = float(rng.uniform(min_start_goal_dist, max_start_goal_dist))
        angle = float(rng.uniform(-np.pi, np.pi))
        gx = robot_x + dist * math.cos(angle)
        gy = robot_y + dist * math.sin(angle)
        if abs(gx) <= (arena_half - wall_clearance) and abs(gy) <= (arena_half - wall_clearance):
            goal_x, goal_y = gx, gy
            break
    else:
        # Fallback: opposite corner direction
        goal_x = float(np.clip(-robot_x, -(arena_half - wall_clearance), arena_half - wall_clearance))
        goal_y = float(np.clip(-robot_y, -(arena_half - wall_clearance), arena_half - wall_clearance))

    start_goal_distance = math.hypot(goal_x - robot_x, goal_y - robot_y)

    # 3. Sample obstacles
    num_active = int(rng.integers(min_obstacles, max_obstacles + 1))
    perm = rng.permutation(len(obstacle_pool))

    placed_obstacles: List[ObstacleSpec] = []
    for i in range(num_active):
        pool_idx = int(perm[i % len(obstacle_pool)])
        kind, name, size = obstacle_pool[pool_idx]
        obs_r = _obs_bounding_radius(kind, size)

        placed = False
        for _ in range(200):
            obs_bound = arena_half - 0.15 - obs_r
            ox = float(rng.uniform(-obs_bound, obs_bound))
            oy = float(rng.uniform(-obs_bound, obs_bound))

            # Clearance from robot start
            if math.hypot(ox - robot_x, oy - robot_y) < (0.40 + robot_radius + obs_r):
                continue
            # Clearance from goal
            if math.hypot(ox - goal_x, oy - goal_y) < (0.40 + obs_r):
                continue
            # Clearance from other placed obstacles
            overlap = False
            for p in placed_obstacles:
                p_r = _obs_bounding_radius(p.kind, p.size)
                if math.hypot(ox - p.x, oy - p.y) < (obs_r + p_r + 0.10):
                    overlap = True
                    break
            if overlap:
                continue

            placed_obstacles.append(ObstacleSpec(
                kind=kind, name=name, size=size,
                x=ox, y=oy, active=True
            ))
            placed = True
            break

        if not placed:
            # Park outside arena
            placed_obstacles.append(ObstacleSpec(
                kind=kind, name=name, size=size,
                x=20.0, y=20.0, active=False
            ))

    return ScenarioSpec(
        scenario_id=scenario_id,
        seed=seed,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        goal_x=goal_x,
        goal_y=goal_y,
        start_goal_distance=start_goal_distance,
        obstacles=placed_obstacles,
        num_active_obstacles=sum(1 for o in placed_obstacles if o.active),
    )


def generate_corridor_scenario(
    seed: int,
    scenario_id: int = 0,
    arena_size: float = 4.0,
    corridor_width: float = 0.75,
) -> ScenarioSpec:
    """Generate a structural corridor scenario with parallel obstacle rows."""
    rng = np.random.default_rng(seed)

    shift_x = float(rng.uniform(-0.2, 0.2))

    # Obstacles forming left and right walls of corridor
    obstacles = [
        ObstacleSpec('box', 'obstacle_box_1', (0.30, 0.30), x=shift_x - corridor_width/2 - 0.15, y=-0.6, active=True),
        ObstacleSpec('box', 'obstacle_box_2', (0.25, 0.25), x=shift_x - corridor_width/2 - 0.15, y=0.0, active=True),
        ObstacleSpec('box', 'obstacle_box_3', (0.35, 0.35), x=shift_x - corridor_width/2 - 0.20, y=0.6, active=True),
        ObstacleSpec('cylinder', 'obstacle_cyl_1', (0.15,), x=shift_x + corridor_width/2 + 0.15, y=-0.6, active=True),
        ObstacleSpec('cylinder', 'obstacle_cyl_2', (0.12,), x=shift_x + corridor_width/2 + 0.15, y=0.0, active=True),
        ObstacleSpec('cylinder', 'obstacle_cyl_3', (0.18,), x=shift_x + corridor_width/2 + 0.20, y=0.6, active=True),
    ]

    robot_x = shift_x
    robot_y = -1.4
    robot_theta = math.pi / 2.0  # Facing into the corridor
    goal_x = shift_x
    goal_y = 1.4
    dist = math.hypot(goal_x - robot_x, goal_y - robot_y)

    return ScenarioSpec(
        scenario_id=scenario_id,
        seed=seed,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        goal_x=goal_x,
        goal_y=goal_y,
        start_goal_distance=dist,
        obstacles=obstacles,
        num_active_obstacles=len(obstacles),
    )


def generate_narrow_gap_scenario(
    seed: int,
    scenario_id: int = 0,
    arena_size: float = 4.0,
    gap_width: float = 0.50,
) -> ScenarioSpec:
    """Generate a dividing barrier with exactly a 0.5m narrow gap."""
    rng = np.random.default_rng(seed)

    shift_y = float(rng.uniform(-0.2, 0.2))
    left_obs_r = 0.15
    right_obs_r = 0.15

    obstacles = [
        # Left boundary of gap
        ObstacleSpec('cylinder', 'obstacle_cyl_1', (left_obs_r,), x=-gap_width/2.0 - left_obs_r, y=shift_y, active=True),
        # Right boundary of gap
        ObstacleSpec('cylinder', 'obstacle_cyl_2', (right_obs_r,), x=gap_width/2.0 + right_obs_r, y=shift_y, active=True),
        # Outer barrier extensions
        ObstacleSpec('box', 'obstacle_box_1', (0.35, 0.35), x=-gap_width/2.0 - left_obs_r - 0.60, y=shift_y, active=True),
        ObstacleSpec('box', 'obstacle_box_2', (0.35, 0.35), x=-gap_width/2.0 - left_obs_r - 1.20, y=shift_y, active=True),
        ObstacleSpec('box', 'obstacle_box_3', (0.35, 0.35), x=gap_width/2.0 + right_obs_r + 0.60, y=shift_y, active=True),
        ObstacleSpec('box', 'obstacle_box_4', (0.35, 0.35), x=gap_width/2.0 + right_obs_r + 1.20, y=shift_y, active=True),
    ]

    robot_x = float(rng.uniform(-0.3, 0.3))
    robot_y = shift_y - 1.2
    robot_theta = math.pi / 2.0
    goal_x = float(rng.uniform(-0.3, 0.3))
    goal_y = shift_y + 1.2
    dist = math.hypot(goal_x - robot_x, goal_y - robot_y)

    return ScenarioSpec(
        scenario_id=scenario_id,
        seed=seed,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        goal_x=goal_x,
        goal_y=goal_y,
        start_goal_distance=dist,
        obstacles=obstacles,
        num_active_obstacles=len(obstacles),
    )


def generate_dead_end_scenario(
    seed: int,
    scenario_id: int = 0,
    arena_size: float = 4.0,
) -> ScenarioSpec:
    """Generate a U-shaped dead-end pocket with the goal behind the back wall."""
    rng = np.random.default_rng(seed)

    center_y = float(rng.uniform(0.0, 0.3))

    obstacles = [
        # Back wall (blocks direct path to goal)
        ObstacleSpec('box', 'obstacle_box_1', (0.35, 0.30), x=-0.35, y=center_y + 0.5, active=True),
        ObstacleSpec('box', 'obstacle_box_2', (0.35, 0.30), x=0.0, y=center_y + 0.5, active=True),
        ObstacleSpec('box', 'obstacle_box_3', (0.35, 0.30), x=0.35, y=center_y + 0.5, active=True),
        # Left wall of pocket
        ObstacleSpec('cylinder', 'obstacle_cyl_1', (0.15,), x=-0.65, y=center_y + 0.1, active=True),
        ObstacleSpec('cylinder', 'obstacle_cyl_2', (0.15,), x=-0.65, y=center_y - 0.3, active=True),
        # Right wall of pocket
        ObstacleSpec('cylinder', 'obstacle_cyl_3', (0.15,), x=0.65, y=center_y + 0.1, active=True),
        ObstacleSpec('cylinder', 'obstacle_cyl_4', (0.15,), x=0.65, y=center_y - 0.3, active=True),
    ]

    robot_x = 0.0
    robot_y = center_y
    robot_theta = math.pi / 2.0  # Facing into the trap

    goal_x = 0.0
    goal_y = center_y + 1.2
    dist = math.hypot(goal_x - robot_x, goal_y - robot_y)

    return ScenarioSpec(
        scenario_id=scenario_id,
        seed=seed,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        goal_x=goal_x,
        goal_y=goal_y,
        start_goal_distance=dist,
        obstacles=obstacles,
        num_active_obstacles=len(obstacles),
    )


def generate_scenario_set(
    split: str,
    count: Optional[int] = None,
    base_seed_offset: int = 0,
) -> List[ScenarioSpec]:
    """Generate a set of deterministic scenarios for a given split.

    Args:
        split: One of 'train', 'validation', 'test', 'stress'.
        count: Number of scenarios to generate (None = full range).
        base_seed_offset: Additional offset for training seed stream.

    Returns:
        List of ScenarioSpec.
    """
    if split not in SEED_RANGES:
        raise ValueError(f"Unknown split '{split}'. Must be one of {list(SEED_RANGES.keys())}")

    start_seed, end_seed = SEED_RANGES[split]
    max_count = end_seed - start_seed + 1

    if count is None:
        count = max_count
    count = min(count, max_count)

    scenarios = []
    if split == 'stress':
        # Explicit structural stress scenarios per specification:
        # corridor, 0.5m narrow gap, dead-end pocket
        types = ['corridor', 'narrow_gap_0.5m', 'dead_end_pocket']
        for i in range(count):
            seed = start_seed + base_seed_offset + i
            stype = types[i % len(types)]
            if stype == 'corridor':
                spec = generate_corridor_scenario(seed=seed, scenario_id=i)
            elif stype == 'narrow_gap_0.5m':
                spec = generate_narrow_gap_scenario(seed=seed, scenario_id=i, gap_width=0.50)
            else:
                spec = generate_dead_end_scenario(seed=seed, scenario_id=i)
            scenarios.append(spec)
    else:
        for i in range(count):
            seed = start_seed + base_seed_offset + i
            spec = generate_scenario(seed=seed, scenario_id=i)
            scenarios.append(spec)

    return scenarios


def save_scenario_set(scenarios: List[ScenarioSpec], output_dir: str) -> str:
    """Save scenario set to a JSON file and return its SHA-256 hash.

    Args:
        scenarios: List of scenarios to save.
        output_dir: Directory to write the scenario file.

    Returns:
        SHA-256 hex digest of the saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = [s.to_dict() for s in scenarios]
    json_str = json.dumps(data, indent=2, sort_keys=True)

    file_path = output_path / "scenarios.json"
    file_path.write_text(json_str, encoding='utf-8')

    sha256 = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    meta = {
        "count": len(scenarios),
        "sha256": sha256,
        "seed_range": [scenarios[0].seed, scenarios[-1].seed] if scenarios else [],
    }
    (output_path / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding='utf-8'
    )

    return sha256


def load_scenario_set(scenario_dir: str) -> List[ScenarioSpec]:
    """Load scenarios from a JSON file.

    Args:
        scenario_dir: Directory containing scenarios.json.

    Returns:
        List of ScenarioSpec.
    """
    file_path = Path(scenario_dir) / "scenarios.json"
    data = json.loads(file_path.read_text(encoding='utf-8'))
    return [ScenarioSpec.from_dict(d) for d in data]


def verify_disjoint_seeds() -> bool:
    """Verify that all seed ranges are disjoint. Returns True if valid."""
    ranges = list(SEED_RANGES.values())
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a_start, a_end = ranges[i]
            b_start, b_end = ranges[j]
            if a_start <= b_end and b_start <= a_end:
                return False
    return True


if __name__ == '__main__':
    import sys

    # Generate and save all fixed scenario sets
    assert verify_disjoint_seeds(), "Seed ranges overlap!"

    base_dir = Path(__file__).parent

    for split in ['validation', 'test', 'stress']:
        scenarios = generate_scenario_set(split)
        sha = save_scenario_set(scenarios, str(base_dir / split))
        print(f"{split}: {len(scenarios)} scenarios, SHA-256={sha[:16]}...")

    # Generate a sample training batch
    train_sample = generate_scenario_set('train', count=10)
    sha = save_scenario_set(train_sample, str(base_dir / 'train'))
    print(f"train (sample): {len(train_sample)} scenarios, SHA-256={sha[:16]}...")

    print("\nAll scenario sets generated successfully.")
