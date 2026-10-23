"""TurtleBot3 Burger LiDAR Navigation Gymnasium Environment.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 4, 5)
Follows standard Gymnasium API: step(), reset(), observation_space, action_space.

Design Requirements:
- Action space: Continuous Box(2,), u in [-1, 1]^2
  Action mapping: v = 0.075 * (u_v + 1.0) in [0.0, 0.15] m/s
                  omega = 0.60 * u_omega in [-0.60, 0.60] rad/s
- Observation space: Continuous Box(43,)
  36 min-pooled 10° LiDAR bins, goal distance, sin/cos bearing, v, omega, prev action
- Estimated-pose model: Integrated from reset odometry frame
- Reward: r_t = 10 * (d_t - d_{t+1}) - 0.05 + 100*[goal] - 100*[collision]
- Terminations:
  terminated = goal_reached (d <= 0.20m) or collision (footprint contact)
  truncated = timeout (step >= 350)
- Ground-truth state strictly confined to reward/termination/evaluation.
"""

from typing import Tuple, Dict, Any, Optional, List
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.env.observation import (
    ObservationPreprocessor,
    apply_motion_distortion,
    OBS_DIM,
    RANGE_MAX,
    ARENA_DIAGONAL,
    V_MAX,
    OMEGA_MAX,
    wrap_to_pi,
)
from rl.train.watchdog import load_watchdog_timeout, MissingScanTimeoutError, WatchdogRecoveryError



class Obstacle:
    """Obstacle geometry representation for collision detection and raycasting."""

    def __init__(self, kind: str, name: str, size: Tuple[float, ...]):
        self.kind = kind  # 'box' or 'cylinder'
        self.name = name
        self.size = size  # box: (w, h), cylinder: (radius,)
        self.x: float = 20.0
        self.y: float = 20.0
        self.active: bool = False

    @property
    def radius(self) -> float:
        """Bounding radius for clearance and overlap rejection."""
        if self.kind == 'cylinder':
            return self.size[0]
        else:
            w, h = self.size
            return 0.5 * math.hypot(w, h)

    def check_collision(self, rx: float, ry: float, robot_radius: float) -> bool:
        """Check collision between robot circle and obstacle."""
        if not self.active:
            return False

        if self.kind == 'cylinder':
            dist = math.hypot(rx - self.x, ry - self.y)
            return dist <= (self.size[0] + robot_radius)
        else:
            w, h = self.size
            dx = max(0.0, abs(rx - self.x) - 0.5 * w)
            dy = max(0.0, abs(ry - self.y) - 0.5 * h)
            return (dx * dx + dy * dy) <= (robot_radius * robot_radius)

    def ray_intersect(self, ox: float, oy: float, angle: float, max_dist: float) -> float:
        """Ray-obstacle intersection returning distance to intersection or max_dist."""
        if not self.active:
            return max_dist

        ca = math.cos(angle)
        sa = math.sin(angle)

        if self.kind == 'cylinder':
            # Ray-circle intersection
            cx, cy = self.x, self.y
            r = self.size[0]
            ocx = ox - cx
            ocy = oy - cy

            b = 2.0 * (ocx * ca + ocy * sa)
            c = ocx * ocx + ocy * ocy - r * r
            disc = b * b - 4.0 * c
            if disc < 0.0:
                return max_dist
            sq_disc = math.sqrt(disc)
            t1 = (-b - sq_disc) / 2.0
            if t1 > 0.0:
                return min(t1, max_dist)
            t2 = (-b + sq_disc) / 2.0
            if t2 > 0.0:
                return min(t2, max_dist)
            return max_dist
        else:
            # Ray-AABB intersection for box
            w, h = self.size
            x_min = self.x - 0.5 * w
            x_max = self.x + 0.5 * w
            y_min = self.y - 0.5 * h
            y_max = self.y + 0.5 * h

            t_near = -1e9
            t_far = 1e9

            # X slab
            if abs(ca) < 1e-8:
                if ox < x_min or ox > x_max:
                    return max_dist
            else:
                t1 = (x_min - ox) / ca
                t2 = (x_max - ox) / ca
                t_near = max(t_near, min(t1, t2))
                t_far = min(t_far, max(t1, t2))

            # Y slab
            if abs(sa) < 1e-8:
                if oy < y_min or oy > y_max:
                    return max_dist
            else:
                t1 = (y_min - oy) / sa
                t2 = (y_max - oy) / sa
                t_near = max(t_near, min(t1, t2))
                t_far = min(t_far, max(t1, t2))

            if t_near <= t_far and t_far > 0.0:
                hit_dist = t_near if t_near > 0.0 else t_far
                return min(hit_dist, max_dist)
            return max_dist


class TurtleBot3NavigationEnv(gym.Env):
    """Gymnasium Environment for TurtleBot3 LiDAR Navigation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        use_gazebo: bool = False,
        arena_size: float = 4.0,
        goal_tolerance: float = 0.20,
        max_episode_steps: int = 350,
        control_rate: float = 5.0,  # Hz (0.2s period)
        robot_radius: float = 0.105,  # meters (TurtleBot3 Burger footprint)
        enable_motion_distortion: bool = True,
        randomizer: Optional[Any] = None,
        seed: Optional[int] = None,
        watchdog: Optional[Any] = None,
        watchdog_timeout_sec: Optional[float] = None,
    ):
        super().__init__()

        self.use_gazebo = use_gazebo
        self.randomizer = randomizer
        self.watchdog = watchdog
        self.watchdog_timeout_sec = (
            float(watchdog_timeout_sec)
            if watchdog_timeout_sec is not None
            else load_watchdog_timeout()
        )
        self._last_scan_monotonic = 0.0
        self.arena_size = float(arena_size)
        self.arena_half = self.arena_size / 2.0
        self.goal_tolerance = float(goal_tolerance)
        self.max_episode_steps = int(max_episode_steps)
        self.control_rate = float(control_rate)
        self.dt = 1.0 / self.control_rate  # 0.20s
        self.robot_radius = float(robot_radius)
        self.enable_motion_distortion = enable_motion_distortion

        # Reward parameters (frozen in specification Section 5)
        self.k_progress = 10.0
        self.c_step = 0.05
        self.r_goal = 100.0
        self.r_col = 100.0
        self.r_clear_coef = 0.0
        self.r_smooth_coef = 0.0

        # Action Space: continuous normalized u in [-1, 1]^2
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation Space: 43 continuous features
        # 36 LiDAR bins [0, 1], goal_dist [0, 1], sin/cos bearing [-1, 1],
        # v [-1, 1], omega [-1, 1], prev_action [-1, 1]^2
        low = np.array([0.0] * 36 + [0.0] + [-1.0] * 6, dtype=np.float32)
        high = np.array([1.0] * 36 + [1.0] + [1.0] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Observation preprocessor
        self.preprocessor = ObservationPreprocessor()

        # Random number generator
        self.np_random = np.random.default_rng(seed)

        # Simulation state
        self.step_count = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_omega = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.prev_goal_dist = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)

        # Pre-spawned obstacle pool matching arena_4x4.world
        self.obstacle_pool: List[Obstacle] = [
            Obstacle('box', 'obstacle_box_1', (0.30, 0.30)),
            Obstacle('box', 'obstacle_box_2', (0.25, 0.25)),
            Obstacle('box', 'obstacle_box_3', (0.35, 0.35)),
            Obstacle('box', 'obstacle_box_4', (0.30, 0.30)),
            Obstacle('cylinder', 'obstacle_cyl_1', (0.15,)),
            Obstacle('cylinder', 'obstacle_cyl_2', (0.12,)),
            Obstacle('cylinder', 'obstacle_cyl_3', (0.18,)),
            Obstacle('cylinder', 'obstacle_cyl_4', (0.15,)),
        ]

        # ROS 2 / Gazebo interface (lazily initialized if use_gazebo is True)
        self.ros_node = None
        self._latest_scan = None
        self._latest_odom = None
        self._latest_models = None
        if self.use_gazebo:
            self._init_gazebo_interface()

    def _init_gazebo_interface(self):
        """Initialize ROS 2 node and subscribers/clients for Gazebo communication."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Twist
            from gazebo_msgs.msg import ModelStates
            from gazebo_msgs.srv import SetEntityState
            from std_srvs.srv import Empty

            if not rclpy.ok():
                rclpy.init()

            self.ros_node = Node('tb3_navigation_gym_env')

            qos_scan = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            qos_odom = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            self._latest_scan = None
            self._latest_odom = None
            self._latest_models = None

            self.sub_scan = self.ros_node.create_subscription(
                LaserScan, '/scan', self._scan_cb, qos_scan
            )
            self.sub_odom = self.ros_node.create_subscription(
                Odometry, '/odom', self._odom_cb, qos_odom
            )
            self.sub_models = self.ros_node.create_subscription(
                ModelStates, '/gazebo/model_states', self._model_cb, 10
            )
            self.pub_cmd = self.ros_node.create_publisher(Twist, '/cmd_vel', 10)
            self.cli_set_state = self.ros_node.create_client(
                SetEntityState, '/gazebo/set_entity_state'
            )
            self.cli_pause = self.ros_node.create_client(
                Empty, '/pause_physics'
            )
            self.cli_unpause = self.ros_node.create_client(
                Empty, '/unpause_physics'
            )
        except Exception as e:
            self.use_gazebo = False
            self.ros_node = None

    def _scan_cb(self, msg):
        self._latest_scan = msg
        self._last_scan_monotonic = time.monotonic()
        if self.watchdog is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.watchdog.record_scan_received(stamp)

    def _odom_cb(self, msg):
        self._latest_odom = msg

    def _model_cb(self, msg):
        self._latest_models = msg

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state according to project specification.

        Sequence:
        1. Reseed if seed provided.
        2. Sample start pose and goal pose with clearance and distance constraints.
        3. Sample 4 to 8 obstacles with overlap rejection.
        4. If using Gazebo, teleport robot and obstacles via SetEntityState.
        5. Reset estimated-pose model and observation preprocessor.
        6. Return initial observation and info.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        # 0. Sample per-episode domain randomization parameters if active (and not pinned)
        if self.randomizer is not None and hasattr(self.randomizer, 'sample_episode'):
            if not getattr(self.randomizer, '_is_pinned_corner', False):
                self.randomizer.sample_episode()

        # 1. Sample robot start pose (cleared from boundary walls >= 0.40m)
        spawn_bound = self.arena_half - 0.40 - self.robot_radius
        self.robot_x = float(self.np_random.uniform(-spawn_bound, spawn_bound))
        self.robot_y = float(self.np_random.uniform(-spawn_bound, spawn_bound))
        self.robot_theta = float(self.np_random.uniform(-np.pi, np.pi))
        self.robot_v = 0.0
        self.robot_omega = 0.0

        # 2. Sample goal with start-goal distance in [1.5, 3.5] m and wall clearance >= 0.4m
        goal_found = False
        for _ in range(500):
            dist = float(self.np_random.uniform(1.5, 3.5))
            angle = float(self.np_random.uniform(-np.pi, np.pi))
            gx = self.robot_x + dist * math.cos(angle)
            gy = self.robot_y + dist * math.sin(angle)
            if abs(gx) <= (self.arena_half - 0.40) and abs(gy) <= (self.arena_half - 0.40):
                self.goal_x = gx
                self.goal_y = gy
                goal_found = True
                break
        if not goal_found:
            # Fallback to bounded coordinate
            self.goal_x = float(np.clip(-self.robot_x, -1.5, 1.5))
            self.goal_y = float(np.clip(-self.robot_y, -1.5, 1.5))

        self.prev_goal_dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

        # 3. Sample 4 to 8 obstacles with overlap rejection
        num_active = int(self.np_random.integers(4, 9))
        permuted_obstacles = list(self.np_random.permutation(self.obstacle_pool))

        placed_obstacles: List[Obstacle] = []
        for obs in self.obstacle_pool:
            obs.active = False
            obs.x = 20.0
            obs.y = 20.0

        for i in range(num_active):
            obs = permuted_obstacles[i]
            obs_radius = obs.radius
            placed = False
            for _ in range(200):
                # Bounds within arena walls
                obs_bound = self.arena_half - 0.15 - obs_radius
                ox = float(self.np_random.uniform(-obs_bound, obs_bound))
                oy = float(self.np_random.uniform(-obs_bound, obs_bound))

                # Clearance from robot start
                if math.hypot(ox - self.robot_x, oy - self.robot_y) < (0.40 + self.robot_radius + obs_radius):
                    continue
                # Clearance from goal
                if math.hypot(ox - self.goal_x, oy - self.goal_y) < (0.40 + obs_radius):
                    continue
                # Clearance from already placed obstacles
                overlap = False
                for p_obs in placed_obstacles:
                    if math.hypot(ox - p_obs.x, oy - p_obs.y) < (obs_radius + p_obs.radius + 0.10):
                        overlap = True
                        break
                if overlap:
                    continue

                obs.x = ox
                obs.y = oy
                obs.active = True
                placed_obstacles.append(obs)
                placed = True
                break

            if not placed:
                obs.active = False

        # 4. Teleport via Gazebo services if active
        if self.use_gazebo and self.ros_node is not None:
            import rclpy
            prev_stamp = None
            if self._latest_scan is not None:
                prev_stamp = self._latest_scan.header.stamp.sec + self._latest_scan.header.stamp.nanosec * 1e-9

            self._teleport_gazebo_entities()

            # Wait for fresh scan post-teleport (up to watchdog_timeout_sec)
            t_wait = time.monotonic()
            while (time.monotonic() - t_wait) < self.watchdog_timeout_sec:
                rclpy.spin_once(self.ros_node, timeout_sec=0.02)
                if self._latest_scan is not None:
                    curr_stamp = self._latest_scan.header.stamp.sec + self._latest_scan.header.stamp.nanosec * 1e-9
                    if prev_stamp is None or curr_stamp > prev_stamp:
                        self._last_scan_monotonic = time.monotonic()
                        if self.watchdog is not None:
                            self.watchdog.record_scan_received(curr_stamp)
                        break

        # 5. Reset observation preprocessor and estimated pose
        self.preprocessor.reset(
            init_x=self.robot_x,
            init_y=self.robot_y,
            init_theta=self.robot_theta,
        )

        # 6. Generate initial observation
        raw_ranges = self._get_lidar_ranges()
        v_meas = self.robot_v
        omega_meas = self.robot_omega
        if self.randomizer is not None:
            sens_rand = getattr(self.randomizer, 'sensor', self.randomizer)
            if hasattr(sens_rand, 'perturb_scan'):
                raw_ranges = sens_rand.perturb_scan(raw_ranges)
            if hasattr(sens_rand, 'perturb_odometry'):
                v_meas, omega_meas = sens_rand.perturb_odometry(v_meas, omega_meas)

        obs = self.preprocessor.step(
            ranges=raw_ranges,
            v_meas=v_meas,
            omega_meas=omega_meas,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            last_action=self.prev_action,
            dt=self.dt,
        )

        info = {
            "step_count": self.step_count,
            "goal_distance": self.prev_goal_dist,
            "is_success": False,
            "is_collision": False,
            "num_obstacles": len(placed_obstacles),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance environment by one control step (0.2s zero-order hold).

        Args:
            action: Continuous action u in [-1, 1]^2.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # 1. Action mapping (Frozen in specification Section 2)
        # u in [-1, 1]^2 -> v in [0, 0.15], omega in [-0.6, 0.6]
        u = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        cmd_v = float(0.075 * (u[0] + 1.0))
        cmd_omega = float(0.60 * u[1])

        # If dynamics randomizer active, perturb commanded action (lag, delay, gain, limits)
        eff_v = cmd_v
        eff_omega = cmd_omega
        if self.randomizer is not None:
            dyn_rand = getattr(self.randomizer, 'dynamics', self.randomizer)
            if hasattr(dyn_rand, 'perturb_action') and hasattr(dyn_rand, 'params'):
                prev_u = self.prev_action
                prev_v = float(0.075 * (prev_u[0] + 1.0))
                prev_omega = float(0.60 * prev_u[1])
                eff_v, eff_omega = dyn_rand.perturb_action(cmd_v, cmd_omega, prev_v, prev_omega, dt=self.dt)

        # 2. Advance physics
        if self.use_gazebo and self.ros_node is not None:
            self._step_gazebo(eff_v, eff_omega)
        else:
            self._step_internal_kinematics(eff_v, eff_omega)

        # 3. Ground-truth evaluation
        current_goal_dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        is_goal = current_goal_dist <= self.goal_tolerance
        is_collision = self._check_collision()

        # 4. Termination vs Truncation (Gymnasium semantics)
        # Goal or collision -> terminated
        # Timeout (step_count >= 350) -> truncated
        terminated = bool(is_goal or is_collision)
        is_timeout = (self.step_count >= self.max_episode_steps)
        truncated = bool(is_timeout and not terminated)

        # 5. Reward calculation (Frozen in specification Section 5)
        # r_t = 10*(d_t - d_{t+1}) - 0.05 + 100*[goal] - 100*[collision]
        progress_reward = self.k_progress * (self.prev_goal_dist - current_goal_dist)
        step_cost = -self.c_step
        goal_reward = self.r_goal if is_goal else 0.0
        collision_penalty = -self.r_col if is_collision else 0.0

        reward = float(progress_reward + step_cost + goal_reward + collision_penalty)
        self.prev_goal_dist = current_goal_dist

        # 6. Sensor reading and observation construction
        raw_ranges = self._get_lidar_ranges()
        if self.enable_motion_distortion:
            effective_ranges = apply_motion_distortion(
                raw_ranges,
                v=self.robot_v,
                omega=self.robot_omega,
                scan_time=self.dt,
            )
        else:
            effective_ranges = raw_ranges

        # Measured velocities (nominal measurement model under P1)
        v_meas = self.robot_v
        omega_meas = self.robot_omega

        # If sensor randomizer active, perturb ranges and odometry measurements
        if self.randomizer is not None:
            sens_rand = getattr(self.randomizer, 'sensor', self.randomizer)
            if hasattr(sens_rand, 'perturb_scan'):
                effective_ranges = sens_rand.perturb_scan(effective_ranges)
            if hasattr(sens_rand, 'perturb_odometry'):
                v_meas, omega_meas = sens_rand.perturb_odometry(v_meas, omega_meas)

        obs = self.preprocessor.step(
            ranges=effective_ranges,
            v_meas=v_meas,
            omega_meas=omega_meas,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            last_action=u,
            dt=self.dt,
        )
        self.prev_action = u.copy()

        info = {
            "step_count": self.step_count,
            "goal_distance": current_goal_dist,
            "is_success": is_goal,
            "is_collision": is_collision,
            "reward_components": {
                "progress": progress_reward,
                "step_penalty": step_cost,
                "goal": goal_reward,
                "collision": collision_penalty,
            },
            "ground_truth": {
                "robot_pose": (self.robot_x, self.robot_y, self.robot_theta),
                "goal_pose": (self.goal_x, self.goal_y),
            },
        }

        if truncated:
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info

    def _step_internal_kinematics(self, cmd_v: float, cmd_omega: float) -> None:
        """Internal kinematic differential-drive stepping over dt."""
        self.robot_v = cmd_v
        self.robot_omega = cmd_omega

        delta_theta = cmd_omega * self.dt
        theta_mid = self.robot_theta + 0.5 * delta_theta

        self.robot_x += cmd_v * self.dt * math.cos(theta_mid)
        self.robot_y += cmd_v * self.dt * math.sin(theta_mid)
        self.robot_theta = wrap_to_pi(self.robot_theta + delta_theta)

    def _check_collision(self) -> bool:
        """Check ground-truth collision against 4m walls and active obstacles."""
        # 1. Boundary walls collision (inner wall boundary at +/- (arena_half - 0.05))
        wall_limit = self.arena_half - 0.05 - self.robot_radius
        if abs(self.robot_x) >= wall_limit or abs(self.robot_y) >= wall_limit:
            return True

        # 2. Obstacles collision
        for obs in self.obstacle_pool:
            if obs.check_collision(self.robot_x, self.robot_y, self.robot_radius):
                return True

        return False

    def _get_lidar_ranges(self) -> np.ndarray:
        """Get 360-beam LiDAR ranges (simulated or from Gazebo /scan topic)."""
        if self.use_gazebo and self._latest_scan is not None:
            return np.array(self._latest_scan.ranges, dtype=np.float32)

        # Internal analytical 360-beam 2D raycaster against walls and obstacles
        ranges = np.full(360, RANGE_MAX, dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

        wall_min = -self.arena_half + 0.05
        wall_max = self.arena_half - 0.05

        rx = self.robot_x
        ry = self.robot_y
        r_theta = self.robot_theta

        for k in range(360):
            ray_angle = wrap_to_pi(r_theta + angles[k])
            ca = math.cos(ray_angle)
            sa = math.sin(ray_angle)

            # Intersect with 4 arena walls
            d_wall = RANGE_MAX
            if ca > 1e-6:
                d_wall = min(d_wall, (wall_max - rx) / ca)
            elif ca < -1e-6:
                d_wall = min(d_wall, (wall_min - rx) / ca)

            if sa > 1e-6:
                d_wall = min(d_wall, (wall_max - ry) / sa)
            elif sa < -1e-6:
                d_wall = min(d_wall, (wall_min - ry) / sa)

            min_dist = max(0.0, d_wall)

            # Intersect with all active obstacles
            for obs in self.obstacle_pool:
                if obs.active:
                    d_obs = obs.ray_intersect(rx, ry, ray_angle, min_dist)
                    if d_obs < min_dist:
                        min_dist = d_obs

            # Clip to valid range [range_min, range_max]
            if min_dist < 0.12:
                ranges[k] = 0.12
            elif min_dist >= RANGE_MAX:
                ranges[k] = float('inf')
            else:
                ranges[k] = float(min_dist)

        return ranges

    def _teleport_gazebo_entities(self):
        """Teleport robot and obstacles via Gazebo SetEntityState service.

        To prevent ODE penetration repulsion impulses:
        1. Pause Gazebo physics.
        2. Teleport all obstacles to their target (or out-of-bounds) positions first.
        3. Teleport robot to (x, y, z=0.012) with explicitly zeroed linear & angular velocities.
        4. Unpause Gazebo physics and allow a brief settle time.
        """
        try:
            import rclpy
            from gazebo_msgs.srv import SetEntityState
            from std_srvs.srv import Empty
            from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

            if not self.cli_set_state.wait_for_service(timeout_sec=1.0):
                return

            # 1. Zero out any active velocity command
            stop_msg = Twist()
            self.pub_cmd.publish(stop_msg)

            # 2. Pause physics
            if hasattr(self, 'cli_pause') and self.cli_pause.wait_for_service(timeout_sec=0.2):
                fut_pause = self.cli_pause.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self.ros_node, fut_pause, timeout_sec=0.2)

            # 3. Teleport obstacles FIRST so previous obstacles are cleared from robot's new spot
            for obs in self.obstacle_pool:
                req_obs = SetEntityState.Request()
                req_obs.state.name = obs.name
                req_obs.state.pose.position = Point(x=obs.x, y=obs.y, z=0.25)
                req_obs.state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                req_obs.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
                req_obs.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
                req_obs.state.reference_frame = 'world'
                fut_obs = self.cli_set_state.call_async(req_obs)
                rclpy.spin_until_future_complete(self.ros_node, fut_obs, timeout_sec=0.1)

            # 4. Teleport robot with explicitly zeroed linear & angular velocity
            req = SetEntityState.Request()
            req.state.name = 'burger'
            req.state.pose.position = Point(x=self.robot_x, y=self.robot_y, z=0.012)
            half_th = self.robot_theta * 0.5
            req.state.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=math.sin(half_th), w=math.cos(half_th)
            )
            req.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
            req.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
            req.state.reference_frame = 'world'

            future = self.cli_set_state.call_async(req)
            rclpy.spin_until_future_complete(self.ros_node, future, timeout_sec=0.5)

            # 5. Unpause physics
            if hasattr(self, 'cli_unpause') and self.cli_unpause.wait_for_service(timeout_sec=0.2):
                fut_unpause = self.cli_unpause.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self.ros_node, fut_unpause, timeout_sec=0.2)

            # 6. Settle physics and update internal state
            self.robot_v = 0.0
            self.robot_omega = 0.0
            rclpy.spin_once(self.ros_node, timeout_sec=0.05)
        except Exception as e:
            print(f"DEBUG: _teleport_gazebo_entities exception: {type(e).__name__}: {e}")

    def _step_gazebo(self, cmd_v: float, cmd_omega: float):
        """Send cmd_vel and wait for next scan in Gazebo."""
        import rclpy
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = cmd_v
        msg.angular.z = cmd_omega
        self.pub_cmd.publish(msg)

        prev_stamp = None
        if self._latest_scan is not None:
            prev_stamp = self._latest_scan.header.stamp.sec + self._latest_scan.header.stamp.nanosec * 1e-9

        # Spin to process callbacks until fresh scan arrives within configured watchdog timeout
        start_t = time.monotonic()
        scan_received = False
        while (time.monotonic() - start_t) < self.watchdog_timeout_sec:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)
            if self._latest_scan is not None:
                curr_stamp = self._latest_scan.header.stamp.sec + self._latest_scan.header.stamp.nanosec * 1e-9
                if prev_stamp is None or curr_stamp > prev_stamp:
                    scan_received = True
                    self._last_scan_monotonic = time.monotonic()
                    if self.watchdog is not None:
                        self.watchdog.record_scan_received(curr_stamp)
                    break

        if not scan_received:
            if self.watchdog is not None:
                self.watchdog.handle_missing_scan(
                    self,
                    current_step=self.step_count,
                    current_episode=getattr(self, 'episode_count', 0),
                )
            else:
                raise MissingScanTimeoutError(
                    f"No fresh LiDAR scan received within watchdog timeout {self.watchdog_timeout_sec:.2f}s"
                )

        if self._latest_odom is not None:
            self.robot_v = self._latest_odom.twist.twist.linear.x
            self.robot_omega = self._latest_odom.twist.twist.angular.z

        if self._latest_models is not None and 'burger' in self._latest_models.name:
            idx = self._latest_models.name.index('burger')
            pose = self._latest_models.pose[idx]
            self.robot_x = pose.position.x
            self.robot_y = pose.position.y
            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    def close(self):
        """Clean up resources."""
        if self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass
        super().close()
