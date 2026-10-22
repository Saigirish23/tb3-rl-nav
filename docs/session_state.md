# Session Snapshot: `sim-real-rl` Project Setup

* **Date**: 2026-10-22
* **Current State**: Day 2 Complete (Stage 5 Verified: Observation Module, Estimated Pose, Gymnasium Environment v1, 10,000-Step Soak Passed) -> Ready for Day 3 (Gate G2 Checklist & Hand-Coded Reactive Controller)

---

## 1. Project & Workspace Status

* **Repository Root**: `/home/saigirish/sim-real-rl`
* **Specification**: Frozen per [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](file:///home/saigirish/sim-real-rl/docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf).
* **Tracked Decisions**: Recorded in [`docs/decisions.md`](file:///home/saigirish/sim-real-rl/docs/decisions.md).
* **Setup History**: Chronologically logged in [`docs/setup_log.md`](file:///home/saigirish/sim-real-rl/docs/setup_log.md).
* **Environment Spec & Scan Audit**: Pinned in [`configs/env.yaml`](file:///home/saigirish/sim-real-rl/configs/env.yaml) and [`docs/environment.md`](file:///home/saigirish/sim-real-rl/docs/environment.md).

---

## 2. Verified on Your Machine

1. **Host & Resources**:
   * **OS**: Ubuntu 22.04.5 LTS, Kernel `6.8.0-138-generic`.
   * **CPU**: Intel Core i7-14650HX (16 cores, 24 threads).
   * **RAM**: 15 GiB (6.2 GiB available).
   * **Disk Space**: ~16 GiB available on `/dev/nvme0n1p7`.
2. **ROS 2 & Middleware**:
   * ROS 2 Humble Hawksbill with `rmw_cyclonedds_cpp`, `ROS_DOMAIN_ID=30`.
   * Gazebo Classic `11.10.2` and `ros-humble-gazebo-ros-pkgs` (3.9.0) operational.
   * TurtleBot3 Burger packages (`turtlebot3_description`, `turtlebot3_gazebo`, `turtlebot3_msgs`) verified.
3. **Simulation Package (`tb3_rl_sim`)**:
   * Package built in `ros2_ws/src/tb3_rl_sim`.
   * World [`worlds/arena_4x4.world`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world): 4m×4m boundary walls, 8 pre-spawned obstacles (4 boxes, 4 cylinders) parked outside arena for instant reset teleportation, `libgazebo_ros_state.so` WorldPlugin loaded.
   * Launch [`launch/headless_sim.launch.py`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/launch/headless_sim.launch.py): Headless `gzserver` with offline model database configuration and Burger spawner.
4. **Gymnasium Environment Stack (`rl/env/`)**:
   * Observation module [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py): Pure NumPy with **zero ROS imports**. Exactly 43 dimensions, 10° min-pooling with bin 0 centered on forward, dead-reckoning estimated pose in reset odometry frame, and per-beam motion distortion.
   * Environment wrapper [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py): Full Gymnasium compliance. Continuous action space $[-1, 1]^2 \to [0, 0.15]\,\text{m/s}, [-0.6, 0.6]\,\text{rad/s}$. Reward function $r_t = 10(d_t - d_{t+1}) - 0.05 + 100 \cdot [\text{goal}] - 100 \cdot [\text{col}]$. Correct Gymnasium `terminated` (goal/collision) and `truncated` (timeout 350 steps) semantics. Ground-truth state strictly isolated from observation vector.
5. **Validation & Test Results**:
   * Unit tests: 15/15 passed in `tests/test_observation.py` and `tests/test_env.py`.
   * Gymnasium `check_env` and Stable-Baselines3 `check_env` passed.
   * 10,000-Step Random Policy Soak ([`tests/soak_random_steps.py`](file:///home/saigirish/sim-real-rl/tests/soak_random_steps.py)): 10,000 steps executed in 20.02 s (499.6 steps/s) with zero crashes, zero non-finite values, and +0.70 MB memory change.
   * **Day 2 Exit Gate Status**: **PASSED**.

---

## 3. Next Milestone: Day 3 (Pre-Training Gate G2 & Hand-Coded Baseline)

Per Section 9 & 10 of the specification:
1. **Hand-Coded Reactive Controller** (`baselines/reactive_controller.py`):
   * Pure reactive policy using the same 43-dim observation.
   * Turn proportionally to goal bearing, scale forward speed by $\cos(\text{bearing})$.
   * Obstacle avoidance: if min range in $\pm 30^\circ$ front sector $< 0.5$ m, turn toward side with larger mean range; if $< 0.25$ m, stop forward motion and rotate.
   * Benchmark against nominal scenarios (target $\ge 90\%$ success).
2. **Scenario Generator** (`scenarios/`):
   * Generate reproducible evaluation and training layout seeds with disjoint seed sequence.
3. **Pre-Training Gate G2 Checklist (Checks 1-10)**:
   * 1,000-reset soak test (pose within 2 cm and 1°, zero velocity, fresh scan).
   * Timing and preprocessing parity tests.
   * Reward audit across stationary, random, spin, and hand-coded traces.
