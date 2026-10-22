# Setup Log: `sim-real-rl` Project

All environment setup, dependency installations, and workspace builds are recorded here chronologically.

---

## Stage 1: System Inspection (Read-Only)
* **Date**: 2026-10-21
* **Objective**: Audit host environment, ROS distribution, hardware resources, existing workspaces, and Gazebo Classic availability.

### Commands Run & Results
1. `lsb_release -a; uname -r; nproc; lscpu | head -20; free -h; df -h ~ /; nvidia-smi`
   * **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish), Kernel `6.8.0-138-generic`
   * **CPU**: Intel Core i7-14650HX (16 physical cores, 24 vCPUs, up to 5.2 GHz)
   * **RAM**: 15 GiB total (2.0 GiB free, 4.5 GiB available for processes), 15 GiB swap
   * **GPU**: NVIDIA GeForce RTX 4060 Laptop (8 GiB VRAM), Driver `580.178.04`, CUDA 13.0
   * **Disk Space**:
     * Filesystem `/dev/nvme0n1p7` mounted on `/` and `~`
     * 260 GiB used / 285 GiB total (97% used)
     * **11 GiB available** -> **Below user threshold of 15 GiB**. Action: Minimize installation sizes, use CPU-only PyTorch wheel.
2. `printenv ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH TURTLEBOT3_MODEL ROS_DOMAIN_ID RMW_IMPLEMENTATION`
   * `ROS_DISTRO`: `humble`
   * `ROS_DOMAIN_ID`: `30`
   * `RMW_IMPLEMENTATION`: `rmw_cyclonedds_cpp`
   * `TURTLEBOT3_MODEL`: unset
   * `AMENT_PREFIX_PATH`: Included overlays from `~/turtlebot3_ws/install` and `~/ros2_ws/install` alongside `/opt/ros/humble`.
3. `which ros2 colcon rosdep vcs git pdftotext; python3 --version; python3 -m venv --help | head -1`
   * All tools present in `/usr/bin` or `/opt/ros/humble/bin`.
   * Python: `3.10.12`, `venv` module operational.
4. `ros2 doctor --report`
   * Verified ROS 2 Humble distribution, active DDS configuration.
5. `gazebo --version`
   * Exited with code 127 (`command not found`). Gazebo Classic binary is not installed on system PATH.
6. `dpkg -l | grep -E '^ii +ros-humble-(turtlebot3|nav2|navigation2|slam-toolbox|gazebo|dynamixel|xacro)'`
   * `ros-humble-navigation2` & `ros-humble-nav2-*`: Installed (`1.1.20-1jammy`).
   * `ros-humble-slam-toolbox`: Installed (`2.6.10-1jammy`).
   * `ros-humble-xacro`: Installed (`2.1.1-1jammy`).
   * `ros-humble-gazebo-msgs`: Installed (`3.9.0-1jammy`).
   * `ros-humble-gazebo-ros` / `gazebo_ros_pkgs` / `gazebo11`: **Missing**.
   * `ros-humble-turtlebot3*`: **Not installed via apt**.
7. `for p in turtlebot3_gazebo turtlebot3_description turtlebot3_msgs gazebo_ros nav2_bringup slam_toolbox; do echo "== $p"; ros2 pkg prefix $p; done`
   * `turtlebot3_gazebo`, `turtlebot3_description`, `turtlebot3_msgs` resolved to `~/turtlebot3_ws/install/`.
   * `gazebo_ros`: `Package not found`.
   * `nav2_bringup`, `slam_toolbox`: `/opt/ros/humble`.
8. `find ~ -maxdepth 4 -type d -name install -not -path '*/.cache/*' 2>/dev/null`
   * Workspaces found: `~/turtlebot3_ws/install`, `~/ros2_ws/install`, `~/rclcpp_dev_ws/install`, `~/tb3_swarm_ws/install`.
9. `grep -n -i 'ros\|source' ~/.bashrc`
   * Auto-sources `turtlebot3_ws`, `/usr/share/gazebo/setup.sh`, `ros2_ws`, and `Warehouse_Simulation`.

---

## Stage 2: Project Creation
* **Date**: 2026-10-21
* **Objective**: Initialize project directory structure, git repository, `.gitignore`, documentation templates, and copy the specification PDF.

### Actions Executed
1. Directory structure created:
   * `ros2_ws/src/`
   * `rl/{env,train,evaluate,analysis}/`
   * `baselines/`
   * `configs/`
   * `scenarios/{train,validation,test,stress}/`
   * `deploy/`
   * `tests/`
   * `runs/`
   * `docs/`
2. `.gitkeep` placeholders added to empty directories.
3. Spec PDF copied from `/home/saigirish/Downloads/Sim2Real_LiDAR_Navigation_Pipeline.pdf` to `docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`.
4. `.gitignore` created to exclude `.venv/`, ROS build/install/log, runs/, checkpoints, model weights, and upstream source clones.
5. `README.md`, `docs/setup_log.md`, and `docs/decisions.md` created.
6. Git repository initialized with `git init`.

---

## Stage 3: Dependencies, Workspace Build, Python Environment
* **Date**: 2026-10-22
* **Objective**: Setup virtual environment with ROS compatibility, install CPU-only PyTorch and RL libraries, identify missing Gazebo Classic APT packages.

### Actions Executed
1. Python virtual environment created:
   `python3 -m venv --system-site-packages /home/saigirish/sim-real-rl/.venv`
   * Verified `import rclpy` succeeds from `/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy`.
2. Installed CPU-only PyTorch and RL stack:
   * `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu` from `https://download.pytorch.org/whl/cpu`.
   * `stable-baselines3==2.7.1`
   * `gymnasium==1.2.3`
   * `numpy==1.26.4` (pinned `<2.0`)
   * `pytest==8.4.2`
   * `pyyaml==5.4.1`
3. Verified full import suite: `rclpy`, `torch`, `gymnasium`, `stable_baselines3`, `numpy`, `yaml`, `pytest` all load without error.
4. Generated `docs/environment.md` with complete pinned stack and activation workflow.
5. Installed missing Gazebo Classic and TurtleBot3 packages via APT:
   * `sudo apt-get install -y gazebo ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3-description ros-humble-turtlebot3-msgs ros-humble-turtlebot3-gazebo`
   * Verified: `gazebo 11.10.2` and `gazebo_ros` in `/opt/ros/humble`.

---

## Stage 4: Simulation Environment, Topic Audit & Benchmark (Day 1 Complete)
* **Date**: 2026-10-22
* **Objective**: Create `tb3_rl_sim` package, implement 4m×4m arena world with pre-spawned obstacle pool and `gazebo_ros_state`, build headless launch system, audit topic QoS & LaserScan metadata, and benchmark single-worker simulation throughput (Gate G1).

### Actions Executed
1. Created ROS 2 package `tb3_rl_sim` in `ros2_ws/src/tb3_rl_sim`:
   * `package.xml` and `CMakeLists.txt` configured with dependencies.
   * Built and installed via `colcon build --symlink-install`.
2. Created SDF World `worlds/arena_4x4.world`:
   * 4.0 m × 4.0 m boundary walls (centered at origin, 0.5 m height, static).
   * Pre-spawned obstacle pool (4 boxes 0.25-0.35 m, 4 cylinders r=0.12-0.18 m) parked outside the arena for instantaneous reset teleportation.
   * Loaded `libgazebo_ros_state.so` WorldPlugin for `/gazebo/set_entity_state` and `/gazebo/get_entity_state`.
3. Created Headless Launch `launch/headless_sim.launch.py`:
   * Runs `gzserver` (no `gzclient` GUI) with `-s libgazebo_ros_init.so -s libgazebo_ros_factory.so`.
   * Sets `GAZEBO_MODEL_DATABASE_URI=''` to avoid network stall on deprecated Gazebo model servers.
   * Sets `robot_state_publisher` and spawner for TurtleBot3 Burger (`TURTLEBOT3_MODEL=burger`).
4. Verified Topics & Services:
   * Topics: `/scan` (5 Hz), `/odom`, `/cmd_vel`, `/clock`, `/gazebo/model_states`, `/gazebo/link_states`.
   * Services: `/pause_physics`, `/unpause_physics`, `/gazebo/set_entity_state`, `/gazebo/get_entity_state`.
5. Topic QoS & LaserScan Metadata Audit:
   * `/scan`: 360 beams (1° resolution), `angle_min=0.0`, `angle_max=6.28`, `range_min=0.12 m`, `range_max=3.5 m`.
   * Invalid / out-of-range encoding verified as `inf`.
   * `/scan` QoS: `RELIABLE`, `KEEP_LAST (5)`.
   * `/odom` QoS: `RELIABLE`, `KEEP_LAST (1)`.
   * `/clock` QoS: `BEST_EFFORT`, `KEEP_LAST (1)`.
   * Recorded in `configs/env.yaml`.
6. Single-Worker Throughput Benchmark (`tests/benchmark_sim.py`):
   * Control step rate: 5.10 steps/s (lockstep with 5.0 Hz scan).
   * Real-Time Factor (RTF): 1.01x (real-time synchronized).
   * Host CPU usage: 4.8% on Intel Core i7-14650HX.
   * Host RAM available: 6.22 GiB (stable, no memory growth).
   * **Gate G1 Exit Status**: **PASSED**.

---

## Stage 5: Gymnasium Environment v1 (Day 2 Milestone)
* **Date**: 2026-10-22
* **Objective**: Implement pure NumPy observation preprocessing with estimated-pose model, continuous action mapping, Gymnasium environment wrapper (`TurtleBot3NavigationEnv`), reward function, termination/truncation semantics, and execute the 10,000-random-step soak test.

### Actions Executed
1. Implemented Observation Preprocessing (`rl/env/observation.py`):
   * Pure NumPy with **zero ROS dependencies** (`no rclpy`).
   * Exactly 43 dimensions: 36 min-pooled 10° LiDAR bins (bin 0 centered on forward), goal distance (norm by arena diagonal 5.657m), goal bearing (sin/cos), measured velocities ($v/0.15, \omega/0.60$), previous action (2).
   * EstimatedPose dead-reckoning model integrated from reset odometry frame.
   * Per-beam motion distortion model for physical scan skew.
2. Implemented Gymnasium Environment (`rl/env/tb3_navigation_env.py`):
   * Continuous action space: $[-1, 1]^2 \to v \in [0, 0.15]\,\text{m/s}, \omega \in [-0.60, 0.60]\,\text{rad/s}$.
   * Obstacle pool placement (4-8 obstacles) with overlap rejection from robot, goal, and other obstacles.
   * Reward function: $r_t = 10(d_t - d_{t+1}) - 0.05 + 100 \cdot [\text{goal}] - 100 \cdot [\text{col}]$.
   * Gymnasium semantics: `terminated` for goal reached ($\le 0.20$m) and collision; `truncated` for timeout (350 steps).
   * Ground-truth isolation: privileged state strictly confined to reward/termination/evaluation.
3. Test Suites Implemented & Verified (`tests/test_observation.py`, `tests/test_env.py`):
   * 15/15 unit tests passed with pytest under `.venv`.
   * Passed Gymnasium `check_env` and Stable-Baselines3 `check_env`.
4. 10,000-Step Random Policy Soak Test (`tests/soak_random_steps.py`):
   * Completed 10,000 steps in 20.02 s (499.6 steps/s).
   * 101 episodes completed (1 goal, 100 collisions, 0 timeouts).
   * Zero crashes, zero non-finite/NaN/inf values.
   * Memory RSS change: +0.70 MB (< 2% growth).
   * **Day 2 Exit Gate Status**: **PASSED**.



