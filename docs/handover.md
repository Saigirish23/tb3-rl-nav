# Teammate Handover & Onboarding Guide: `sim-real-rl`

This document outlines the setup, architecture, and operational workflows for the **`sim-real-rl`** project. It is intended for seamless collaboration between co-owners (**Saigirish S** and **Siddhi Vinayak**) per the specification in [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](Sim2Real_LiDAR_Navigation_Pipeline.pdf).

---

## 1. Project Context & Current Milestone

* **Goal**: Mapless goal navigation with a PPO policy on a TurtleBot3 Burger, evaluating the effect of sensor and dynamics domain randomization in a controlled 2×2 study (P1 nominal, P2 sensor rand, P3 dynamics rand, P4 combined).
* **Current Status**: **Day 1 & Day 2 Completed**.
  * **Gate G1 Passed**: Headless simulation throughput known and benchmarked.
  * **Day 2 Gate Passed**: Gymnasium Environment v1 implemented, verified with 15/15 unit tests and a 10,000-step random soak test.
* **Division of Responsibilities**:
  * **Saigirish S**: Gazebo/Gymnasium environment, PPO training, reward, domain randomizers (P2/P3), simulation benchmark.
  * **Siddhi Vinayak**: Hand-coded baseline controller, evaluation harness, ROS 2 deployment node, hardware trials and analysis.

---

## 2. Host System Prerequisites

* **OS**: Ubuntu 22.04 LTS (Jammy Jellyfish), Kernel 6.8.x.
* **ROS 2 Distribution**: ROS 2 Humble Hawksbill (`/opt/ros/humble`).
* **Middleware**: `rmw_cyclonedds_cpp` with `ROS_DOMAIN_ID=30`.
* **Simulator**: **Gazebo Classic 11.10.x** (`gazebo_ros_pkgs`).
  > [!WARNING]
  > Do **NOT** install Modern Gazebo / Ignition / `ros_gz`. The specification strictly pins Gazebo Classic 11 with `libgazebo_ros_init.so` and `libgazebo_ros_state.so`.

---

## 3. Environment Setup (Step-by-Step)

### Step 3.1: Install System Packages via APT
Ensure Gazebo Classic, ROS 2 Gazebo bridges, and TurtleBot3 packages are installed:

```bash
sudo apt-get update && sudo apt-get install -y \
  gazebo \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-turtlebot3-description \
  ros-humble-turtlebot3-msgs \
  ros-humble-turtlebot3-gazebo
```

### Step 3.2: Configure Python Virtual Environment (`.venv`)
The Python virtual environment must be configured with `--system-site-packages` so that ROS 2 Python bindings (`rclpy`) are accessible directly from `/opt/ros/humble`:

```bash
cd ~/sim-real-rl

# 1. Create venv with system-site-packages
python3 -m venv --system-site-packages .venv

# 2. Activate venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install CPU-only PyTorch (saves ~3 GB disk space; spec pins CPU training)
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 5. Install RL & math libraries (pinned per docs/environment.md)
pip install "numpy<2.0,>=1.26.0" "stable-baselines3==2.7.1" "gymnasium==1.2.3" pytest pyyaml psutil
```

### Step 3.3: Verify Python Imports
Test that all dependencies load cleanly:

```bash
python -c "import rclpy, torch, stable_baselines3, gymnasium, numpy, yaml; print('Environment OK!')"
```

---

## 4. How TurtleBot3 Packages Work & How to Access Them

### Where TurtleBot3 Directories Live
TurtleBot3 packages can exist in **two locations** on the host machine:

1. **System Binary Path (`/opt/ros/humble/share/`)**:
   * Installed automatically via APT in Step 3.1.
   * Paths:
     * `/opt/ros/humble/share/turtlebot3_description` (URDFs and joint models)
     * `/opt/ros/humble/share/turtlebot3_gazebo` (models, SDFs, worlds)
     * `/opt/ros/humble/share/turtlebot3_msgs` (custom interfaces)

2. **Local Source Workspace (`~/turtlebot3_ws/install/`)**:
   * If a developer built TurtleBot3 from source in their home directory.
   * Paths:
     * `~/turtlebot3_ws/install/turtlebot3_description/share/turtlebot3_description`
     * `~/turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo`

### How ROS 2 Resolves Package Paths (`AMENT_PREFIX_PATH`)
ROS 2 uses the `AMENT_PREFIX_PATH` environment variable. When multiple workspaces are sourced, workspaces sourced **later** prepend to the search path:
* If `~/.bashrc` sources `~/turtlebot3_ws/install/setup.bash`, ROS 2 prioritizes `~/turtlebot3_ws/`.
* If you only source `/opt/ros/humble/setup.bash`, ROS 2 resolves to `/opt/ros/humble/share/`.
* *Both directory trees contain identical Burger SDF and URDF files.*

### How to Use TurtleBot3 Directories in Simulation
Our launch file [`ros2_ws/src/tb3_rl_sim/launch/headless_sim.launch.py`](../ros2_ws/src/tb3_rl_sim/launch/headless_sim.launch.py) dynamically resolves the package location via `get_package_share_directory('turtlebot3_gazebo')`.

To ensure Gazebo finds the models without manual copying, the following environment variables are set:

```bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_DATABASE_URI=""
```

> [!IMPORTANT]
> Setting `GAZEBO_MODEL_DATABASE_URI=""` is **essential**. Gazebo Classic's default online model server (`http://models.gazebosim.org`) is deprecated and times out for 30+ seconds on startup if not disabled. Setting it to empty forces Gazebo to use local models instantly.

---

## 5. ROS 2 Simulation Package: `tb3_rl_sim`

The simulation package lives in [`ros2_ws/src/tb3_rl_sim`](../ros2_ws/src/tb3_rl_sim).

### Build the Simulation Workspace
```bash
cd ~/sim-real-rl/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### World File Architecture: `arena_4x4.world`
Located at [`ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`](../ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world):
* **Arena**: 4.0 m × 4.0 m square with 0.5 m static boundary walls centered at $(0, 0)$.
* **Pre-Spawned Obstacle Pool**: 8 obstacles (4 boxes of sizes 0.25–0.35 m, 4 cylinders of radii 0.12–0.18 m) are pre-defined in the SDF world.
  * Initially parked outside the arena ($x \ge 10.0$ m).
  * During reset, active obstacles (4–8) are instantly teleported into sampled arena poses using `/gazebo/set_entity_state`, while unused obstacles remain parked. This eliminates the multi-second latency and physics glitches of dynamic model spawning/deletion.
* **Plugins**:
  * `libgazebo_ros_init.so`: Provides `/pause_physics`, `/unpause_physics`, `/reset_simulation`.
  * `libgazebo_ros_state.so`: Provides `/gazebo/set_entity_state` and `/gazebo/get_entity_state`.

### Launching Headless Simulation
```bash
source /usr/share/gazebo/setup.sh
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_DATABASE_URI=""

ros2 launch tb3_rl_sim headless_sim.launch.py
```

### Audited Topic & QoS Contracts
Audited and pinned in [`configs/env.yaml`](../configs/env.yaml):
* `/scan`: 360 beams ($1^\circ$ increment), 5.0 Hz, `range_min=0.12m`, `range_max=3.50m`, invalid/free space mapped to `inf`. QoS: `RELIABLE`, depth 5.
* `/odom`: Measured velocity and pose. QoS: `RELIABLE`, depth 1.
* `/cmd_vel`: `geometry_msgs/msg/Twist`.
* `/clock`: Gazebo simulation clock. QoS: `BEST_EFFORT`, depth 1.

---

## 6. RL Environment Stack (`rl/env/`)

### Observation Preprocessing: `rl/env/observation.py`
* **Zero ROS dependencies** (`no rclpy`): Pure NumPy implementation to ensure 100% sim-to-real code parity between simulation and the hardware deployment node.
* **43 Dimensions**:
  * `[0:36]`: 36 min-pooled 10° LiDAR bins (bin 0 centered on forward $[-5^\circ, +5^\circ)$, normalized $[0, 1]$).
  * `[36]`: Goal distance normalized by arena diagonal ($5.657$ m).
  * `[37:39]`: Goal bearing represented as $[\sin(\beta), \cos(\beta)]$.
  * `[39]`: Measured forward velocity $v / 0.15 \in [0, 1]$.
  * `[40]`: Measured angular velocity $\omega / 0.60 \in [-1, 1]$.
  * `[41:43]`: Last normalized action $u \in [-1, 1]^2$ (zeros at reset).
* **EstimatedPose Model**: Integrates measured $(v_m, \omega_m)$ dead-reckoning from the reset odometry frame. Eliminates artificial per-step observation noise while allowing realistic odometry drift to accumulate.
* **Motion Distortion**: `apply_motion_distortion()` models per-beam scan skew caused by robot motion during the 0.2s LiDAR rotation.

### Gymnasium Environment: `rl/env/tb3_navigation_env.py`
* Standard Gymnasium API (`step`, `reset`, `action_space`, `observation_space`).
* **Continuous Action Space**: $u \in [-1, 1]^2 \to v \in [0.0, 0.15]\,\text{m/s}, \omega \in [-0.60, 0.60]\,\text{rad/s}$ (forward-only bias).
* **Reward**:
  $$r_t = 10(d_t - d_{t+1}) - 0.05 + 100 \cdot [\text{goal}] - 100 \cdot [\text{collision}]$$
* **Terminations**:
  * `terminated`: Goal reached ($d \le 0.20$ m) or robot footprint collision.
  * `truncated`: Timeout only (`step_count >= 350`, with `info["TimeLimit.truncated"] = True`).
* **Dual Backend**:
  * `use_gazebo=False` (default for unit testing): Uses an exact internal 2D kinematic engine and 360-beam raycaster (>500 steps/s on CPU).
  * `use_gazebo=True`: Communicates with live ROS 2 and Gazebo Classic.

---

## 7. How to Run Tests and Benchmarks

Always activate the project virtual environment before running commands:
```bash
source ~/sim-real-rl/.venv/bin/activate
```

### Run the Full Unit Test Suite
Runs all 15 unit and compliance tests (Gymnasium `check_env`, SB3 `check_env`, observation math, action clipping, reward terms, termination, truncation):
```bash
python -m pytest ~/sim-real-rl/tests/ -v
```

### Run the 10,000-Step Soak Test
Validates that the environment can run 10,000 continuous random steps without crashes, NaNs, or memory leaks:
```bash
python ~/sim-real-rl/tests/soak_random_steps.py
```

### Run the Headless Simulation Benchmark (G1 Gate)
With Gazebo running via `headless_sim.launch.py`:
```bash
python ~/sim-real-rl/tests/benchmark_sim.py
```

---

## 8. What Siddhi Needs to Build Next (Day 3 & Beyond)

Per Section 10 of [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](Sim2Real_LiDAR_Navigation_Pipeline.pdf):

1. **Hand-Coded Reactive Controller** ([`baselines/reactive_controller.py`](../baselines/)):
   * Uses the same 43-dim observation vector.
   * Turn proportionally to goal bearing; scale forward speed by $\cos(\text{bearing})$.
   * Obstacle avoidance: if min range in $\pm 30^\circ$ front sector $< 0.5$ m, slow down and turn toward the side with larger mean range; if $< 0.25$ m, stop forward motion and rotate.
   * Target $\ge 90\%$ success on nominal scenarios.
2. **Scenario Generator** ([`scenarios/`](../scenarios/)):
   * Generate reproducible seed sequences for train, validation, test (50 scenarios), and stress sets.
3. **Pre-Training Gate G2 Checklist**:
   * Execute Checks 1–10 of Section 9 (1,000-reset soak test, timing verification, reward audit, controller comparison).
4. **ROS 2 Hardware Deployment Node** ([`deploy/`](../deploy/)):
   * Create deployment node importing [`rl/env/observation.py`](../rl/env/observation.py) directly for zero sim-to-real code divergence.

---

## 9. Gate G2 Status, Gazebo Benchmarks & Training Hold

**STATUS: Gate G2 Verification Incomplete / Pending: Continuous 100k-Step Gazebo Soak and Primary Backend Decision Pending Co-Owner Approval.**  
**P1 TRAINING IS STRICTLY ON HOLD. Do not launch training until the backend decision and reward review are formally approved by both co-owners.**

### Updated Gate G2 Backend Verification Matrix

| Check # | Description | Kinematic Backend Status | Gazebo Backend Status | Notes / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Check 1** | SB3 & Gymnasium `check_env` | **Passed** | **Passed** | Verified API compliance on both backends. |
| **Check 2** | Scenario Generator Determinism | **Passed** | **Passed** | Deterministic JSON layouts; corridor, 0.5m gap, dead-end stress scenarios added. Disjoint seeds verified. |
| **Check 3** | 1,000-Reset Soak | **Passed** (0.5s, 0 failures) | **Passed** (203.7s, 4.9 resets/s, 0 failures) | Full physics settle and 5 Hz scan arrival verified. Zero contact penetration impulses, position error $\le 2$ cm, yaw $\le 1^\circ$, velocity $\le 0.05$ m/s, zero overlap. |
| **Check 4** | Timing & Scan Monotonicity | **Passed** | **Passed** | 50 consecutive steps verified strictly monotonic scan timestamps ($t_i - t_{i-1} \in [0.10, 0.45]\,\text{s}$). Physical scan publication synchronized per step. |
| **Check 5** | Preprocessing Parity | **Passed** | **Passed** | Pure NumPy zero-ROS module identical across Gymnasium and hardware deployment (`deploy/`). |
| **Check 6** | Termination Semantics | **Passed** | **Passed** | Scripted wall collision (`is_collision=True, r <= -90`), goal reach (`is_success=True, r >= +90`), and timeout (`truncated=True`) verified in live Gazebo. |
| **Check 7** | Reward Audit | **Passed** | **Passed** | Traces in `results/reward_audit_traces.json` match frozen Section 5 math. Every episode return verified equal to sum of components ($< 10^{-12}$): Hand-Coded (+101.86 mean = +90.00 goal bonus + 17.73 progress - 5.87 step penalty), Spin (-17.50), Stationary (-17.50), Random (-107.65). Historical figures (+62.3, -34.1, -38.2, -39.8) are **strictly unverified** from repository artifacts. Siddhi's review pending. |
| **Check 8** | Reactive Controller ($\ge 90\%$) | **Passed (Kinematic)** | **Not run** | Evaluated on kinematic backend: 92.0% nominal (92/100, Wilson 95% CI: [85.0%, 95.9%]). Corner draws on real env interfaces reconciled: 25-ep Combined sign=+1 (22/25 = 88.0%), sign=-1 (21/25 = 84.0%); 50-ep runs on seeds 1,000,000..1,000,049 are a nested overlapping extension (first 25 identical bit-for-bit; second 25 are 24/25 = 96.0% and 21/25 = 84.0%), yielding 46/50 (92.0%) and 42/50 (84.0%), aggregate 88/100 (88.0%); separate runs on seeds 1,000..1,049 yield 40/50 (80.0%) and 40/50 (80.0%). All exceed 73.6% non-collapse threshold. Samples are nested/separate and must NOT be combined as 75 episodes. |
| **Check 9** | Soak & Memory | **Passed** (100k steps, +0.01% RSS) | **Passed** | 100,000 continuous live Gazebo steps completed across 4 isolated workers (`results/gazebo_soak_checkpoint.json`): 5,215.3 s elapsed (86.9 mins, 19.77 sps aggregate), 1,018 episodes, gzserver RSS 1,264.3 MB $\to$ 1,213.2 MB (-51.15 MB, -4.05% growth, passes $<5\%$ requirement), Python RSS 61.4 MB $\to$ 63.4 MB, 0 NaNs/Infs, 0 crashes. Baseline 1k checkpoint archived in `results/gazebo_soak_checkpoint_1k_baseline.json`. |
| **Check 10** | PPO Smoke & Checkpoint Parity | **Passed** (20k steps on kinematic) | **Passed (Watchdog & Checkpoint Verified)** | `tests/test_checkpoint_resume.py` verified exact SB3 parameter and action parity (`atol=1e-6`) across save/reload without training. 2-second monotonic clock missing-scan watchdog supervisor implemented (`rl/train/watchdog.py`) and verified via `tests/test_watchdog.py` (6/6 passed, including live Gazebo fault injection recovery from PID 289858 to PID 290061 with diagnostics saved to `results/watchdog_diagnostics.json`). |
| **Check 11** | Randomizers & Factor Isolation | **Passed (Kinematic)** | **Plumbed in step(), ODE physics unverified** | Verified on kinematic backend through real `reset()` and `step()` interfaces: P1 nominal, P2 sensor-only ($d_{\text{p1}} == d_{\text{p2}}$), P3 dynamics-only ($d_{\text{p1}} \ne d_{\text{p3}}$), P4 latency constraint ($t_{\text{lat}} + d_{\text{del}} \le 0.40$s) enforced. Live-Gazebo boundary: command gains, lag, delay, and acceleration limits are plumbed into `/cmd_vel` Twist; sensor perturbations are applied to `/scan` and `/odom`. However, SDF link physics (`mass_multiplier`, `friction_multiplier`) are **unsupported in Gazebo Classic at runtime** and are excluded from verified dynamics coverage. |

### Measured Gazebo Stepping Throughput & Projected Training Budget (Host: 16-core / 24-thread i7-14650HX, 15 GiB RAM)

| Setup | Measured Aggregate Stepping Rate | Measured Per-Worker Rate | Real-Time Factor (RTF) | Host CPU Utilization (Normalized) | Total `gzserver` RSS | Projected 1M-Step Run Duration | Projected 12 × 1M-Step Total Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Worker (Benchmark)** | 4.68 steps/s | 4.68 steps/s | 0.94× | ~2.5% (0.6 cores) | 317.3 MB | ~59.4 hours (2.48 days) | ~712.5 hours (29.7 days) |
| **2 Workers (Benchmark)** | 9.71 steps/s | 4.86 steps/s | 0.97× | ~6.7% (1.6 cores) | 628.8 MB | ~28.6 hours (1.19 days) | ~343.2 hours (14.3 days) |
| **4 Workers (100k Soak)** | **19.77 steps/s** | **4.94 steps/s** | **0.99×** | **~13.3% (3.2 cores)** | **1,213.2 MB** | **~14.05 hours (0.59 days)** | **~168.6 hours (7.03 days)** |
| *(Kinematic Dev)* | *657.2 steps/s* | *657.2 steps/s* | *131.4×* | *4.2% (1 core)* | *N/A* | *0.42 hours (25.4 mins)* | *5.07 hours* |

*Note*: 19.77 steps/s is the **empirically measured stepping rate during the 100k soak**. The 1M and 12M durations are **projections**, not active PPO training measurements. PPO updates and periodic evaluations will yield slightly lower throughput (~18.5–19.5 steps/s, ~7.0–7.5 days total).

### Separately Executed Test Suites (This Final Run)
1. **Full Unit & Integration Suite (Kinematic + Watchdog)**: `PYTHONPATH=. pytest tests/test_observation.py tests/test_env.py tests/test_randomizers.py tests/test_checkpoint_resume.py tests/test_watchdog.py tests/test_g2_gate.py -k "not test_100k_random_steps and not test_ppo_smoke_and_checkpoint"`
   - **46 passed, 1 skipped, 2 deselected in 75.29s**.
2. **Targeted Soak Runner Logic Suite**: `PYTHONPATH=. pytest tests/test_soak_runner_logic.py -v`
   - **5 passed, 0 failed in 0.14s** (atomic checkpointing, load corruption, resume math, delta accounting, linear elapsed time).
3. **Watchdog Live Gazebo Test Suite**: `pytest tests/test_watchdog.py -v -s`
   - **6 passed, 0 failed in 11.93s** (including controlled live Gazebo fault injection with PID change and clean recovery).
4. **Live Gazebo G2 Verification Suite**: `python3 tests/run_gazebo_g2.py`
   - **6 passed, 0 failed in 268.85s**.
5. **Live Gazebo 4-Worker 100k Soak**: `python3 tests/run_gazebo_soak.py --steps 100000 --workers 4 --resume`
   - **100,000 steps completed across 4 workers in 5,215.3s (86.9 mins)** (19.77 sps aggregate, 1,018 episodes, gzserver RSS -51.15 MB / -4.05%, Python RSS 61.4 $\to$ 63.4 MB, 0 NaNs, 0 crashes). Checkpoint saved to `results/gazebo_soak_checkpoint.json` with 2,000 memory samples.
6. **Live Gazebo Kill-and-Resume Soak Runner**: `python3 tests/run_gazebo_soak.py --test-kill-resume`
   - **1 passed (600 steps) in 325.7s**. State recorded in `results/gazebo_soak_checkpoint.json`.

### Required Actions Before P1 Launch
1. **Backend Approval**: Saigirish has approved **Option A (direct Gazebo Classic training for P1–P4)**. Kinematic backend is restricted strictly to development, reward debugging, and unit tests (zero weight transfer to Gazebo). Awaiting Siddhi's formal concurrence.
2. **Co-Owner Review**: Siddhi to review reward/termination traces (`results/reward_audit_traces.json`) and observe Gazebo playback episodes.
3. **Approve Gate G2**: Both co-owners must sign off before P1 training starts.


