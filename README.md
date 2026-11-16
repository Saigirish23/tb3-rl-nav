# Sim-to-Real 2D LiDAR Navigation Pipeline (`sim-real-rl`)

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Gazebo Classic 11](https://img.shields.io/badge/Gazebo-Classic%2011-orange.svg)](http://gazebosim.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source reinforcement learning pipeline evaluating the independent and combined effects of **sensor observation randomization** and **command-response dynamics randomization** for mapless 2D LiDAR navigation on the TurtleBot3 Burger differential-drive mobile robot.

Authoritative specification: [`docs/phase2_benchmark_specification.md`](docs/phase2_benchmark_specification.md).

---

## 1. Project Overview & Research Objective

### Problem Statement
Standard mobile robot navigation architectures rely heavily on Simultaneous Localization and Mapping (SLAM) and layered costmaps (e.g., Nav2). While effective in static, structured environments, they impose high onboard computational loads and can degrade in featureless, dynamically changing, or GPS-denied environments. Mapless navigation using deep reinforcement learning maps raw local 2D range observations directly to continuous motor velocity commands $(v, \omega)$. Transferring simulation-trained policies to physical robots encounters the **sim-to-real transfer gap** caused by sensing corruption (beam dropouts, range noise, odometry drift) and actuation mismatch (motor lag, communication delay, wheel slip).

### Research Objective
In a controlled $2 \times 2$ factorial experimental design, quantify the individual and interaction effects of:
1. **Sensor Observation Randomization** (distance-dependent noise, beam dropouts, contiguous block occlusions, sensor latency, odometry perturbations)
2. **Command-Response Dynamics Randomization** (velocity-level command gains, first-order actuator lag, control delay, acceleration limits)

in improving policy generalization across geometric, perceptual, and actuation stress without retraining.

---

## 2. System Architecture

```
                  +------------------------------------------------+
                  |            TurtleBot3 Burger (Sim)             |
                  |     - LDS-01 LiDAR (360 beams, 5 Hz)           |
                  |     - Wheel Encoders (Odometry @ 30 Hz)        |
                  +-----------------------+------------------------+
                                          |  /scan, /odom
                                          v
                  +------------------------------------------------+
                  |    Observation Preprocessor (Pure NumPy)       |
                  |    - 36 Min-Pooled LiDAR Bins (10 deg, [0, 1]) |
                  |    - 1 Goal Distance (Normalized by 5.657 m)   |
                  |    - 2 Goal Bearing Unit Vector [sin, cos]     |
                  |    - 2 Normalized Measured Velocities (v, w)   |
                  |    - 2 Previous Action Coordinates             |
                  +-----------------------+------------------------+
                                          |  State s_t (43-D)
                                          v
                  +------------------------------------------------+
                  |         PPO Actor Policy (NumPy / SB3)         |
                  |    MLP: 43 -> 128 (tanh) -> 128 (tanh) -> 2    |
                  |    Continuous action u_t in [-1.0, 1.0]^2      |
                  +-----------------------+------------------------+
                                          |  Action u_t
                                          v
                  +------------------------------------------------+
                  |          Action Mapping & Safety Layer         |
                  |    - cmd_v = 0.075 * (u[0] + 1) in [0, 0.15]   |
                  |    - cmd_omega = 0.60 * u[1] in [-0.60, 0.60]  |
                  |    - Stale scan watchdog: stop if scan > 2.0s  |
                  +-----------------------+------------------------+
                                          |  /cmd_vel
                                          v
                  +------------------------------------------------+
                  |          Robot Actuation (Gazebo / Sim)        |
                  +------------------------------------------------+
```

### 2.1 Observation Space ($D = 43$ Continuous Dimensions)
Direct inspection of `rl/env/observation.py` establishes the exact observation vector:
* **LiDAR Range Bins (`obs[0:36]`, 36 dims)**: 36 min-pooled range bins ($10^\circ$ angular span per bin), normalized by $R_{\text{max}} = 3.50\,\text{m}$ into $[0.0, 1.0]$. Bin 0 is centered directly forward ($[-5^\circ, +5^\circ)$).
* **Normalized Goal Distance (`obs[36]`, 1 dim)**: Euclidean distance to goal normalized by arena diagonal: $d_{\text{goal}} / D_{\text{diag}}$, where $D_{\text{diag}} = \sqrt{4.0^2 + 4.0^2} \approx 5.6569\,\text{m}$.
* **Goal Bearing Unit Vector (`obs[37:39]`, 2 dims)**: $[\sin\theta_{\text{rel}}, \cos\theta_{\text{rel}}]$, where $\theta_{\text{rel}} = \text{wrap\_to\_pi}(\theta_{\text{goal}} - \theta_{\text{robot}})$.
* **Measured Velocities (`obs[39:41]`, 2 dims)**: Forward linear velocity $v_{\text{meas}} / 0.15\,\text{m/s}$ (clipped to $[-1, 1]$) and angular velocity $\omega_{\text{meas}} / 0.60\,\text{rad/s}$ (clipped to $[-1, 1]$).
* **Previous Normalized Action (`obs[41:43]`, 2 dims)**: Commanded action $\mathbf{u}_{t-1} \in [-1.0, 1.0]^2$ from previous cycle (initialized to $[0.0, 0.0]$ at episode reset).

### 2.2 Action Mapping
Continuous policy output $\mathbf{u} \in [-1.0, 1.0]^2$ is mapped linearly to forward linear and angular velocities:
$$v = 0.075 \cdot (u_v + 1.0) \in [0.0, 0.15]\,\text{m/s}$$
$$\omega = 0.60 \cdot u_\omega \in [-0.60, +0.60]\,\text{rad/s}$$

### 2.3 Reward Function
Defined in `rl/env/tb3_navigation_env.py`:
$$r_t = 10.0 \cdot (d_t - d_{t+1}) - 0.05 + 100.0 \cdot \mathbb{I}[\text{goal}] - 100.0 \cdot \mathbb{I}[\text{collision}]$$
* **Goal Radius**: $r_{\text{goal}} = 0.20\,\text{m}$ (arrival tolerance during training).
* **Collision Penalty**: $-100.0$ on chassis penetration.
* **Step Penalty**: $-0.05$ per control step ($\Delta t = 0.20\,\text{s}, 5\,\text{Hz}$).

---

## 3. Four-Policy Factorial Design

The four Gen1 policies represent a complete $2 \times 2$ factorial ablation over domain randomization:

| Policy | Sensor Randomization | Command-Response Randomization | Checkpoint File | Checkpoint Timestep | Total Training Steps |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **$P_1$** | No | No | `deploy/p1_best_policy.npz` | **450,000** | **852,816** |
| **$P_2$** | Yes | No | `deploy/p2_best_policy.npz` | **300,000** | **704,512** |
| **$P_3$** | No | Yes | `deploy/p3_best_policy.npz` | **600,000** | **704,512** |
| **$P_4$** | Yes | Yes | `deploy/p4_best_policy.npz` | **400,000** | **704,512** |

### Critical Physics Grounding Note
* **Command-Response Randomization ($P_3, P_4$)**: Operates strictly at the software velocity-command level via first-order actuator lag ($\tau \in [0.0, 0.30]\,\text{s}$), discrete control delay ($d \in [0.0, 0.20]\,\text{s}$), command gains ($k_v, k_\omega \in [0.85, 1.15]$), and acceleration limits ($a_{\text{lin}} \in [0.20, 0.60]\,\text{m/s}^2$).
* **Physics Engine Status**: Sampled friction multipliers ($\mu \in [0.70, 1.30]$) and mass multipliers ($m \in [0.90, 1.20]$) are stored in Python dataclasses but **never applied to Gazebo physics**. Simulated robot mass ($1.0\,\text{kg}$) and Gazebo wheel surface friction ($\mu = 1.0$) remain at nominal defaults. $P_3$ and $P_4$ exhibit **command-response / actuation robustness**, not physical tribological adaptation.

### Sampled Sensor Randomization ($P_2, P_4$)
Sensor noise parameters are sampled independently per episode:
$$a \sim U(0.003, 0.015)\,\text{m}, \quad b \sim U(0.000, 0.010), \quad \sigma(r) = a + b \cdot r$$
$$r_{\text{noisy}} = \text{clip}(r + \mathcal{N}(0, \sigma^2(r)), 0.12, 3.50)$$
Accompanied by beam dropout ($p \in [0, 0.05]$), block occlusions ($p \in [0, 0.30]$), scan delivery latency ($0$ or $1$ scan), mounting offsets ($\pm 3^\circ, \pm 2\,\text{cm}$), and odometry noise.

---

## 4. PPO Architecture & Training Hyperparameters

All four policies share an identical neural architecture and training configuration (`configs/ppo.yaml`):

```yaml
algorithm: PPO
policy: MlpPolicy
policy_kwargs:
  net_arch:
    pi: [128, 128]  # Actor MLP: 2 layers of 128 units
    vf: [128, 128]  # Critic MLP: 2 layers of 128 units
  activation_fn: tanh
learning_rate: 0.0003
n_steps: 2048
batch_size: 256
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
```

---

## 5. Existing Preliminary Evaluation Results (50 Scenarios)

Prior preliminary evaluations tested $P_1$ and $P_4$ on 50 held-out test scenarios (seeds $2,000,000 \dots 2,000,049$):

| Policy | Success Rate | Collisions | Timeouts | Mean Path Length | Mean Steps | Wilson 95% CI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$ (Nominal)** | **47 / 50 (94.0%)** | 1 (2.0%) | 2 (4.0%) | $3.58\,\text{m}$ | 119.4 | $[83.5\%, 97.9\%]$ |
| **$P_4$ (Combined)** | **49 / 50 (98.0%)** | 1 (2.0%) | 0 (0.0%) | $3.62\,\text{m}$ | 120.8 | $[89.5\%, 99.6\%]$ |

### Explicit Scientific Limitations of Preliminary Results
1. **Statistical Significance**: A paired McNemar test on these 50 matched scenarios yields an exact two-sided $p$-value of **$p = 0.625$** ($p > 0.05$). The preliminary result does **not** establish statistically significant superiority of $P_4$ over $P_1$.
2. **Single Random Seed**: Evaluated across a single training run seed ($s=0$).
3. **Training Budget Asymmetry**: $P_1$ received $852,816$ steps vs. $P_4$'s $704,512$ steps ($\approx 17\%$ fewer training interactions).
4. **Scope**: These preliminary trials evaluated static random obstacles; they do not represent the finalized nine-condition Phase 2 stress benchmark.

---

## 6. Phase 2 Stress-Testing Benchmark

The Phase 2 benchmark subjects all four policies to nine challenging stress conditions across three orthogonal axes:

```
                                  PHASE 2 BENCHMARK CONDITIONS
                                                |
          +-------------------------------------+-------------------------------------+
          |                                     |                                     |
  Geometric Stress                      Perceptual Stress                     Actuation Stress
          |                                     |                                     |
  Case 1: Narrow Corridor (0.45m)        Case 7A: Distance Noise (OOD)        Case 8: OOD Actuation Lag
  Case 2: Concave U-Trap Escape          Case 7B: Forward 30° Blind Sector            (tau=0.30s, delay=0.20s,
  Case 3: S-Curve Chicane Baffles                                                      gains 0.80 / 1.20)
  Case 4: Blind 90° L-Junction
  Case 5: Dynamic Crossing Obstacle
  Case 6: Stepped Funnel (0.35m throat)
```

### 6.1 Planned Benchmark Allocations (Status: Planned / Pending Execution)
* **Planned Pilot Run**: $9\,\text{conditions} \times 4\,\text{policies} \times 1\,\text{seed} = \mathbf{36\,\text{episodes}}$ (technical verification of resets, spawns, and logging).
* **Planned Main Benchmark**: $9\,\text{conditions} \times 4\,\text{policies} \times 20\,\text{matched seeds} = \mathbf{720\,\text{episodes}}$.

### 6.2 Conditions Requiring Empirical Pilot Validation
1. **Case 5 Initial Tangency**: The moving cylinder starts at $(-0.80, 0.00)\,\text{m}$ ($R = 0.15\,\text{m}$). Rack 1 East face is at $x = -0.95\,\text{m}$. Initial clearance is exactly **0 cm (tangency)**. Pilot execution must verify that contact sensors do not register false collision impulses at spawn before describing the trajectory as collision-free.
2. **Case 2 Choke Lip Clearance**: The provisional grid-planner A\* route ($3.718\,\text{m}$) passes the U-trap lip at $3.1\,\text{cm}$ clearance. Pilot execution must confirm whether wheel wobble or caster bounce triggers contact collisions.
3. **Cases 3 & 4 Swept-Footprint Tracking**: S-curve and L-junction paths are analytically designed reference geometries whose continuous tracking must be validated in Gazebo.

### 6.3 Evaluation Metrics & Selection Bias Statement
* **Full-Cohort Metrics**: Success Rate ($SR$), Collision Rate ($CR$), Timeout Rate ($TO$), and **Penalized Completion Time** ($T_{\text{penalized}} = T_{\text{goal}}$ if success, else $100.0\,\text{s}$).
* **Conditional Trajectory Metrics**: Path length ($L$), completion time ($T_{\text{goal}}$), and action smoothness ($\text{TV}_{\text{norm}}(\omega)$) are **conditional on successful episodes only**. They characterize kinematic efficiency among successful episodes, but do not represent failed trials.
* **Goal Acceptance Threshold**: $d_{\text{goal}} \le 0.25\,\text{m}$ (accommodates TurtleBot3 physical diameter of $0.226\,\text{m}$ in tight fixtures).
* **Dual Collision Source of Truth**: Geometric footprint clearance ($\le 0.0\,\text{m}$) and Gazebo sensor contact ($\le 0.120\,\text{m}$) are both logged; an episode is classified as collision if either triggers.

---

## 7. Installation, Build & Execution

### 7.1 Prerequisites
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill (`/opt/ros/humble`)
- Gazebo Classic 11.10.2
- Python 3.10 with PyTorch and Stable-Baselines3 2.7.1

### 7.2 Build ROS 2 Workspace
```bash
source /opt/ros/humble/setup.bash
cd ~/sim-real-rl/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 7.3 Run Tests
```bash
cd ~/sim-real-rl
# Run all unit and benchmark evaluation tests
pytest -v tests/test_benchmark_evaluation.py tests/test_inference_correctness.py tests/test_observation.py
```

### 7.4 Launch Unified Evaluation Suite

#### Method 1: ROS 2 Launch (Gazebo Integration)
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash

# Run Pilot Mode (Default: 36 episodes)
ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=pilot

# Run Main Benchmark Mode (720 episodes across 20 matched seeds)
ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=benchmark

# Evaluate Specific Policy and Condition
ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=pilot policy:=P4 condition:=case1_corridor

# Fast Regression Test (Simulation-free analytical environment)
ros2 launch tb3_rl_sim evaluate_policies.launch.py use_gazebo:=false mode:=pilot
```

#### Method 2: Standalone Evaluation Runner CLI
```bash
# Fast mock evaluation across all 4 policies on Case 1 and Case 5
python3 scripts/evaluate_benchmark.py --mode pilot --policy all --condition case1_corridor,case5_dynamic --use-gazebo false

# Full 36-episode pilot in simulation-free mode
python3 scripts/evaluate_benchmark.py --mode pilot --use-gazebo false --out-dir runs/pilot_results
```

### 7.5 Output Locations
All evaluation runs generate two machine-readable artifacts:
- **Raw Episode Telemetry**: `<output_dir>/raw_episodes.csv`
- **Benchmark Aggregate Summary**: `<output_dir>/benchmark_summary.json`

---

## 8. Reproducibility & Limitations

1. **Local Work Only**: All training, inference, and evaluation artifacts are stored locally.
2. **Model Preservation**: All four trained Gen1 policies (`deploy/p*_best_policy.npz`) and checkpoints (`runs/main_p*/`) remain unmodified.
3. **What Has Been Validated**:
   - Python-side environment reset, step, and raycasting.
   - 43-D observation and `[128, 128]` MLP inference.
   - All 9 benchmark scenario setups and matched-seed determinism.
   - Launch file syntax and argument parsing.
   - Output CSV and JSON serialization schemas.
4. **What Remains Pending**:
   - Physical execution of the 36-episode pilot in Gazebo Classic.
   - Physical execution of the 720-episode main benchmark in Gazebo Classic.
   - Experimental validation of Case 5 moving-obstacle contact clearance and Case 2 choke lip margin.

---

## 9. Contributors

- **Saigirish S** (`B24ME1066`, Department of Mechanical Engineering, IIT Jodhpur)
- **Siddhi Vinayak** (`B24BB1036`, Department of Biosciences and Bioengineering, IIT Jodhpur)
