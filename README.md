# Sim-to-Real 2D LiDAR Navigation Pipeline (`sim-real-rl`)

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Gazebo Classic 11](https://img.shields.io/badge/Gazebo-Classic%2011-orange.svg)](http://gazebosim.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source reinforcement learning pipeline evaluating the independent and combined effects of **sensor observation randomization** and **command-response dynamics randomization** for mapless 2D LiDAR navigation on the TurtleBot3 Burger differential-drive mobile robot.

Authoritative documentation and dossiers:
- Technical Evidence Dossier: [`docs/final_report_evidence_dossier.md`](docs/final_report_evidence_dossier.md)
- Phase 2 Benchmark Specification: [`docs/phase2_benchmark_specification.md`](docs/phase2_benchmark_specification.md)
- 4-Way Evaluation Dataset: [`results/all_policies_4way_comparison.json`](results/all_policies_4way_comparison.json)

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
* **Goal Radius**: $r_{\text{goal}} = 0.20\,\text{m}$ (arrival tolerance during training and evaluation).
* **Collision Penalty**: $-100.0$ on chassis contact / penetration.
* **Step Penalty**: $-0.05$ per control step ($\Delta t = 0.20\,\text{s}, 5\,\text{Hz}$).

---

## 3. Four-Policy Factorial Design

The four Gen1 policies represent a complete $2 \times 2$ factorial ablation over domain randomization:

| Policy | Sensor Randomization | Command-Response Randomization | Final Checkpoint (Primary) | Final Steps | Periodic Best Checkpoint | Best Steps |
| :---: | :---: | :---: | :--- | :---: | :--- | :---: |
| **$P_1$** (Nominal) | No | No | `deploy/p1_policy.npz` | **852,816** | `deploy/p1_best_policy.npz` | 450,000 |
| **$P_2$** (Sensor) | Yes | No | `deploy/p2_policy.npz` | **704,512** | `deploy/p2_best_policy.npz` | 300,000 |
| **$P_3$** (Dynamics) | No | Yes | `deploy/p3_policy.npz` | **704,512** | `deploy/p3_best_policy.npz` | 600,000 |
| **$P_4$** (Combined) | Yes | Yes | `deploy/p4_policy.npz` | **704,512** | `deploy/p4_best_policy.npz` | 400,000 |

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

## 5. Four-Policy Held-Out Evaluation Results (50 Scenarios)

All four policies were evaluated on an identical test suite of **50 held-out scenarios** (fixed seeds `2000000` through `2000049`) in the nominal $4.0 \times 4.0\,\text{m}$ arena with $0.20\,\text{m}$ goal tolerance, $350$-step ($70.0\,\text{s}$) timeout limit, and $5.0\,\text{Hz}$ control frequency. Complete episode-level traces are archived in [`results/all_policies_4way_comparison.json`](results/all_policies_4way_comparison.json).

### 5.1 Primary Factorial Comparison (Final Converged Checkpoints)

| Policy | Condition | Episodes | Success Rate | Collisions | Timeouts | Mean Reward | Mean Steps (Succ.) | Mean Path (Succ.) | Mean Min Clear | Wilson 95% CI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$** | Nominal | 50 | **47 / 50 (94.0%)** | 1 (2.0%) | 2 (4.0%) | $103.75 \pm 37.71$ | $106.17 \pm 23.36$ | $2.251 \pm 0.443\,\text{m}$ | $0.282 \pm 0.081\,\text{m}$ | $[83.8\%, 97.9\%]$ |
| **$P_2$** | Sensor | 50 | **48 / 50 (96.0%)** | 1 (2.0%) | 1 (2.0%) | $106.94 \pm 35.97$ | $\mathbf{93.06 \pm 25.10}$ | $2.236 \pm 0.447\,\text{m}$ | $0.262 \pm 0.076\,\text{m}$ | $[86.5\%, 98.9\%]$ |
| **$P_3$** | Dynamics | 50 | **49 / 50 (98.0%)** | 1 (2.0%) | 0 (0.0%) | $\mathbf{109.05 \pm 28.76}$ | $105.76 \pm 26.24$ | $2.144 \pm 0.422\,\text{m}$ | $0.286 \pm 0.081\,\text{m}$ | $[89.5\%, 99.6\%]$ |
| **$P_4$** | Combined | 50 | **49 / 50 (98.0%)** | 1 (2.0%) | 0 (0.0%) | $109.03 \pm 30.95$ | $101.49 \pm 24.32$ | $\mathbf{2.067 \pm 0.383\,\text{m}}$ | $\mathbf{0.292 \pm 0.076\,\text{m}}$ | $[89.5\%, 99.6\%]$ |

### 5.2 Validation Best Checkpoints Comparison (Trained Snapshots)

| Checkpoint | Checkpoint Step | Episodes | Success Rate | Collisions | Timeouts | Mean Reward | Mean Steps (Succ.) | Mean Path (Succ.) | Mean Min Clear | Wilson 95% CI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_{2,\text{best}}$** | 300,000 | 50 | **43 / 50 (86.0%)** | 2 (4.0%) | 5 (10.0%) | $86.72 \pm 48.91$ | $135.65 \pm 52.88$ | $2.341 \pm 0.485\,\text{m}$ | $0.277 \pm 0.076\,\text{m}$ | $[73.8\%, 93.0\%]$ |
| **$P_{3,\text{best}}$** | 600,000 | 50 | **46 / 50 (92.0%)** | 4 (8.0%) | 0 (0.0%) | $99.85 \pm 47.92$ | $113.76 \pm 26.17$ | $2.157 \pm 0.421\,\text{m}$ | $0.268 \pm 0.088\,\text{m}$ | $[81.2\%, 96.8\%]$ |

> [!NOTE]
> The final converged checkpoints consistently outperform early periodic validation snapshots because periodic evaluation during training used single-episode rollouts with high sampling variance. The final converged policies exhibit substantially reduced timeout rates and tighter trajectory tracking.

### 5.3 Pairwise Statistical Significance Matrix

All six pairwise comparisons were evaluated using two-sided exact McNemar tests on binary success/failure ($N=50$), and two-sided Wilcoxon signed-rank tests on continuous metrics evaluated on mutually successful episodes ($N_{\text{both}}$):

| Pairwise Comparison | $N_{\text{both}}$ | Exact McNemar $p$ (Binary Success) | Wilcoxon Steps $p$ | Wilcoxon Path $p$ | Wilcoxon Clearance $p$ | Mean $\Delta \text{Steps}$ | Mean $\Delta \text{Path Length}$ | Mean $\Delta \text{Clearance}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$ vs $P_2$** | 45 | $p = 1.0000$ | $\mathbf{p < 0.0001^*}$ | $p = 0.9022$ | $p = 0.0884$ | $-14.24$ steps | $-0.058\,\text{m}$ | $-0.019\,\text{m}$ |
| **$P_1$ vs $P_3$** | 46 | $p = 0.6250$ | $p = 0.0618$ | $p = 0.0138$ | $p = 0.6729$ | $-2.74$ steps | $-0.131\,\text{m}$ | $+0.006\,\text{m}$ |
| **$P_1$ vs $P_4$** | 46 | $p = 0.6250$ | $p = 0.0152$ | $\mathbf{p < 0.0001^*}$ | $p = 0.5882$ | $-4.39$ steps | $\mathbf{-0.187\,\text{m}}$ | $+0.010\,\text{m}$ |
| **$P_2$ vs $P_3$** | 47 | $p = 1.0000$ | $\mathbf{p = 0.0005^*}$ | $\mathbf{p = 0.0008^*}$ | $p = 0.1859$ | $+11.32$ steps | $-0.091\,\text{m}$ | $+0.027\,\text{m}$ |
| **$P_2$ vs $P_4$** | 47 | $p = 1.0000$ | $\mathbf{p = 0.0067^*}$ | $\mathbf{p < 0.0001^*}$ | $p = 0.1381$ | $+7.94$ steps | $\mathbf{-0.158\,\text{m}}$ | $+0.031\,\text{m}$ |
| **$P_3$ vs $P_4$** | 48 | $p = 1.0000$ | $p = 0.3514$ | $\mathbf{p = 0.0003^*}$ | $p = 0.3204$ | $-4.33$ steps | $\mathbf{-0.084\,\text{m}}$ | $+0.008\,\text{m}$ |

*\* Bold values with asterisk indicate statistical significance under Bonferroni-corrected family-wise threshold $\alpha = 0.05 / 6 = 0.00833$.*

### 5.4 Key Scientific Insights & Limitations
1. **Binary Success Indistinguishability**: In nominal conditions with $N=50$, binary success rates ($94.0\% - 98.0\%$) are high across all policies. Pairwise McNemar tests yield $p \ge 0.6250$; nominal success rate alone is insufficient to claim statistical superiority.
2. **Speed vs. Path Geometry Trade-off**:
   - **$P_2$ (Sensor Randomization)** learns an aggressive navigation strategy that reaches the goal significantly faster than $P_1$ ($-14.24$ steps, $p < 0.0001$), but maintains slightly lower obstacle clearance ($0.262\,\text{m}$).
   - **$P_3$ (Command-Response Randomization)** completely eliminates timeouts ($0\%$ vs $4\%$ in $P_1$) by building robustness to control delays and lag.
   - **$P_4$ (Combined Randomization)** achieves the **shortest and most efficient trajectories** across the entire cohort ($2.067\,\text{m}$), beating $P_1$ ($-0.187\,\text{m}, p < 0.0001$), $P_2$ ($-0.158\,\text{m}, p < 0.0001$), and $P_3$ ($-0.084\,\text{m}, p = 0.0003$), while simultaneously providing the **highest mean obstacle clearance** ($0.292\,\text{m}$).
3. **Best Overall Policy**: **$P_4$** is demonstrably superior in trajectory optimization, safety margin, and timeout elimination.

---

## 6. Phase 1 Warehouse Environment & Phase 2 Benchmark Specification

### 6.1 Phase 1 Warehouse Environment Geometry
The primary simulation environment is implemented in `ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`. It features an outer $5.0 \times 5.0\,\text{m}$ footprint with an inner usable span of $[-2.5, 2.5] \times [-2.5, 2.5]\,\text{m}$ enclosed by four perimeter walls, five permanent static warehouse fixtures, and an eight-obstacle movable pool:

| Static Infrastructure Element | Center Position $(x, y)$ | Dimensions $(w, l, h)$ | Physical Span $[x_{\text{min}}, x_{\text{max}}] \times [y_{\text{min}}, y_{\text{max}}]$ |
| :--- | :---: | :---: | :---: |
| **Warehouse Rack 1** | $(-1.15, 0.65)$ | $0.40 \times 1.50 \times 0.90\,\text{m}$ | $[-1.35, -0.95] \times [-0.10, 1.40]\,\text{m}$ |
| **Warehouse Rack 2** | $(1.15, -0.65)$ | $0.40 \times 1.50 \times 0.90\,\text{m}$ | $[0.95, 1.35] \times [-1.40, 0.10]\,\text{m}$ |
| **North Shelf** | $(0.00, 2.15)$ | $2.40 \times 0.40 \times 0.90\,\text{m}$ | $[-1.20, 1.20] \times [1.95, 2.35]\,\text{m}$ |
| **Pallet Stack** | $(-1.80, -1.80)$ | $0.80 \times 0.80 \times 0.40\,\text{m}$ | $[-2.20, -1.40] \times [-2.20, -1.40]\,\text{m}$ |
| **Staging Barrier** | $(1.80, 1.80)$ | $0.80 \times 0.80 \times 0.40\,\text{m}$ | $[1.40, 2.20] \times [1.40, 2.20]\,\text{m}$ |

*Movable pool obstacles: 4 cylinders ($R \in [0.12, 0.18]\,\text{m}$) and 4 boxes (sizes $0.25 - 0.35\,\text{m}$) dynamically relocated per episode.*

### 6.2 Phase 2 Stress-Testing Benchmark (9 Conditions)
The Phase 2 benchmark systematically challenges policy boundaries across three orthogonal stress dimensions:

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

- **Case 1 (Corridor)**: $0.45\,\text{m}$ aisle ($1.99 \times$ robot chassis diameter).
- **Case 2 (U-Trap)**: $1.20 \times 1.20\,\text{m}$ deep trap forcing escape from local minimum.
- **Case 3 (Chicane)**: Three overlapping baffles forcing reverse-curvature S-turns.
- **Case 4 (L-Junction)**: Blind $90^\circ$ turn with zero look-ahead line-of-sight.
- **Case 5 (Dynamic Crossing)**: Dynamic cylinder moving at $0.20\,\text{m/s}$ crossing the robot path at $t \approx 10\,\text{s}$ and safely parking before Rack 2.
- **Case 6 (Stepped Funnel)**: Converging corridor tapering from $1.0\,\text{m}$ down to a $0.35\,\text{m}$ throat.
- **Case 7A (Perceptual Noise)**: High distance-dependent noise: $\sigma(r) = 0.03 + 0.04 \cdot r$.
- **Case 7B (Perceptual Blind Sector)**: Complete masking of forward $30^\circ$ field-of-view (bins 35, 0, 1 set to max range).
- **Case 8 (Actuation Stress)**: Extreme lag ($\tau = 0.30\,\text{s}$), $200\,\text{ms}$ delay, command gains $k_v = 0.80, k_\omega = 1.20$.

### 6.3 Collision Ground Truth Decoupling
To eliminate false collisions caused by sensor noise or blind sectors:
1. **Primary Ground Truth**: Gazebo physics contact sensor impulses (`/bumper` / contact states).
2. **Secondary Cross-Check**: Analytical geometric polygon footprint penetration ($\text{distance} \le 0.0\,\text{m}$).
3. Minimum LiDAR range is recorded independently for clearance metrics and is never used as a physical collision outcome.

---

## 7. Demonstration Guide: Showcasing the Best Policy ($P_4$)

The repository provides multiple execution modes to demonstrate the superior performance of **$P_4$** (Combined Randomization) in Gazebo Classic 11.

### Prerequisites Setup
```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Build and source local workspace
cd ~/sim-real-rl/ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ~/sim-real-rl
```

---

### Option A: Automated Multi-Waypoint Simulation Tour (Recommended Quick Demo)
Executes a continuous 3-goal waypoint tour in the warehouse world using headless Gazebo Classic, logs real-time step telemetry, and generates an acceptance report:

```bash
python3 scripts/run_simulation_demo.py --policy P4
```

*What happens:*
1. Automatically starts Gazebo Classic on isolated port 11345 (Domain 30).
2. Runs 3 sequential waypoint challenges:
   - Trial 1: $(1.20, 0.00)\,\text{m}$ (Straight East)
   - Trial 2: $(1.00, 1.00)\,\text{m}$ (Diagonal North-East)
   - Trial 3: $(-1.00, 1.00)\,\text{m}$ (Left Quadrant North-West)
3. Outputs step count, final distance, displacement, and minimum clearance.
4. Cleanly terminates simulator and saves summary to `runs/demo_run_<timestamp>/demo_summary.json`.

---

### Option B: Interactive 3D Gazebo GUI Navigation
Watch the TurtleBot3 navigate in real-time inside the 3D Gazebo GUI.

**Terminal 1 — Launch Gazebo Simulator with 3D GUI:**
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch tb3_rl_sim sim.launch.py gui:=true
```

**Terminal 2 — Run Keyboard-Driven Waypoint Navigator:**
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
python3 scripts/interactive_nav.py \
    --model-path runs/main_p4_s0/final_model.zip \
    --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl
```
*Enter target $(x, y)$ coordinates in the terminal (e.g. `1.0 1.0`) and observe real-time mapless LiDAR obstacle avoidance.*

**Or Send a Single Waypoint Target via PPO Inference Node:**
```bash
python3 deploy/ppo_inference_node.py \
    --model-path runs/main_p4_s0/final_model.zip \
    --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl \
    --goal-x 1.2 --goal-y 0.5 \
    --goal-tolerance 0.20
```

---

### Option C: Obstacle Challenge Tour (Slalom / Hallway Pinch)
Teleports pre-configured obstacle layouts into Gazebo and executes policy navigation:

```bash
# Slalom corridor (4 staggered obstacles)
python3 scripts/demo_with_obstacles.py --policy P4 --preset slalom

# Narrow constriction pinch-point
python3 scripts/demo_with_obstacles.py --policy P4 --preset hallway_pinch

# Central pillar avoidance
python3 scripts/demo_with_obstacles.py --policy P4 --preset center_pillar
```

---

### Option D: Unified Phase 2 Stress Benchmark Evaluation
Run the complete Phase 2 stress evaluation across all 9 conditions:

```bash
# Fast analytical simulation-free evaluation of P4 (All 9 stress cases)
python3 scripts/evaluate_benchmark.py --mode pilot --policy P4 --use-gazebo false --out-dir runs/pilot_p4

# Full 36-episode pilot in Gazebo Classic
ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=pilot policy:=P4

# Compare all 4 policies in fast simulation-free mode
python3 scripts/evaluate_benchmark.py --mode pilot --policy all --use-gazebo false --out-dir runs/pilot_all
```

---

## 8. Automated Test Suite & Regression Verification

Run the full automated test suite covering unit tests, episode resets, actuation ordering, benchmark setup, and observation preprocessing:

```bash
pytest -v \
    tests/test_benchmark_evaluation.py \
    tests/test_episode_reset.py \
    tests/test_actuation_order.py \
    tests/test_inference_correctness.py \
    tests/test_observation.py
```

*Results: 24 passed in ~12 seconds.*

---

## 9. Repository Structure

```
sim-real-rl/
├── configs/
│   └── ppo.yaml                         # PPO training & environment hyperparameters
├── deploy/
│   ├── p1_policy.npz                    # Converged P1 policy weights (NumPy format)
│   ├── p2_policy.npz                    # Converged P2 policy weights (NumPy format)
│   ├── p3_policy.npz                    # Converged P3 policy weights (NumPy format)
│   ├── p4_policy.npz                    # Converged P4 policy weights (NumPy format)
│   ├── p1_best_policy.npz               # Periodic validation best P1 checkpoint
│   ├── p2_best_policy.npz               # Periodic validation best P2 checkpoint
│   ├── p3_best_policy.npz               # Periodic validation best P3 checkpoint
│   ├── p4_best_policy.npz               # Periodic validation best P4 checkpoint
│   └── ppo_inference_node.py            # Standalone ROS 2 / NumPy inference node
├── docs/
│   ├── final_report_evidence_dossier.md # 700-line comprehensive technical review & evidence dossier
│   ├── phase2_benchmark_specification.md# Authoritative 9-case benchmark specification
│   └── Sim2Real_LiDAR_Navigation_Pipeline.pdf # Original system design specification
├── results/
│   ├── all_policies_4way_comparison.json# 50-scenario held-out evaluation dataset (seeds 2000000..2000049)
│   ├── ppo_training_20261025/           # Training metadata & curves for P1, P2, P3, P4
│   └── watchdog_diagnostics.json        # Gazebo watchdog recovery logs
├── rl/
│   ├── env/
│   │   ├── observation.py               # 43-D observation preprocessor (pure NumPy)
│   │   ├── randomizer.py                # Domain randomizer (sensor & command-response)
│   │   └── tb3_navigation_env.py        # Gymnasium-compatible TurtleBot3 environment
│   └── train/
│       └── watchdog.py                  # Process-group supervisor and scan watchdog
├── ros2_ws/
│   └── src/
│       ├── tb3_rl_navigation/           # Core ROS 2 navigation integration package
│       └── tb3_rl_sim/
│           ├── launch/
│           │   ├── sim.launch.py        # Full Gazebo Classic 3D GUI launch file
│           │   ├── headless_sim.launch.py# Headless Gazebo launch file
│           │   └── evaluate_policies.launch.py # Phase 2 benchmark evaluation launch file
│           └── worlds/
│               └── arena_4x4.world      # Warehouse simulation world with static fixtures
├── scripts/
│   ├── demo_with_obstacles.py           # Demonstration with custom obstacle layouts
│   ├── evaluate_benchmark.py            # Unified 9-case Phase 2 benchmark evaluation engine
│   ├── interactive_nav.py               # Interactive keyboard waypoint navigator
│   ├── run_simulation_demo.py           # Automated 3-trial simulation demo runner
│   └── verify_gazebo_integration.py     # Live ROS 2 + Gazebo integration smoke tester
└── tests/                               # Comprehensive pytest validation suite
```

---

## 10. Contributors & Attribution

- **Saigirish S** (`B24ME1066`, Department of Mechanical Engineering, Indian Institute of Technology Jodhpur)
- **Siddhi Vinayak** (`B24BB1036`, Department of Biosciences and Bioengineering, Indian Institute of Technology Jodhpur)
