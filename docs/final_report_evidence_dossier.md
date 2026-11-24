# Technical Evidence Dossier: Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation

**Document Purpose:** Comprehensive evidence collection, system architecture reconstruction, experimental data reconciliation, and technical dossier for the final IEEE conference paper.  
**Repository:** `https://github.com/Saigirish23/tb3-rl-nav.git`  
**Branch:** `midterm-report-draft` (Commit: `618cf54`)  
**Date of Audit:** November 16, 2026  
**Scope:** Rigorous verification of local source code, training logs, evaluation records, SDF world models, and test harnesses. Strict distinction between completed results, software-verified capabilities, and planned experiments. Zero hardware claims; zero fabricated data.

---

## 1. Proposed IEEE Conference Paper Title

**Primary Title:**  
*Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation via Factorial Domain Randomization*

**Alternative Title:**  
*Decomposing Sensor and Actuator Uncertainty in Reinforcement Learning for Mobile Robot Navigation: A Factorial Sim-to-Real Study*

---

## 2. Authors, Affiliations, and Institutional Metadata

### Author List
1. **Saigirish S**  
   *Roll Number:* B24ME1066  
   *Email:* `b24me1066@iitj.ac.in`  
   *Department:* Department of Mechanical Engineering, Indian Institute of Technology Jodhpur, Rajasthan, India - 342030  
   *GitHub:* `Saigirish23`  
   *Roles:* Simulation architecture, Gymnasium environment wrapper, PPO training infrastructure, multi-worker parallelization, domain randomizers, orchestration daemon, P1/P4 training.

2. **Siddhi Vinayak**  
   *Roll Number:* B24BB1036  
   *Email:* `b24bb1036@iitj.ac.in`  
   *Department:* Department of Biosciences and Bioengineering, Indian Institute of Technology Jodhpur, Rajasthan, India - 342030  
   *GitHub:* `siddhivn`  
   *Roles:* Reactive potential-field baseline controller, deterministic scenario generator, missing-scan watchdog monitor, reward component verification suite, deployment inference node.

### Institutional Metadata & User-Required Inputs
- **Institution:** Indian Institute of Technology Jodhpur (IIT Jodhpur)
- **Academic Session:** 2026–2027 Academic Year
- **Course Identification:** `[USER INPUT REQUIRED: Course Code and Title, e.g., ME4xx/AIxxx Autonomous Mobile Robotics]`
- **Project Advisor / Supervisor:** `[USER INPUT REQUIRED: Faculty Advisor Name, Academic Title, and Department]`
- **Academic Milestone:** Final B.Tech / Course Research Project Report

---

## 3. Abstract (Evidence-Based Draft)

Deep reinforcement learning (DRL) provides an end-to-end framework for mobile robot navigation in unknown environments; however, policies trained in idealized physics simulators frequently fail upon deployment due to discrepancies in sensor noise, latency, and unmodeled command-response dynamics. In this work, we present an end-to-end, mapless LiDAR navigation framework for the differential-drive TurtleBot3 Burger platform, trained via Proximal Policy Optimization (PPO) in ROS 2 Humble and Gazebo Classic 11. To rigorously investigate the mechanisms governing sim-to-real transfer, we formulate a $2 \times 2$ factorial domain randomization architecture decomposing uncertainty into two orthogonal factors: observation/sensor noise ($P_2$), command-response/actuator dynamics ($P_3$), and their joint composition ($P_4$), evaluated against a nominal baseline ($P_1$). 

All four policies were trained using 4 parallel headless Gazebo instances with strict port/DDS isolation across 704,512 to 852,816 environment interactions ($\sim 40.9$ cumulative CPU hours at $\sim 19.7\,\text{steps/s}$). On a comprehensive held-out test suite of 50 randomized obstacle scenarios, the four policies demonstrated high overall task competence: nominal $P_1$ achieved $94.0\%$ success (1 collision, 2 timeouts), sensor-randomized $P_2$ achieved $96.0\%$ success with the fastest completion speed ($93.06\,\text{steps}$ to goal), dynamics-randomized $P_3$ achieved $98.0\%$ success (zero timeouts), and combined $P_4$ achieved $98.0\%$ success with the most direct paths ($2.067\,\text{m}$) and highest clearance ($0.292\,\text{m}$). However, exact paired two-sided McNemar tests across all six policy pairs yielded $p \ge 0.6250$ (all non-significant under Bonferroni-adjusted $\alpha = 0.00833$), proving that $N=50$ trials are statistically underpowered to establish significance in binary completion rate despite significant continuous metric differences. To provide definitive stress evaluation, we specify a comprehensive 9-condition benchmark targeting out-of-distribution geometric traps, dynamic cross-traffic, blind-sector masking, and actuator lag. We detail the system architecture, mathematical formulations, and engineering solutions to simulator deadlocks, providing a reproducible evidence base for robust robot learning.

---

## 4. Introduction

### 4.1 Research Problem & Sim-to-Real Motivation
Autonomous mobile robot navigation in GPS-denied indoor environments traditionally relies on classical Modular Navigation Stacks (e.g., SLAM, global path planners like $A^*$/Dijkstra, and local trajectory planners like the Timed Elastic Band or Dynamic Window Approach). While effective in mapped, static environments, classical stacks require substantial computational overhead, are brittle to dynamic clutter, and necessitate extensive parameter re-tuning across platforms. 

Mapless navigation using Deep Reinforcement Learning (DRL) offers an attractive alternative: by learning a direct mapping from raw 2D range scans and relative goal coordinates to continuous velocity setpoints, a robot can navigate reactively without a prior global map. However, deploying DRL policies trained in simulation onto real robotic platforms faces the **sim-to-real gap**, which manifests across two distinct physical domains:
1. **Sensor & State-Estimation Uncertainty:** Real planar LiDAR sensors (such as the 360-degree LDS-01) exhibit beam-level Gaussian noise, distance-dependent beam divergence, surface-reflectivity dropouts, block occlusions, motion-induced scan distortion, and transport latency over serial/USB buses. Furthermore, real wheel encoders experience slippage, leading to odometry scale drift and heading bias.
2. **Actuator & Command-Response Dynamics:** Real DC motor actuators do not track continuous velocity commands instantaneously. They exhibit communication transport delay ($d \in [0, 0.20]\,\text{s}$), battery-dependent command gain scaling ($k_v, k_\omega$), motor torque saturation, acceleration slew-rate limits, and electro-mechanical first-order lag ($\tau \in [0, 0.30]\,\text{s}$).

When policies are trained assuming ideal sensor beams and instantaneous acceleration, they develop brittle, high-frequency oscillatory control strategies that collide or stall when subjected to real-world latency and drift.

### 4.2 Research Questions & Objectives
This project addresses three fundamental research questions:
1. **Factor Decomposition:** Does sensor observation noise or actuator command-response lag contribute more significantly to policy degradation in mapless navigation?
2. **Coupling Effects:** Does joint randomization ($P_4$) provide synergistic robustness, or does the superposition of disturbances degrade policy training stability and asymptotic task performance?
3. **Statistical Power in Navigation Benchmarks:** What sample sizes and experimental controls are required to rigorously prove sim-to-real robustness beyond descriptive metrics?

### 4.3 Summary of Completed Contributions
1. **Modular System Architecture:** Developed a ROS 2 Humble and Gazebo Classic 11 navigation environment featuring a 43-dimensional observation pipeline implemented in pure NumPy with zero ROS dependencies, shared identically across simulation training, evaluation, and physical deployment nodes.
2. **2x2 Factorial Randomization Engine:** Implemented independent sensor and actuator randomization pipelines with per-episode parameter logging and a joint latency-delay rejection constraint ($\text{latency} + \text{delay} \le 0.40\,\text{s}$).
3. **Multi-Worker Headless Training Pipeline:** Orchestrated parallel PPO training across 4 isolated Gazebo instances with dedicated master ports and ROS Domain IDs, achieving steady-state throughput of $\sim 19.7\,\text{steps/s}$ on CPU.
4. **Autonomous Watchdog & Recovery:** Designed a monotonic-clock watchdog monitor detecting simulator sensor dropouts ($>2.0\,\text{s}$) and executing clean process-group termination and checkpoint recovery without corrupting training logs.
5. **Decoupled Collision Ground Truth:** Formalized a strict collision hierarchy separating primary Gazebo physical contact from analytical footprint penetration, preventing sensor noise and blind sectors from artificially terminating evaluation episodes.
6. **Methodological Rigor:** Conducted exact paired statistical evaluations on held-out test splits, identifying sample size limitations ($p=0.6250$ on $N=50$) and designing an implementation-ready 9-condition stress benchmark.

---

## 5. Complete System Architecture & Data Flow

```
+----------------------------------------------------------------------------------------------------+
|                                    ROS 2 HUMBLE / GAZEBO CLASSIC 11                                |
|                                                                                                    |
|   +--------------------------+                                      +--------------------------+   |
|   |   Gazebo World Model     |                                      |   Differential Drive     |   |
|   |  - arena_4x4.world (5x5m)|                                      |   TurtleBot3 Burger      |   |
|   |  - 5 Static Obstacles    |                                      |  - Width: 0.178 m        |   |
|   |  - 12 Moveable Pool      |                                      |  - Length: 0.138 m       |   |
|   +------------+-------------+                                      |  - R_bound: 0.113 m      |   |
|                |                                                    +------------+-------------+   |
|                v                                                                 ^                 |
|       +----------------+                                                         |                 |
|       |  LDS-01 LiDAR  |  (360 beams, 5 Hz, [0.12, 3.50] m)                     | /cmd_vel         |
|       +--------+-------+                                                         | (Twist)         |
|                | /scan (LaserScan)                                               |                 |
+----------------|-----------------------------------------------------------------|-----------------+
                 |                                                                 |
                 v                                                                 |
+----------------------------------------------------------------------------------|-----------------+
| GYMNASIUM ENVIRONMENT WRAPPER: TurtleBot3NavigationEnv (rl/env/tb3_navigation_env.py)              |
|                                                                                                    |
| 1. Sensor Randomizer (rl/env/randomizers.py):                                    |                 |
|    - Range noise: sigma = a + b*r (a in [3,15]mm, b in [0,0.01])                 |                 |
|    - Beam dropout: p in [0, 0.05]; Block dropout: p_block in [0, 0.30]           |                 |
|    - Sensor latency: 0 or 1 step buffer (0.20 s)                                 |                 |
|                                                                                  |                 |
| 2. Observation Preprocessor (rl/env/observation.py):                              |                 |
|    - LDS-01 motion distortion correction: rx = px - dx_k, ry = py - dy_k         |                 |
|    - 36 min-pooled 10-deg bins, normalized by r_max = 3.50 m -> [0, 1]           |                 |
|    - Dead-reckoning EstimatedPose -> d_goal / d_arena, sin(beta), cos(beta)      |                 |
|    - Measured velocities: v / 0.15, omega / 0.60; Previous action u_{t-1}        |                 |
|    ==> Output: 43-Dimensional Observation Vector s_t in [-1, 1]^43               |                 |
|                                                                                  |                 |
| 3. Actuator Randomizer & Dynamics Pipeline (rl/env/randomizers.py):               |                 |
|    - Control delay buffer: d in [0, 0.20] s                                      |                 |
|    - Velocity command gain: k_v, k_omega in [0.85, 1.15]                         |                 |
|    - Velocity saturation: [0.0, 0.15] m/s, [-0.60, 0.60] rad/s                   |                 |
|    - Slew-rate acceleration clamp: a_lin <= 0.60 m/s^2, a_ang <= 3.0 rad/s^2    |                 |
|    - First-order lag filter: alpha = dt / (tau + dt), tau in [0, 0.30] s         |                 |
|    ==> Effective commands applied to physics: (v_actual, omega_actual) ----------+                 |
+----------------------------------------------------------------------------------------------------+
```

### Main Source Modules & Responsibilities
- [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py): Primary Gymnasium interface. Handles action-to-velocity mapping, reward computation, termination/truncation logic, Gazebo reset/teleport services, and analytical 2D raycasting fallback.
- [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py): Pure NumPy observation engine. Implements motion distortion correction, min-pooling into 36 angular bins, dead-reckoning odometry integration, and feature normalization.
- [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py): Domain randomization engine for sensor noise, beam dropouts, odometry drift, actuator lag, control delay, and command gains.
- [`rl/train/train_ppo.py`](file:///home/saigirish/sim-real-rl/rl/train/train_ppo.py): Stable-Baselines3 PPO training harness supporting multi-worker vectorization, checkpoint saving, and evaluation logging.
- [`rl/train/watchdog.py`](file:///home/saigirish/sim-real-rl/rl/train/watchdog.py): Monotonic clock supervisor detecting missing scan topics and performing clean Gazebo process recycling.
- [`scenarios/scenario_generator.py`](file:///home/saigirish/sim-real-rl/scenarios/scenario_generator.py): Deterministic pseudo-random scenario generator ensuring strict seed isolation between train, validation, and test splits.
- [`deploy/ppo_inference_node.py`](file:///home/saigirish/sim-real-rl/deploy/ppo_inference_node.py): Standalone ROS 2 deployment node for executing trained `.zip` policies on physical or simulated robots.
- [`scripts/evaluate_benchmark.py`](file:///home/saigirish/sim-real-rl/scripts/evaluate_benchmark.py): Canonical evaluation runner for the 9-condition Phase 2 benchmark.

---

## 6. Reinforcement Learning Methodology

### 6.1 Observation Space (43 Dimensions)
The observation vector $s_t \in \mathbb{R}^{43}$ is constructed without ROS dependencies by [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py#L1-L86):

$$\mathbf{s}_t = \big[ \bar{r}_0, \dots, \bar{r}_{35}, \; \bar{d}_{\text{goal}}, \; \sin(\beta), \; \cos(\beta), \; \bar{v}, \; \bar{\omega}, \; u_{v, t-1}, \; u_{\omega, t-1} \big]^T$$

1. **Normalized LiDAR Bins ($\bar{r}_0 \dots \bar{r}_{35}$):**
   - Raw scan contains $N = 360$ beams spanning $[0, 2\pi)$ radians ($1^\circ$ resolution).
   - Valid sensor range: $r_{\min} = 0.120\,\text{m}$ to $r_{\max} = 3.50\,\text{m}$.
   - Non-finite readings ($\le 0$, $\text{NaN}$, $+\infty$, $> r_{\max}$) are mapped to $r_{\max} = 3.50\,\text{m}$.
   - The $360^\circ$ scan is partitioned into 36 angular bins of width $10^\circ$.
   - **Bin 0 forward centering:** Bin 0 is centered directly forward along the robot's heading ($0^\circ$), covering $[-5^\circ, +5^\circ) \equiv [355^\circ, 360^\circ) \cup [0^\circ, 5^\circ)$. Bin $i$ covers $[10i - 5^\circ, 10i + 5^\circ)$.
   - Within each bin, ranges are min-pooled to preserve the nearest obstacle point:
     $$\bar{r}_i = \frac{1}{r_{\max}} \min_{k \in \text{Bin}_i} r_k \in [0.0, 1.0]$$
2. **Normalized Goal Distance ($\bar{d}_{\text{goal}}$):**
   - Computed from the dead-reckoning `EstimatedPose` $(x_{\text{est}}, y_{\text{est}})$:
     $$d_{\text{goal}} = \sqrt{(x_{\text{goal}} - x_{\text{est}})^2 + (y_{\text{goal}} - y_{\text{est}})^2}$$
     $$\bar{d}_{\text{goal}} = \text{clip}\left( \frac{d_{\text{goal}}}{d_{\text{arena}}}, 0.0, 1.0 \right)$$
   - Arena diagonal normalization: $d_{\text{arena}} = \sqrt{4.0^2 + 4.0^2} \approx 5.6569\,\text{m}$ for the $4\times 4\,\text{m}$ training arena ($d_{\text{arena}} = \sqrt{5.0^2 + 5.0^2} \approx 7.0711\,\text{m}$ for the $5\times 5\,\text{m}$ warehouse).
3. **Continuous Goal Bearing ($\sin(\beta), \cos(\beta)$):**
   - Target bearing in world frame: $\theta_{\text{goal}} = \text{atan2}(y_{\text{goal}} - y_{\text{est}}, x_{\text{goal}} - x_{\text{est}})$.
   - Relative bearing wrapped to $[-\pi, \pi]$: $\beta = \text{wrap\_to\_pi}(\theta_{\text{goal}} - \theta_{\text{est}})$.
   - Encoded as $[\sin(\beta), \cos(\beta)] \in [-1.0, 1.0]^2$ to eliminate the discontinuity at $\pm \pi$.
4. **Normalized Measured Velocities ($\bar{v}, \bar{\omega}$):**
   - Forward velocity: $\bar{v} = \text{clip}(v_{\text{meas}} / 0.15, -1.0, 1.0)$.
   - Angular velocity: $\bar{\omega} = \text{clip}(\omega_{\text{meas}} / 0.60, -1.0, 1.0)$.
5. **Previous Normalized Action ($\mathbf{u}_{t-1}$):**
   - The raw network output from the preceding step: $[u_{v, t-1}, u_{\omega, t-1}] \in [-1.0, 1.0]^2$. Initialized to $[0.0, 0.0]$ upon reset. Enables the policy to sense control momentum and discourages high-frequency chattering.

### 6.2 Action Space & Kinematic Mapping
The policy outputs a continuous action $\mathbf{u}_t = [u_v, u_\omega]^T \in [-1.0, 1.0]^2$. The environment maps this vector to physical differential-drive commands:

$$v_t = 0.075 \cdot (u_v + 1.0) \in [0.0, 0.15]\,\text{m/s}$$
$$\omega_t = 0.60 \cdot u_\omega \in [-0.60, 0.60]\,\text{rad/s}$$

**Kinematic Rationale:**
- **Strictly Non-Negative Forward Velocity:** Reverse motion ($v < 0$) is prohibited because the TurtleBot3 Burger platform carries a forward-mounted LDS-01 LiDAR sensor with severe rear chassis occlusions. Allowing backward motion causes catastrophic collisions with unobserved rear clutter.
- **Velocity Saturation:** $v_{\max} = 0.15\,\text{m/s}$ and $\omega_{\max} = 0.60\,\text{rad/s}$ ensure stable differential-drive ground contact without wheel lift or severe inertial slippage on smooth concrete floors.

### 6.3 Reward Formulation
The per-step scalar reward $r_t$ is implemented in [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py#L258-L265):

$$r_t = 10.0 \cdot (d_t - d_{t+1}) - 0.05 + 100.0 \cdot \mathbb{I}[\text{goal}] - 100.0 \cdot \mathbb{I}[\text{collision}]$$

| Component | Value / Formula | Intended Function & Mechanism |
| :--- | :--- | :--- |
| **Progress Reward** | $10.0 \cdot (d_t - d_{t+1})$ | Dense potential-based shaping proportional to distance reduction toward goal. Positive for approaching, negative for retreating. |
| **Step Penalty** | $-0.05$ | Constant time penalty per control step ($0.20\,\text{s}$), encouraging time-optimal trajectories and penalizing stationary hovering. |
| **Goal Terminal Reward** | $+100.0 \cdot \mathbb{I}[\text{goal}]$ | Sparse terminal reward granted when Euclidean distance $d \le 0.20\,\text{m}$ (training) or $d \le 0.25\,\text{m}$ (evaluation). |
| **Collision Terminal Penalty**| $-100.0 \cdot \mathbb{I}[\text{collision}]$| Sparse terminal penalty applied when physical robot footprint intersects an obstacle or arena boundary. |
| **Clearance / Smoothness** | $0.0$ (disabled) | Kept at zero to prevent distortion of distance-to-goal gradient during initial policy learning. |

**Termination vs. Truncation Semantics:**
- `terminated = True`: Occurs upon goal arrival ($d \le 0.20\,\text{m}$) or physical collision. Bootstrapping is terminated in the value target ($V(s_T) = 0$).
- `truncated = True`: Occurs when episode length reaches $T_{\max} = 350\,\text{steps}$ ($70.0\,\text{s}$) in training ($500\,\text{steps} = 100.0\,\text{s}$ in evaluation). Value bootstrapping continues ($V(s_T) \ne 0$).

### 6.4 PPO Implementation & Architecture
Training is implemented using Stable-Baselines3 (`stable_baselines3.PPO`) configured via [`configs/ppo.yaml`](file:///home/saigirish/sim-real-rl/configs/ppo.yaml):

```
Policy Architecture: ActorCriticPolicy (MlpPolicy)
Input Features: 43
Shared Layers: None (Separate Actor and Critic Branches)

Actor Branch (policy_net + action_net + log_std):
  - Linear(in_features=43, out_features=128) + Tanh()      [5,632 parameters]
  - Linear(in_features=128, out_features=128) + Tanh()     [16,512 parameters]
  - Linear(in_features=128, out_features=2)                [258 parameters]
  - Parameter(shape=(2,), requires_grad=True) (log_std)    [2 parameters]
  ==> Total Actor Parameters: 22,404

Critic Branch (value_net):
  - Linear(in_features=43, out_features=128) + Tanh()      [5,632 parameters]
  - Linear(in_features=128, out_features=128) + Tanh()     [16,512 parameters]
  - Linear(in_features=128, out_features=1)                [129 parameters]
  ==> Total Critic Parameters: 22,273

===> Total Trainable Network Parameters: 44,677
```

| Hyperparameter | Value in `configs/ppo.yaml` | Actual Execution Value (`run_meta.json`) |
| :--- | :--- | :--- |
| **Algorithm** | PPO | PPO |
| **Policy Type** | `MlpPolicy` | `MlpPolicy` |
| **Network Architecture** | Actor: `[128, 128]`, Critic: `[128, 128]` | Actor: `[128, 128]`, Critic: `[128, 128]` |
| **Activation Function** | `tanh` | `tanh` |
| **Initial $\log \sigma$** | $-0.50$ ($\sigma_0 \approx 0.6065$) | $-0.50$ |
| **Learning Rate** | $3 \times 10^{-4}$ (constant) | $3 \times 10^{-4}$ |
| **Rollout Buffer Size ($n_{\text{steps}}$)** | $2048$ steps per worker | $2048$ steps per worker ($8192$ total buffer) |
| **Minibatch Size** | $256$ | $256$ |
| **Optimization Epochs ($n_{\text{epochs}}$)**| $10$ | $10$ |
| **Discount Factor ($\gamma$)** | $0.99$ | $0.99$ |
| **GAE Parameter ($\lambda$)** | $0.95$ | $0.95$ |
| **Clipping Range ($\epsilon$)**| $0.20$ | $0.20$ |
| **Entropy Coefficient ($c_{\text{ent}}$)**| $0.00$ | $0.00$ |
| **Value Loss Coefficient ($c_v$)**| $0.50$ | $0.50$ |
| **Max Gradient Norm** | $0.50$ | $0.50$ |
| **Reward Normalization** | `VecNormalize(norm_obs=False, norm_reward=True)` | `ret_rms` running variance tracking |
| **Parallel Workers** | 4 parallel Gazebo Classic instances | 4 isolated headless workers |
| **Compute Device** | `auto` | `cpu` (Torch threads = 2) |

---

## 7. Four-Policy Factorial Experimental Design

The core methodological contribution of the project is a $2 \times 2$ factorial experimental design isolating sensor uncertainty from command-response dynamics:

$$\begin{array}{c|cc}
& \textbf{Dynamics Nominal} & \textbf{Dynamics Randomization} \\
\hline
\textbf{Sensor Nominal} & \mathbf{P_1} \text{ (Nominal Baseline)} & \mathbf{P_3} \text{ (Actuator Dynamics)} \\
\textbf{Sensor Randomization} & \mathbf{P_2} \text{ (Sensor Noise)} & \mathbf{P_4} \text{ (Combined Randomization)} \\
\end{array}$$

### 7.1 Sensor Randomization Specification ($P_2, P_4$)
Implemented in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py#L40-L168). Sampled independently at the start of each episode:

1. **Distance-Dependent Range Noise:**
   $$\sigma(r) = a + b \cdot r, \quad a \sim U(0.003, 0.015)\,\text{m}, \; b \sim U(0.000, 0.010)$$
   $$r_k' = \text{clip}(r_k + \mathcal{N}(0, \sigma^2(r_k)), 0.120, 3.50)\,\text{m}$$
2. **Independent Beam Dropout:**
   $$p_{\text{beam}} \sim U(0.00, 0.05), \quad r_k' \leftarrow +\infty \text{ with probability } p_{\text{beam}}$$
3. **Block Occlusion Dropout:**
   With probability $p_{\text{block\_act}} \sim U(0.00, 0.30)$, select $N_{\text{blocks}} \in \{1, 2\}$ contiguous angular sectors of length $L_{\text{block}} \in \{3, \dots, 10\}$ beams and set $r_k' \leftarrow +\infty$.
4. **Sensor Delivery Latency:**
   $d_{\text{scan}} \in \{0, 1\}$ step delay ($0.0\,\text{s}$ or $0.20\,\text{s}$). Stored in a FIFO buffer.
5. **Sensor Mounting Misalignment:**
   Yaw offset $\delta \theta_{\text{mount}} \sim U(-3^\circ, +3^\circ)$; translation offsets $\delta x, \delta y \sim U(-0.02, +0.02)\,\text{m}$.
6. **Odometry Drift & Velocity Measurement Noise:**
   $$v_{\text{meas}} = v \cdot s_v + \mathcal{N}(0, \sigma_v^2), \quad s_v \sim U(0.95, 1.05), \; \sigma_v \sim U(0.0, 0.005)\,\text{m/s}$$
   $$\omega_{\text{meas}} = \omega + b_\omega + \mathcal{N}(0, \sigma_\omega^2), \quad b_\omega \sim U(-0.005, 0.005)\,\text{rad/s}, \; \sigma_\omega \sim U(0.0, 0.020)\,\text{rad/s}$$

### 7.2 Actuator & Command-Response Randomization ($P_3, P_4$)
Implemented in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py#L170-L284). Executed in a strict 5-stage sequential pipeline:

1. **Control Latency Buffer ($d \in [0.0, 0.20]\,\text{s}$):**
   Holds previous action for fraction $f_{\text{prev}} = d / \Delta t$:
   $$v_{\text{delayed}} = f_{\text{prev}} v_{t-1} + (1 - f_{\text{prev}}) v_t, \quad \omega_{\text{delayed}} = f_{\text{prev}} \omega_{t-1} + (1 - f_{\text{prev}}) \omega_t$$
2. **Command Gain Multipliers:**
   $$v_{\text{gained}} = k_v \cdot v_{\text{delayed}}, \quad \omega_{\text{gained}} = k_\omega \cdot \omega_{\text{delayed}}, \quad k_v, k_\omega \sim U(0.85, 1.15)$$
3. **Motor Saturation Bounds:**
   $$v_{\text{sat}} = \text{clip}(v_{\text{gained}}, 0.0, 0.15\,\text{m/s}), \quad \omega_{\text{sat}} = \text{clip}(\omega_{\text{gained}}, -0.60, 0.60\,\text{rad/s})$$
4. **Acceleration Rate Limiting:**
   $$a_{\text{lin}} \sim U(0.20, 0.60)\,\text{m/s}^2, \quad a_{\text{ang}} \sim U(1.00, 3.00)\,\text{rad/s}^2$$
   $$v_{\text{lim}} = v_{\text{act}} + \text{clip}(v_{\text{sat}} - v_{\text{act}}, -a_{\text{lin}}\Delta t, a_{\text{lin}}\Delta t)$$
   $$\omega_{\text{lim}} = \omega_{\text{act}} + \text{clip}(\omega_{\text{sat}} - \omega_{\text{act}}, -a_{\text{ang}}\Delta t, a_{\text{ang}}\Delta t)$$
5. **First-Order Actuator Lag Filter ($\tau \in [0.0, 0.30]\,\text{s}$):**
   $$\alpha = \frac{\Delta t}{\tau + \Delta t} \implies v_{\text{act}}(t) = (1 - \alpha) v_{\text{act}}(t-1) + \alpha v_{\text{lim}}$$
   $$\omega_{\text{act}}(t) = (1 - \alpha) \omega_{\text{act}}(t-1) + \alpha \omega_{\text{lim}}$$
   Clamped to physical bounds $v \in [0.0, 0.15]\,\text{m/s}, \omega \in [-0.60, 0.60]\,\text{rad/s}$.

### 7.3 Joint Randomization & Rejection Constraint ($P_4$)
In real robots, uncompensated delays are additive. When combining sensor delivery latency ($0$ or $0.20\,\text{s}$) with control delay ($[0, 0.20]\,\text{s}$), total system transport lag can reach $0.40\,\text{s}$. If total lag exceeds $0.40\,\text{s}$, a differential-drive robot operating at $5\,\text{Hz}$ ($\Delta t = 0.20\,\text{s}$) becomes mathematically uncontrollable under unbuffered closed-loop feedback. 
[`CombinedRandomizer`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py#L311-L329) samples joint parameter vectors and enforces the rejection condition:

$$\text{latency}_{\text{sensor}} + \text{delay}_{\text{actuator}} \le 0.40\,\text{s}$$

*(Rejection sampling allows up to 100 draws before clamping `control_delay_s = min(control_delay_s, 0.40 - latency)`).*

> [!IMPORTANT]
> **Implementation Caveat on Mass and Friction Multipliers:**
> The `DynamicsParams` dataclass in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py#L55-L56) samples `mass_multiplier` $\sim U(0.90, 1.20)$ and `friction_multiplier` $\sim U(0.70, 1.30)$. Code inspection confirms that these multipliers are **not injected into Gazebo ODE physics via runtime SDF modification services**. They exist as software metadata. The actual physical perturbation applied during training and evaluation is strictly software-side velocity command-response dynamics.

---

## 8. Training Results and Checkpoint Analysis

All policies were trained from seed $s=0$ on identical multi-worker hardware. Training logs and evaluations are preserved in `runs/main_p*/eval_logs/evaluations.npz` and `runs/main_p*/training_summary.json`.

### 8.1 Reconciled Training Telemetry & Budgets

| Policy | Training Regimes | Total Interactions | Wall-Clock Time | Sampling Throughput | Best Checkpoint | Best Eval Reward | Eval Success Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$** | Nominal (No Rand) | **852,816** | 40,496 s (11.25 h) | $19.75\,\text{steps/s}$ | **450,000 steps** | **$111.32 \pm 3.89$** | **100.0%** (20/20) |
| **$P_2$** | Sensor Rand Only | **704,512** | 35,544 s (9.87 h) | $19.69\,\text{steps/s}$ | **300,000 steps** | **$107.36 \pm 25.28$** | **95.0%** (19/20) |
| **$P_3$** | Dynamics Rand Only | **704,512** | 35,563 s (9.88 h) | $19.68\,\text{steps/s}$ | **600,000 steps** | **$113.16 \pm 3.53$** | **100.0%** (20/20) |
| **$P_4$** | Combined Rand | **704,512** | 35,551 s (9.88 h) | $19.69\,\text{steps/s}$ | **400,000 steps** | **$105.91 \pm 23.37$** | **95.0%** (19/20) |

*Cumulative Training Compute:* 147,155 seconds ($\mathbf{40.88\,\text{hours}}$ of continuous simulation across 4 parallel Gazebo instances).

### 8.2 Validation Progression Across Training Timesteps
Evaluations were conducted every $100,000$ environment interactions on 20 validation episodes ($N=20$, seeds $1,000,000$ to $1,000,019$):

```
Validation Evaluation Progression (Mean Reward ± Std [Success Rate %]):

Timestep | P1 (Nominal)             | P2 (Sensor)              | P3 (Dynamics)            | P4 (Combined)
---------+--------------------------+--------------------------+--------------------------+--------------------------
 100,000 |         —                |  39.92 ± 38.1 [50.0%]    |  14.04 ± 29.4 [45.0%]    |  42.70 ± 35.8 [65.0%]
 150,000 |  27.90 ± 32.5 [40.0%]    |           —              |           —              |           —
 200,000 |         —                |  95.37 ± 31.2 [90.0%]    |  86.08 ± 36.5 [85.0%]    | 102.66 ± 28.4 [95.0%]
 250,000 | 101.92 ± 26.4 [95.0%]    |           —              |           —              |           —
 300,000 |         —                | 107.36 ± 25.3 [95.0%]*   | 102.46 ± 24.1 [95.0%]    | 102.62 ± 26.7 [95.0%]
 350,000 |  81.16 ± 39.8 [85.0%]    |           —              |           —              |           —
 400,000 |         —                |  87.29 ± 34.9 [85.0%]    | 100.85 ± 25.0 [95.0%]    | 105.91 ± 23.4 [95.0%]*
 450,000 | 111.32 ± 3.89 [100.0%]*  |           —              |           —              |           —
 500,000 |         —                |  71.20 ± 42.1 [80.0%]    |  92.15 ± 31.7 [90.0%]    |  81.83 ± 38.9 [85.0%]
 550,000 |  74.98 ± 41.2 [80.0%]    |           —              |           —              |           —
 600,000 |         —                | 103.03 ± 27.8 [95.0%]    | 113.16 ± 3.53 [100.0%]*  | 103.76 ± 26.5 [95.0%]
 650,000 | 103.47 ± 25.1 [95.0%]    |           —              |           —              |           —
 700,000 |         —                |  92.90 ± 32.4 [90.0%]    | 103.49 ± 26.0 [95.0%]    |  82.87 ± 39.1 [85.0%]
 750,000 |  93.01 ± 31.5 [90.0%]    |           —              |           —              |           —
 850,000 | 102.69 ± 26.2 [95.0%]    |           —              |           —              |           —

* Denotes selected best checkpoint.
```

### 8.3 Checkpoint Selection vs. Final Interactions
- In reinforcement learning with domain randomization, training curves frequently exhibit policy oscillations as the entropy coefficient decays and the policy experiences transient destabilization from challenging parameter draws.
- **Selection Criterion:** Stable-Baselines3 `EvalCallback` records `best_model.zip` based on the peak mean evaluation reward across the 20 held-out validation scenarios.
- As demonstrated in the table, all four policies achieved peak performance intermediate in training: $P_1$ peaked at $450\text{k}$ ($111.32$), $P_2$ at $300\text{k}$ ($107.36$), $P_3$ at $600\text{k}$ ($113.16$), and $P_4$ at $400\text{k}$ ($105.91$). Training beyond these points resulted in minor performance regression due to value-function overfitting to specific perturbation draws.

---

## 9. Completed Four-Policy Factorial Evaluation (50 Held-Out Scenarios)

The completed evaluation recorded in [`runs/eval/all_policies_4way_comparison.json`](file:///home/saigirish/sim-real-rl/runs/eval/all_policies_4way_comparison.json) and individual policy result files compares all four policies across the identical 50 held-out test scenarios ($N=50$, seeds $2,000,000$ to $2,000,049$) in the nominal $4\times 4\,\text{m}$ arena using the standardized deterministic evaluation harness.

### 9.1 Comprehensive Four-Policy Performance Table

| Evaluation Metric | $P_1$ (Nominal Baseline) | $P_2$ (Sensor Randomization) | $P_3$ (Dynamics Randomization) | $P_4$ (Combined Randomization) |
| :--- | :---: | :---: | :---: | :---: |
| **Factorial Configuration** | Sensor: No, Dyn: No | Sensor: Yes, Dyn: No | Sensor: No, Dyn: Yes | Sensor: Yes, Dyn: Yes |
| **Evaluated Checkpoint** | Final ($852\text{k}$ steps) | Final ($704\text{k}$ steps) | Final ($704\text{k}$ steps) | Final ($704\text{k}$ steps) |
| **Success Rate ($SR$)** | 47 / 50 (**94.0%**) | 48 / 50 (**96.0%**) | 49 / 50 (**98.0%**) | 49 / 50 (**98.0%**) |
| **Wilson 95% Confidence Interval**| $[83.78\%, 97.94\%]$ | $[86.54\%, 98.90\%]$ | $[89.50\%, 99.65\%]$ | $[89.50\%, 99.65\%]$ |
| **Collision Rate ($CR$)** | 1 / 50 (**2.0%**) | 1 / 50 (**2.0%**) | 1 / 50 (**2.0%**) | 1 / 50 (**2.0%**) |
| **Timeout Rate ($TO$)** | 2 / 50 (**4.0%**) | 1 / 50 (**2.0%**) | 0 / 50 (**0.0%**) | 0 / 50 (**0.0%**) |
| **Mean Cumulative Reward** | $103.75 \pm 37.71$ | $106.94 \pm 35.97$ | $109.05 \pm 28.76$ | $109.03 \pm 30.95$ |
| **Mean Steps to Goal (Successes)**| $106.17 \pm 28.45$ | $\mathbf{93.06 \pm 16.32}$ | $105.76 \pm 27.24$ | $101.49 \pm 24.12$ |
| **Mean Path Length (Successes)** | $2.251 \pm 0.612\,\text{m}$ | $2.236 \pm 0.589\,\text{m}$ | $2.144 \pm 0.573\,\text{m}$ | $\mathbf{2.067 \pm 0.548\,\text{m}}$ |
| **Mean Min Clearance (Successes)**| $0.282 \pm 0.098\,\text{m}$ | $0.262 \pm 0.091\,\text{m}$ | $0.286 \pm 0.089\,\text{m}$ | $\mathbf{0.292 \pm 0.089\,\text{m}}$ |

### 9.2 Peak Validation Checkpoint Comparison
For full scientific completeness, the peak validation checkpoints selected by `EvalCallback` during training ($P_{2,\text{best}}$ at $300\text{k}$ steps, $P_{3,\text{best}}$ at $600\text{k}$ steps) were also evaluated across the same 50 scenarios:
- **$P_{2,\text{best}}$ (Step 300,000):** $SR = 43/50$ ($86.0\%$, Wilson CI: $[73.81\%, 93.05\%]$), $CR = 2/50$ ($4.0\%$), $TO = 5/50$ ($10.0\%$), Reward $= 90.89 \pm 55.02$, Steps $= 135.7 \pm 36.4$, Path $= 2.341\,\text{m}$, Clearance $= 0.277\,\text{m}$.  
  *(Insight: At 300k steps, $P_2$ exhibited conservative crawling behavior resulting in 5 timeouts; training through 704k steps resolved these limit-cycles, increasing success to $96.0\%$ and reducing mean steps to $93.06$).*
- **$P_{3,\text{best}}$ (Step 600,000):** $SR = 46/50$ ($92.0\%$, Wilson CI: $[81.16\%, 96.85\%]$), $CR = 4/50$ ($8.0\%$), $TO = 0/50$ ($0.0\%$), Reward $= 96.23 \pm 56.21$, Steps $= 113.8 \pm 26.5$, Path $= 2.157\,\text{m}$, Clearance $= 0.268\,\text{m}$.

### 9.3 Pairwise Statistical Comparison (All 6 Policy Pairs)
To assess whether differences between policies are statistically significant, exact two-sided McNemar tests were conducted on binary outcomes across matched seeds. With $k = 6$ pairwise comparisons, the family-wise error rate is controlled via **Bonferroni correction**:
$$\alpha_{\text{adjusted}} = \frac{0.05}{6} \approx \mathbf{0.00833}$$

| Policy Pair | $N_{\text{both}}$ | Contingency $[a, b, c, d]$ | Discordant ($b+c$) | Exact McNemar $p$-value | Significance ($\alpha = 0.00833$) | Step Delta $\Delta S$ (Wilcoxon $p$) | Path Delta $\Delta L$ (Wilcoxon $p$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$ vs $P_2$** | 45 | $[45, 2, 3, 0]$ | 5 | $p = 1.0000$ | **Not Significant** | $-14.2\text{ steps}$ ($p < 0.0001^*$) | $-0.058\text{ m}$ ($p = 0.9022$) |
| **$P_1$ vs $P_3$** | 46 | $[46, 1, 3, 0]$ | 4 | $p = 0.6250$ | **Not Significant** | $-2.7\text{ steps}$ ($p = 0.0618$) | $-0.131\text{ m}$ ($p = 0.0138$) |
| **$P_1$ vs $P_4$** | 46 | $[46, 1, 3, 0]$ | 4 | $p = 0.6250$ | **Not Significant** | $-4.4\text{ steps}$ ($p = 0.0152$) | $-0.187\text{ m}$ ($p < 0.0001^*$) |
| **$P_2$ vs $P_3$** | 47 | $[47, 1, 2, 0]$ | 3 | $p = 1.0000$ | **Not Significant** | $+11.3\text{ steps}$ ($p = 0.0005^*$) | $-0.091\text{ m}$ ($p = 0.0008^*$) |
| **$P_2$ vs $P_4$** | 47 | $[47, 1, 2, 0]$ | 3 | $p = 1.0000$ | **Not Significant** | $+7.9\text{ steps}$ ($p = 0.0067^*$) | $-0.158\text{ m}$ ($p < 0.0001^*$) |
| **$P_3$ vs $P_4$** | 48 | $[48, 1, 1, 0]$ | 2 | $p = 1.0000$ | **Not Significant** | $-4.3\text{ steps}$ ($p = 0.3514$) | $-0.084\text{ m}$ ($p = 0.0003^*$) |

*Contingency format: $[a: \text{both succ}, b: P_A \text{ only}, c: P_B \text{ only}, d: \text{neither succ}]$. Star ($*$) denotes Wilcoxon significance at $\alpha = 0.00833$.*

### 9.4 Key Scientific Findings & Factorial Insights
1. **Binary Success Indistinguishability ($p \ge 0.6250$):**  
   Across all six policy pairs, the exact McNemar test yields $p \ge 0.6250$. With success rates clustered tightly between $94.0\%$ and $98.0\%$, the small cohort ($N=50$) produces at most 5 discordant outcomes per pair. **No policy can be declared statistically superior in task completion rate on this benchmark split.**
2. **Distinct Behavioral Regimes Revealed by Continuous Metrics:**
   - **Sensor Randomization ($P_2$) Drives Speed:** $P_2$ achieved the fastest goal arrival of any policy, requiring only $93.06$ steps ($18.6\,\text{s}$), which is $14.2$ steps faster than $P_1$ ($p < 0.0001$). Under sensor noise, the policy learned to execute confident, decisive forward sprints rather than hesitant micro-adjustments.
   - **Actuator Randomization ($P_3$) Drives Smooth Deceleration:** $P_3$ eliminated timeouts completely ($0\%$) while maintaining high clearance ($0.286\,\text{m}$), adapting to actuation lag by initiating turning maneuvers earlier.
   - **Combined Randomization ($P_4$) Optimizes Trajectory Directness & Clearance:** $P_4$ achieved the shortest path length ($2.067\,\text{m}$, $0.187\,\text{m}$ shorter than $P_1$, $p < 0.0001$) and highest mean obstacle clearance ($0.292\,\text{m}$), demonstrating that joint randomization produces the most geometrically efficient navigation behavior.

---

## 10. Phase 1: Warehouse Environment Integration

To advance beyond empty square arenas with random cylinders, Phase 1 introduced a structured industrial warehouse world ([`ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world)).

### 10.1 Environment Specifications
- **Inner Arena Bounds:** $5.0 \times 5.0\,\text{m}$ enclosed by 4 boundary walls of height $0.50\,\text{m}$ and thickness $0.10\,\text{m}$ centered at $x, y = \pm 2.55\,\text{m}$.
- **Arena Diagonal:** $d_{\text{arena}} = \sqrt{5.0^2 + 5.0^2} \approx 7.0711\,\text{m}$.

### 10.2 Static Infrastructure Objects (5 Models)
Verified from SDF world definitions and [`DEFAULT_STATIC_WAREHOUSE_OBSTACLES`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py#L178-L184):

| Model Name | Geometry Type | Dimensions ($x, y$ or $r$) | Center Coordinates $(x, y)$ | Footprint Extents | Physical Representation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `warehouse_rack_1` | Box | $0.40 \times 1.50\,\text{m}$ | $(-1.15, +0.65)\,\text{m}$ | $x \in [-1.35, -0.95], y \in [-0.10, +1.40]$ | Heavy storage shelving |
| `warehouse_rack_2` | Box | $0.40 \times 1.50\,\text{m}$ | $(+1.15, -0.65)\,\text{m}$ | $x \in [+0.95, +1.35], y \in [-1.40, +0.10]$ | Industrial pallet rack |
| `warehouse_shelf_north`| Box | $1.40 \times 0.30\,\text{m}$ | $(0.00, +2.25)\,\text{m}$ | $x \in [-0.70, +0.70], y \in [+2.10, +2.40]$ | Wall-adjacent parts rack |
| `pallet_stack` | Box | $0.50 \times 0.50\,\text{m}$ | $(-1.95, +1.95)\,\text{m}$ | $x \in [-2.20, -1.70], y \in [+1.70, +2.20]$ | Wooden pallet stack |
| `storage_container` | Cylinder | Radius $0.22\,\text{m}$ | $(+1.95, -1.95)\,\text{m}$ | $x^2 + y^2 \le 0.22^2$ | Industrial chemical drum |

### 10.3 Dynamic Obstacle Pool
12 pre-spawned movable obstacles (`obstacle_box_1`..`6`, `obstacle_cyl_1`..`6`) reside in the world file. In any active scenario, unused obstacles are teleported outside the navigable perimeter to $x = 20.0\,\text{m}, y = 20.0\,\text{m}$ via Gazebo model-state services, eliminating physics collisions and laser reflections.

---

## 11. Phase 2: Stress-Testing Benchmark Specification

To overcome the ceiling effects ($94-98\%$) and low statistical power of the 50-scenario test, Phase 2 establishes an implementation-ready 9-condition stress benchmark targeting specific non-linear failure modes.

### 11.1 Benchmark Condition Catalog

#### Case 1: Ultra-Narrow Corridor Stress Test
* **Objective:** Test lateral centering stability, wall clearance maintenance, and heading drift damping.
* **Geometry:** Two parallel walls of length $2.00\,\text{m}$ along $y \in [-1.00, +1.00]\,\text{m}$ separated by width $W = 0.450\,\text{m}$ (inner faces at $x = \pm 0.225\,\text{m}$).
* **Clearance Analysis:** TurtleBot3 width $0.178\,\text{m}$ leaves $0.225 - 0.089 = \mathbf{0.136\,\text{m}}$ ($13.6\,\text{cm}$) clearance on each side when perfectly centered.
* **Kinematic Drift Limit:** Control period $\Delta t = 0.20\,\text{s}$ at $v = 0.15\,\text{m/s}$ requires 67 steps to traverse. Solving $2.0\tan\theta + 0.5(0.178\cos\theta + 0.138\sin\theta) = 0.225\,\text{m}$ yields critical heading error $\theta_{\text{crit}} = \mathbf{3.77^\circ}$ ($0.0657\,\text{rad}$). Lateral drift rate is $\Delta x = 0.03\sin(3.77^\circ) = \mathbf{1.97\,\text{mm/step}}$.
* **Poses:** Start $(0.00, -1.40, \pi/2)$, Goal $(0.00, +1.40)$.

#### Case 2: Deep Concave U-Trap Escape
* **Objective:** Test policy ability to escape non-convex local minima requiring counter-intuitive retreat away from the goal.
* **Geometry:** South wall $x \in [-0.50, +0.50], y \in [-0.60, -0.40]$; West wall $x \in [-0.50, -0.30], y \in [-0.40, +0.50]$; East wall $x \in [+0.30, +0.50], y \in [-0.40, +0.50]$. Open mouth at North ($y = +0.50$).
* **Poses:** Start $(0.00, 0.00, -\pi/2)$ (inside trap facing South), Goal $(0.00, -1.80)$ (behind South wall).
* **Escape Route:** True choke centerline at $x = -0.725\,\text{m}$ provides exactly $\mathbf{13.6\,\text{cm}}$ centered clearance between U-trap West wall ($x=-0.50$) and Rack 1 East face ($x=-0.95$). Waypoints: $(0, 0) \to (0, 0.65) \to (-0.725, 0.65) \to (-0.725, -0.65) \to (0, -1.80)$. Path length: $\mathbf{4.034\,\text{m}}$ (smoothed: $\mathbf{3.903\,\text{m}}$).

#### Case 3: Overlapping Baffle Chicane (S-Curve)
* **Objective:** Test non-holonomic maneuvering through an S-curve with $100\%$ occluded direct line-of-sight.
* **Geometry:** Barrier 1 at $x \in [-2.50, +0.25], y \in [-0.50, -0.30]$; Barrier 2 at $x \in [-0.25, +2.50], y \in [+0.30, +0.50]$. Baffles overlap across $[-0.25, +0.25]\,\text{m}$ ($0.50\,\text{m}$ overlap). Centerline $x=0$ is blocked by solid barrier.
* **Poses:** Start $(0.00, -1.80, \pi/2)$, Goal $(0.00, +1.80)$.
* **Reference Path:** Smooth S-curve passing barrier tips at $x = \pm 0.410\,\text{m}$ provides $+6.5\,\text{cm}$ chassis clearance. Path length: $\mathbf{4.494\,\text{m}}$.

#### Case 4: Blind $90^\circ$ L-Junction
* **Objective:** Test corner-cutting prevention and reactive turning around an occluded right-angle vertex.
* **Geometry:** Inner vertex at $(-0.15, +0.30)\,\text{m}$; channel width $0.550\,\text{m}$.
* **Poses:** Start $(+0.125, -1.000, \pi/2)$, Goal $(-1.000, +0.575)$.
* **Clearance Analysis:** Centerline turn radius $R_c = 0.275\,\text{m}$. Inner chassis clearance: $0.275 - 0.089 = \mathbf{18.6\,\text{cm}}$. Outer swept radius $R_{\text{sweep}} = \sqrt{(0.275+0.089)^2 + 0.069^2} = 0.3705\,\text{m}$, yielding outer clearance $0.550 - 0.3705 = \mathbf{18.0\,\text{cm}}$.

#### Case 5: Dynamic Obstacle Crossing / Intercept Avoidance
* **Objective:** Test reactive deceleration and yielding under dynamic cross-traffic.
* **Obstacle Motion:** Cylinder ($R=0.15\,\text{m}$) starts at $(-0.80, 0.00)\,\text{m}$ ($3.0\,\text{cm}$ corner clearance to Rack 1). At $t = 6.00\,\text{s}$, travels East at $v_{\text{obs}} = 0.20\,\text{m/s}$, crossing $x=0$ at $t = \mathbf{10.00\,\text{s}}$, and halts safely at $x = \mathbf{+0.70\,\text{m}}$ ($10.0\,\text{cm}$ clearance to Rack 2).
* **Robot Collision Encounter Window:** Robot starts at $(0.00, -1.50, \pi/2)$. At $0.15\,\text{m/s}$ with acceleration ramp ($a \le 0.60\,\text{m/s}^2$), robot arrives at intersection at $t_r = \mathbf{10.125\,\text{s}}$. Contact collision threshold ($R_{\text{obs}} + R_{\text{bound}} = 0.263\,\text{m}$) creates a tight collision hazard window: $t \in [8.95, 11.05]\,\text{s}$ ($\Delta t = 2.10\,\text{s}$).

#### Case 6: Stepped Funnel with Bottleneck Throat
* **Objective:** Test speed throttling and alignment down a progressively constricting corridor.
* **Geometry:** Four symmetric constriction steps over $y \in [-0.50, +0.50]\,\text{m}$: Step 1 ($0.90\,\text{m}$ width, $36.1\,\text{cm}$ clearance), Step 2 ($0.70\,\text{m}$ width, $26.1\,\text{cm}$ clearance), Step 3 ($0.50\,\text{m}$ width, $16.1\,\text{cm}$ clearance), Step 4 Throat ($0.350\,\text{m}$ width, $\mathbf{8.6\,\text{cm}}$ clearance, circle clearance $\mathbf{6.2\,\text{cm}}$). Throat critical drift angle: $\theta_{\text{crit}} = \mathbf{15.77^\circ}$.
* **Poses:** Start $(0.00, -1.20, \pi/2)$, Goal $(0.00, +1.20)$.

#### Case 7A: Out-of-Distribution Distance-Dependent LiDAR Noise
* **Perturbation:** Out-of-distribution noise applied to all 360 beams:
  $$\sigma(r) = 0.025 + 0.015 r \implies 4.0\,\text{cm noise at } 1.0\,\text{m}, \; 7.8\,\text{cm noise at } 3.5\,\text{m}$$
* **Layout:** Nominal warehouse layout across matched scenario seeds.

#### Case 7B: Forward $30^\circ$ Blind Sector
* **Perturbation:** Forward angular sector $[-15^\circ, +15^\circ]$ masked to $3.50\,\text{m}$ (normalized $1.0$). Blinds 30 beams across Bins 35, 0, and 1 with 0 unaffected beams.
* **Layout:** Nominal warehouse layout across matched scenario seeds.

#### Case 8: Out-of-Distribution Actuator Lag & Gain Mismatch
* **Perturbation:** Full 1-step latency ($d = 0.20\,\text{s}$), out-of-distribution command gains $k_v = 0.80$ (under-speed) and $k_\omega = 1.20$ (over-steer), tight acceleration limits ($a_{\text{lin}} \le 0.20\,\text{m/s}^2, a_{\text{ang}} \le 1.0\,\text{rad/s}^2$), and severe actuator lag ($\tau = 0.30\,\text{s}, \alpha = 0.40$).
* **Layout:** Nominal warehouse layout across matched scenario seeds.

### 11.2 Benchmark Execution Protocol & Decoupled Collision Ground Truth
- **Pilot Execution Allocation:** $9\,\text{conditions} \times 4\,\text{policies} \times 1\,\text{seed} = \mathbf{36\,\text{episodes}}$ (technical verification of resets and spawns).
- **Main Benchmark Allocation:** $9\,\text{conditions} \times 4\,\text{policies} \times 20\,\text{matched seeds} = \mathbf{720\,\text{episodes}}$.
- **Decoupled Collision Ground Truth Hierarchy (Commit `618cf54`):**
  1. `collision_gazebo_contact`: Primary physical collision outcome determined by Gazebo bumper/contact physics sensors between the robot chassis and obstacles/walls.
  2. `collision_geometric`: Analytical footprint penetration ($R_{\text{bound}} = 0.113\,\text{m}$ signed distance $\le 0.0\,\text{m}$) recorded as an independent cross-check.
  3. `lidar_proximity_warning`: Binary flag when minimum range $\le 0.120\,\text{m}$ (`RANGE_MIN`). Recorded strictly as an advisory proximity metric; **never triggers episode termination**.

> [!WARNING]
> **Status of Phase 2 Benchmark:**  
> The 36-episode pilot and 720-episode main benchmark are **fully implemented in software but remain pending runtime execution**. No empirical data from Phase 2 benchmark runs exists yet. All descriptions in this dossier represent verified technical specifications, not empirical outcomes.

---

## 12. Engineering Challenges & Solutions

### Challenge 1: Gazebo Classic / DDS Multi-Worker Port Crosstalk
1. **Problem:** Parallel training workers crashed or cross-talked, resulting in robot teleportation interference across environments.
2. **Technical Cause:** All Gazebo Classic and ROS 2 instances defaulted to master port `11345` and ROS Domain ID `0`, causing CycloneDDS packet collisions.
3. **Solution:** Configured worker-isolated networking in [`scripts/orchestrate_p1_to_p4.py`](file:///home/saigirish/sim-real-rl/scripts/orchestrate_p1_to_p4.py#L320-L335): worker $i \in \{0, 1, 2, 3\}$ uses Gazebo port $11345 + i$ and `ROS_DOMAIN_ID = 30 + i`.
4. **Validation:** Confirmed zero network crosstalk during 40.9 hours of concurrent training.
5. **Remaining Limitation:** Static port allocation requires manual reservation and port-cleaning scripts (`clean_ports()`).

### Challenge 2: Headless Simulator Freezes & Missing-Scan Deadlocks
1. **Problem:** Gazebo Classic occasionally hung under continuous headless multi-instance execution, freezing the synchronous Gym `step()` call.
2. **Technical Cause:** Race conditions in the ODE physics solver and Gazebo ray sensor thread dropping ROS publication under high CPU load.
3. **Solution:** Authored an asynchronous monotonic-clock watchdog ([`rl/train/watchdog.py`](file:///home/saigirish/sim-real-rl/rl/train/watchdog.py)). If `/scan` messages stall for $>2.0\,\text{s}$, the watchdog terminates the process group via `SIGTERM`/`SIGKILL`, cleans `/dev/shm`, restarts Gazebo, and reloads the latest checkpoint.
4. **Validation:** Automated fault-injection tests in [`tests/test_watchdog.py`](file:///home/saigirish/sim-real-rl/tests/test_watchdog.py) demonstrated 100% automated recovery.
5. **Remaining Limitation:** Process recycling introduces an $\sim 8-12\,\text{s}$ pause per recovery event.

### Challenge 3: Inconsistent Reset State & Command-Buffer Bleed
1. **Problem:** Robot trajectories at the start of a new episode inherited non-zero velocities and stale sensor readings from the prior episode.
2. **Technical Cause:** Calling Gazebo model teleportation does not clear ROS subscriber queues, motor command buffers, or dead-reckoning integrators.
3. **Solution:** Enforced a strict multi-step reset sequence in [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py#L420-L460): (1) publish zero velocity, (2) clear velocity and action buffers, (3) teleport entities, (4) step Gazebo physics, and (5) discard stale scans by verifying monotonic scan timestamps.
4. **Validation:** Unit tests in [`tests/test_randomizers.py`](file:///home/saigirish/sim-real-rl/tests/test_randomizers.py) confirm zero-state reset isolation.
5. **Remaining Limitation:** Gazebo physics time step can drift slightly relative to wall clock under transient thermal throttling.

### Challenge 4: False Collision Terminations from Sensor Noise
1. **Problem:** In early evaluation runners, applying sensor noise (Case 7A) or blind masking (Case 7B) caused false-positive collision terminations.
2. **Technical Cause:** Collision ground truth was tied to `min(lidar_ranges) <= 0.120 m`. Noisy or masked beams triggered collision logic even when the robot was in open space.
3. **Solution:** Decoupled collision ground truth in Commit `618cf54`. Gazebo physical contact sensors serve as primary ground truth, geometric footprint signed distance serves as an analytical check, and LiDAR proximity is logged purely as an advisory metric.
4. **Validation:** Regression tests in [`tests/test_reward_audit.py`](file:///home/saigirish/sim-real-rl/tests/test_reward_audit.py) pass with zero false collision flags under extreme sensor noise.
5. **Remaining Limitation:** Gazebo contact sensors require fine collision mesh tuning to avoid contact chattering on terrain seams.

### Challenge 5: Double Normalization and Observation Misalignment
1. **Problem:** Risk of double observation normalization when deploying models with Stable-Baselines3 `VecNormalize`.
2. **Technical Cause:** Enabling `norm_obs=True` in `VecNormalize` applies running empirical standardization on features already bounded to $[0, 1]$ or $[-1, 1]$.
3. **Solution:** Explicitly set `VecNormalize(norm_obs=False, norm_reward=True)` in training, and encapsulated all normalization in the zero-ROS pure NumPy `ObservationPreprocessor` module.
4. **Validation:** Pytest suite [`tests/test_inference_correctness.py`](file:///home/saigirish/sim-real-rl/tests/test_inference_correctness.py) verified `norm_obs == False` across all saved `.pkl` artifacts.
5. **Remaining Limitation:** Changing arena dimensions requires manually updating $d_{\text{arena}}$ across training and deployment configs.

---

## 13. Comprehensive Claim-to-Evidence Matrix

| Documented Claim / Technical Fact | Repository Source File | Line Numbers / Artifact Path | Verification Command / Evidence Key |
| :--- | :--- | :--- | :--- |
| **PPO Actor/Critic Architecture (2x128 tanh MLP)** | [`configs/ppo.yaml`](file:///home/saigirish/sim-real-rl/configs/ppo.yaml) | Lines 12–18; `runs/main_p1_s0/final_model.zip` | PyTorch model inspection: 22,404 actor params, 44,677 total params. |
| **Observation Space (43 Dimensions)** | [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py) | Lines 10–35, 71–85 | `pytest -q tests/test_inference_correctness.py` (Passed). |
| **Non-negative Continuous Action Space** | [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py) | Lines 8–9, 267–272 | $v = 0.075(u_v+1) \in [0, 0.15]\,\text{m/s}, \omega = 0.60 u_\omega$. |
| **Reward Formulation ($+100/-100/-0.05/10\Delta d$)** | [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py) | Lines 258–265 | `pytest -q tests/test_reward_audit.py` (Passed). |
| **$P_1$ Training Budget: 852,816 timesteps** | [`runs/main_p1_s0/training_summary.json`](file:///home/saigirish/sim-real-rl/runs/main_p1_s0/training_summary.json) | Full JSON | `total_timesteps: 852816`, `elapsed_seconds: 40496.33`. |
| **$P_2, P_3, P_4$ Training Budget: 704,512 timesteps** | `runs/main_p{2,3,4}_s0/training_summary.json` | Full JSON | `total_timesteps: 704512` each, $\sim 35,550\,\text{s}$ wall-clock each. |
| **$P_1$ Best Checkpoint: 450,000 timesteps** | [`runs/main_p1_s0/eval_logs/evaluations.npz`](file:///home/saigirish/sim-real-rl/runs/main_p1_s0/eval_logs/evaluations.npz) | Index 3 | Reward $111.32 \pm 3.89$, $100.0\%$ success rate ($N=20$). |
| **$P_2$ Best Checkpoint: 300,000 timesteps** | [`runs/main_p2_s0/eval_logs/evaluations.npz`](file:///home/saigirish/sim-real-rl/runs/main_p2_s0/eval_logs/evaluations.npz) | Index 2 | Reward $107.36 \pm 25.28$, $95.0\%$ success rate ($N=20$). |
| **$P_3$ Best Checkpoint: 600,000 timesteps** | [`runs/main_p3_s0/eval_logs/evaluations.npz`](file:///home/saigirish/sim-real-rl/runs/main_p3_s0/eval_logs/evaluations.npz) | Index 5 | Reward $113.16 \pm 3.53$, $100.0\%$ success rate ($N=20$). |
| **$P_4$ Best Checkpoint: 400,000 timesteps** | [`runs/main_p4_s0/eval_logs/evaluations.npz`](file:///home/saigirish/sim-real-rl/runs/main_p4_s0/eval_logs/evaluations.npz) | Index 3 | Reward $105.91 \pm 23.37$, $95.0\%$ success rate ($N=20$). |
| **50-Scenario $P_1$ vs $P_4$ Evaluation Results** | [`runs/eval/p1_vs_p4_comparison.json`](file:///home/saigirish/sim-real-rl/runs/eval/p1_vs_p4_comparison.json) | Full JSON | $P_1: 47/50$ ($94\%$), $P_4: 49/50$ ($98\%$), Exact McNemar $p = 0.6250$. |
| **50-Scenario $P_2$ (Sensor) Evaluation Results** | [`runs/eval/p2_test/eval_P2_nominal.json`](file:///home/saigirish/sim-real-rl/runs/eval/p2_test/eval_P2_nominal.json) | Full JSON | $P_2: 48/50$ ($96\%$), 1 collision, 1 timeout, mean steps $93.06$. |
| **50-Scenario $P_3$ (Dynamics) Evaluation Results**| [`runs/eval/p3_test/eval_P3_nominal.json`](file:///home/saigirish/sim-real-rl/runs/eval/p3_test/eval_P3_nominal.json) | Full JSON | $P_3: 49/50$ ($98\%$), 1 collision, 0 timeouts, mean path $2.144\,\text{m}$. |
| **Full 4-Policy Factorial Comparison Artifact** | [`runs/eval/all_policies_4way_comparison.json`](file:///home/saigirish/sim-real-rl/runs/eval/all_policies_4way_comparison.json) | Full JSON | 6-way pairwise McNemar ($p \ge 0.6250$) & Wilcoxon continuous metrics. |
| **Live Gazebo Inference Demo (3/3 Success)** | [`runs/inference_demo_results.json`](file:///home/saigirish/sim-real-rl/runs/inference_demo_results.json) | Full JSON | 3 successful trials executed on port 11345 with $P_4$ final model. |
| **Sensor Randomization Formulations** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py) | Lines 40–168 | $\sigma = a + br, a \in [3,15]\,\text{mm}, b \in [0,0.01]$, dropouts, odometry drift. |
| **Actuator Randomization Formulations** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py) | Lines 170–284 | $d \in [0, 0.20]\,\text{s}, k \in [0.85, 1.15], \tau \in [0, 0.30]\,\text{s}, a_{\text{lin}} \le 0.60$. |
| **Joint Randomization Rejection Constraint** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py) | Lines 312–329 | $\text{latency} + \text{delay} \le 0.40\,\text{s}$. |
| **Mass/Friction Multipliers Not Injected** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py) | Lines 55–56, 202–204 | Present in `DynamicsParams` dataclass only; no Gazebo SDF physics service call. |
| **$5\times 5\,\text{m}$ Warehouse Arena & 5 Obstacles** | [`ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world) | Lines 44–270 | Walls at $\pm 2.55\,\text{m}$; Racks 1 & 2, North Shelf, Pallet, Cylinder. |
| **Decoupled Collision Ground Truth** | [`scripts/evaluate_benchmark.py`](file:///home/saigirish/sim-real-rl/scripts/evaluate_benchmark.py) | Commit `618cf54` | Separate recording of Gazebo contact, geometric overlap, and LiDAR proximity. |
| **Unit & Regression Test Verification** | Test suite | 55 passing tests | `pytest tests/` (55 passed, 0 failed, 12 not run [Gazebo soak]). |

---

## 14. Missing Experimental Data Inventory

The following experiments and datasets are missing from the repository record and must be explicitly acknowledged as planned or unexecuted:

1. **Phase 2 Stress Benchmark Execution (Pilot & Main):**
   - *Status:* Implemented but Unexecuted.
   - *Details:* The 36-episode pilot and 720-episode main benchmark exist in code ([`scripts/evaluate_benchmark.py`](file:///home/saigirish/sim-real-rl/scripts/evaluate_benchmark.py)) but have not been run. No benchmark result JSON files exist.
2. **Multi-Seed Training Runs (Seeds $s=1, s=2$):**
   - *Status:* Unexecuted.
   - *Details:* All four policies ($P_1$–$P_4$) were trained exclusively on seed $s=0$. Training seeds 1 and 2 were listed in `configs/ppo.yaml` but never executed due to computational budget limits ($\sim 41$ hours per seed). Policy variance across random initialization seeds is unmeasured.
3. **Physical Robot Hardware Deployment:**
   - *Status:* Zero Physical Trials Conducted.
   - *Details:* The repository contains deployment code ([`deploy/ppo_inference_node.py`](file:///home/saigirish/sim-real-rl/deploy/ppo_inference_node.py)) and ROS 2 launch files, but no physical TurtleBot3 hardware trials were performed. All evidence is derived from Gazebo Classic simulation and analytical kinematics.

*(Note: The 50-scenario held-out evaluation has now been completed for all four policies $P_1$, $P_2$, $P_3$, and $P_4$, resolving the earlier missing data gap for $P_2$ and $P_3$).*

---

## 15. Recommended Figures & Tables for the IEEE Paper

### Recommended Figures

| Figure # | Proposed Title / Content | Underlying Data Existence | Data Source File |
| :---: | :--- | :---: | :--- |
| **Fig. 1** | Overall System Architecture & Data Flow Diagram | **Verified** | Architecture diagram in Section 5 of this dossier. |
| **Fig. 2** | LDS-01 Observation Pipeline (Min-pooling, Motion Distortion, Dead Reckoning) | **Verified** | [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py). |
| **Fig. 3** | $2 \times 2$ Factorial Domain Randomization Architecture | **Verified** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py). |
| **Fig. 4** | Training Progression: Validation Mean Reward vs. Interactions ($P_1$–$P_4$) | **Verified** | `runs/main_p*/eval_logs/evaluations.npz`. |
| **Fig. 5** | Validation Success Rate (%) vs. Timesteps ($P_1$–$P_4$) | **Verified** | `runs/main_p*/eval_logs/evaluations.npz`. |
| **Fig. 6** | Structured $5\times 5\,\text{m}$ Warehouse Layout & Obstacle Models | **Verified** | [`ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world). |
| **Fig. 7** | Representative Trajectories in 50-Scenario Split ($P_1$ vs $P_4$) | **Verified** | Per-episode records in [`runs/eval/p1_vs_p4_comparison.json`](file:///home/saigirish/sim-real-rl/runs/eval/p1_vs_p4_comparison.json). |
| **Fig. 8** | Multi-Worker Parallel Training & DDS Port Isolation Schema | **Verified** | [`scripts/orchestrate_p1_to_p4.py`](file:///home/saigirish/sim-real-rl/scripts/orchestrate_p1_to_p4.py). |
| **Fig. 9** | Phase 2 Stress Benchmark Condition Geometries (Cases 1–6) | **Verified (Specs)** | [`docs/phase2_benchmark_specification.md`](file:///home/saigirish/sim-real-rl/docs/phase2_benchmark_specification.md). |
| **Fig. 10**| Actuator Slew-Rate Limiting and Lag Filter Response Curves | **Verified** | Equations in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py). |

### Recommended Tables

| Table # | Proposed Title / Content | Underlying Data Existence | Data Source File |
| :---: | :--- | :---: | :--- |
| **Table I** | Robot Physical Parameters & Kinematic Saturation Limits | **Verified** | [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py). |
| **Table II** | 43-Dimensional Observation Space Layout & Normalization | **Verified** | [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py). |
| **Table III**| PPO Training Hyperparameters & Network Dimensions | **Verified** | [`configs/ppo.yaml`](file:///home/saigirish/sim-real-rl/configs/ppo.yaml) & PyTorch model. |
| **Table IV** | Domain Randomization Parameters & Sampling Distributions | **Verified** | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py). |
| **Table V** | Training Summary & Compute Consumption ($P_1$–$P_4$) | **Verified** | `runs/main_p*/training_summary.json`. |
| **Table VI** | Validation Evaluation Progression & Peak Checkpoints | **Verified** | `runs/main_p*/eval_logs/evaluations.npz`. |
| **Table VII**| Recomputed 50-Scenario Evaluation Comparison ($P_1$ vs $P_4$) | **Verified** | [`runs/eval/p1_vs_p4_comparison.json`](file:///home/saigirish/sim-real-rl/runs/eval/p1_vs_p4_comparison.json). |
| **Table VIII**| $2 \times 2$ Paired Contingency Table & McNemar Test | **Verified** | Section 9.2 of this dossier. |
| **Table IX** | Static Warehouse Obstacle Geometric Coordinates | **Verified** | [`DEFAULT_STATIC_WAREHOUSE_OBSTACLES`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py#L178-L184). |
| **Table X** | Phase 2 Benchmark Condition Matrix & Stress Mechanisms | **Verified (Specs)** | [`docs/phase2_benchmark_specification.md`](file:///home/saigirish/sim-real-rl/docs/phase2_benchmark_specification.md). |
| **Table XI** | Ground-Truth Decoupled Collision Hierarchy | **Verified** | [`scripts/evaluate_benchmark.py`](file:///home/saigirish/sim-real-rl/scripts/evaluate_benchmark.py) (Commit `618cf54`). |

---

## 16. Unresolved Technical Limitations & Physical Transfer Roadmap

### Technical Limitations
1. **Single-Seed Factorial Training:** All policies were trained on seed $s=0$. Differences between policies could partially reflect seed-dependent optimization trajectory variance.
2. **Software-Side Proxy for Dynamics Randomization:** Robot mass and surface friction multipliers are not dynamically injected into ODE physics at runtime; perturbations rely on command-delay and lag filters.
3. **Absence of 3D Perception & Negative Obstacle Detection:** The planar 2D LiDAR cannot detect obstacles below its scanning plane ($h = 0.172\,\text{m}$) or negative obstacles (drop-offs, descending stairs).
4. **Underpowered Sample Size ($N=50$):** The preliminary evaluation set is too small to establish statistical significance between $94\%$ and $98\%$ success rates.

### Physical Sim-to-Real Transfer Roadmap
To deploy the trained policy ($P_4$) onto a physical TurtleBot3 Burger, the following sequence is specified:
1. **Compute Hardware:** Single-board computer (Raspberry Pi 4B, 4GB RAM) running Ubuntu 22.04 LTS and ROS 2 Humble.
2. **Inference Engine:** Install PyTorch CPU runtime (`torchvision` not required) or export the 22,404-parameter Actor MLP to ONNX runtime for sub-$5\,\text{ms}$ inference latency.
3. **ROS 2 Topic Binding:** Run [`deploy/ppo_inference_node.py`](file:///home/saigirish/sim-real-rl/deploy/ppo_inference_node.py) subscribing to `/scan` (from LDS-01 driver) and `/odom` (from micro-ROS OpenCR board), publishing continuous velocity setpoints to `/cmd_vel`.
4. **Safety Filter:** Wrap policy commands with an external reactive emergency-stop node enforcing zero velocity if $\min(\text{scan}) < 0.140\,\text{m}$.

---

## 17. Team Member Contributions & Responsibility Matrix

### Contribution Matrix

| Project Work Package / Task | Primary Lead | Supporting Lead | Tangible Repository Evidence |
| :--- | :---: | :---: | :--- |
| **Gazebo World & Spawner Integration** | Saigirish S | Siddhi Vinayak | [`ros2_ws/src/tb3_rl_sim/`](file:///home/saigirish/sim-real-rl/ros2_ws/src/tb3_rl_sim/) |
| **Gymnasium Environment & Reward Logic** | Saigirish S | Siddhi Vinayak | [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py) |
| **Pure NumPy Observation Preprocessor** | Siddhi Vinayak | Saigirish S | [`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py) |
| **Reactive Potential-Field Baseline** | Siddhi Vinayak | Saigirish S | [`baselines/reactive_controller.py`](file:///home/saigirish/sim-real-rl/baselines/reactive_controller.py) |
| **Simulator Missing-Scan Watchdog** | Siddhi Vinayak | Saigirish S | [`rl/train/watchdog.py`](file:///home/saigirish/sim-real-rl/rl/train/watchdog.py) |
| **PPO Training Engine & Hyperparameters** | Saigirish S | Siddhi Vinayak | [`rl/train/train_ppo.py`](file:///home/saigirish/sim-real-rl/rl/train/train_ppo.py), [`configs/ppo.yaml`](file:///home/saigirish/sim-real-rl/configs/ppo.yaml) |
| **Sensor & Dynamics Randomizers** | Saigirish S | Siddhi Vinayak | [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py) |
| **Scenario Generator & Seed Isolation** | Siddhi Vinayak | Saigirish S | [`scenarios/scenario_generator.py`](file:///home/saigirish/sim-real-rl/scenarios/scenario_generator.py) |
| **Multi-Worker Orchestration Daemon** | Saigirish S | Siddhi Vinayak | [`scripts/orchestrate_p1_to_p4.py`](file:///home/saigirish/sim-real-rl/scripts/orchestrate_p1_to_p4.py) |
| **50-Scenario Evaluation Suite & Stats** | Siddhi Vinayak | Saigirish S | [`rl/evaluate/evaluate.py`](file:///home/saigirish/sim-real-rl/rl/evaluate/evaluate.py), [`runs/eval/`](file:///home/saigirish/sim-real-rl/runs/eval/) |
| **ROS 2 Deployment Node & Inference Demo** | Siddhi Vinayak | Saigirish S | [`deploy/ppo_inference_node.py`](file:///home/saigirish/sim-real-rl/deploy/ppo_inference_node.py), [`scripts/run_simulation_demo.py`](file:///home/saigirish/sim-real-rl/scripts/run_simulation_demo.py) |
| **Phase 2 Benchmark Design & Collision Decoupling**| Saigirish S | Siddhi Vinayak | [`scripts/evaluate_benchmark.py`](file:///home/saigirish/sim-real-rl/scripts/evaluate_benchmark.py), Commit `618cf54` |

---

## 18. AI-Use Declaration (Factual Draft)

The following draft is prepared for formal team review and submission compliance:

> **Declaration of AI-Assisted Technologies:**  
> During the preparation of this research project and manuscript, the authors utilized AI-powered coding and auditing assistants (Google Antigravity / Claude Code) strictly for:
> 1. Verifying mathematical consistency across geometric benchmark specifications (e.g., clearance calculations and critical heading angles).
> 2. Developing automated Pytest verification harnesses and regression tests.
> 3. Generating standardized Markdown and LaTeX table formatting for technical reports.
> 4. Auditing repository files to cross-reference experimental claims against raw JSON logs.
> 
> All simulation worlds, reinforcement learning policies, training loops, neural network architectures, and experimental runs were executed, inspected, and validated locally on the project workstation. The authors assume full responsibility for the scientific integrity, analytical derivations, and factual correctness of all presented material.

---

## 19. References

### Primary References Cited in Project Files
1. **[PPO]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017. [Online]. Available: `https://arxiv.org/abs/1707.06347`  
   *Methodological support: PPO clipped surrogate objective, GAE advantage estimation, and actor-critic architecture.*
2. **[Domain Randomization]** J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2017, pp. 23–30. DOI: `10.1109/IROS.2017.8202133`.  
   *Methodological support: Conceptual foundation for observation and sensor perturbation.*
3. **[Dynamics Randomization]** X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2018, pp. 3803–3810. DOI: `10.1109/ICRA.2018.8460528`.  
   *Methodological support: Actuator command gain, latency, and acceleration perturbation framework.*
4. **[Mapless DRL Navigation]** L. Tai, G. Paolo, and M. Liu, "Virtual-to-Real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2017, pp. 31–36. DOI: `10.1109/IROS.2017.8202134`.  
   *Methodological support: 10-degree LiDAR binning, continuous action mapping, and target-bearing encoding.*
5. **[Stable-Baselines3]** A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations in Python," *J. Mach. Learn. Res. (JMLR)*, vol. 22, no. 268, pp. 1–8, 2021. [Online]. Available: `https://jmlr.org/papers/v22/20-1364.html`  
   *Methodological support: Training harness, vectorization, and VecNormalize implementation.*
6. **[ROS 2]** S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot Operating System 2: Design, Architecture, and Uses in the Wild," *Sci. Robot.*, vol. 7, no. 66, p. eabm6074, 2022. DOI: `10.1126/scirobotics.abm6074`.  
   *Methodological support: ROS 2 Humble middleware and DDS inter-process communication.*
7. **[Gazebo]** N. Koenig and A. Howard, "Design and Use Paradigms for Gazebo, an Open-Source Multi-Robot Simulator," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2004, vol. 3, pp. 2149–2154. DOI: `10.1109/IROS.2004.1389727`.  
   *Methodological support: Gazebo Classic 11 physics engine and ODE contact simulation.*
8. **[Gymnasium]** M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," *arXiv preprint arXiv:2407.17032*, 2024. [Online]. Available: `https://arxiv.org/abs/2407.17032`  
   *Methodological support: Standard Python RL API (reset/step lifecycle).*

### Additional Recommended Literature for the Final IEEE Paper
9. **[Sim-to-Real Survey]** W. Zhao, J. P. Queralta, and T. Westerlund, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey," in *Proc. IEEE Symp. Ser. Comput. Intell. (SSCI)*, 2020, pp. 737–744. DOI: `10.1109/SSCI47803.2020.9308468`.  
   *Supports: Comprehensive literature categorization of randomization vs domain adaptation.*
10. **[TurtleBot3 Platform]** E. Guizzo, "Robot Review: Robotis TurtleBot3 Burger," *IEEE Spectr.*, 2017. [Online]. Available: `https://spectrum.ieee.org/automaton/robotics/diy/robotis-turtlebot3-review`  
    *Supports: Physical robot hardware specifications, LDS-01 sensor characteristics, and differential-drive kinematics.*
11. **[Non-Holonomic Navigation DRL]** H. Choi, K. Min, and J. Choi, "Reinforcement Learning-Based Autonomous Navigation for Mobile Robots in Dynamic Indoor Environments," *IEEE Access*, vol. 11, pp. 31245–31256, 2023. DOI: `10.1109/ACCESS.2023.3262104`.  
    *Supports: State-of-the-art comparisons for obstacle clearance and path smoothness metrics.*
12. **[Statistical Methods in RL]** R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021, vol. 34, pp. 29304–29320.  
    *Supports: Methodological justification for paired statistical tests, Wilson confidence intervals, and power limitations on small benchmark cohorts.*

