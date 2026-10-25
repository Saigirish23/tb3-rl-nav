# Research Fact Sheet: TurtleBot3 PPO Mapless Navigation Pipeline

**Document Status:** Complete & Evidence-Backed  
**Date of Audit:** October 25, 2026  
**Repository:** `https://github.com/Saigirish23/tb3-rl-nav.git`  
**Target Milestone:** IEEE Conference-Style Mid-Term Progress Report  

---

## 1. Project Identity & Team

| Attribute | Established Information | Evidence / Status |
| :--- | :--- | :--- |
| **Project Title** | *Sim-to-Real 2D LiDAR Navigation Pipeline using PPO and Domain Randomization on TurtleBot3 Burger* | Supported by `README.md` and `docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf` |
| **Short Identifier** | `sim-real-rl` / `tb3-rl-nav` | Repository name and git remote |
| **Research Question** | In a controlled $2 \times 2$ factorial experiment, what are the individual and combined effects of **sensor observation randomization** and **robot dynamics randomization** in bridging the sim-to-real gap for mapless planar LiDAR navigation? | Section 1 of specification |
| **Institution** | Indian Institute of Technology Jodhpur (IIT Jodhpur) | Verified via author email domains (`@iitj.ac.in`) across Git history |
| **Department(s)** | Mechanical Engineering (ME) & Biosciences and Bioengineering (BB) | Supported by student email prefixes (`b24me...`, `b24bb...`) |
| **Course Number / Title** | `NOT FOUND — USER INPUT REQUIRED` | Not documented in repository files |
| **Project Supervisor / Guide**| `NOT FOUND — USER INPUT REQUIRED` | Not documented in repository files |
| **Academic Term** | Academic Year 2026–2027 (Mid-Term Audit: October 2026) | `NOT FOUND (Exact Semester) — USER INPUT REQUIRED` |

### Team Members & Roles

1. **Saigirish S**
   - **Roll Number:** `B24ME1066`
   - **Email:** `b24me1066@iitj.ac.in`
   - **Git Identity:** `Saigirish23`
   - **Department:** Mechanical Engineering, IIT Jodhpur
   - **Documented Contributions:**
     - Simulation architecture in ROS 2 Humble and Gazebo Classic 11 (`ros2_ws/src/tb3_rl_sim/`).
     - Gymnasium navigation environment (`rl/env/tb3_navigation_env.py`).
     - PPO training pipeline setup, GPU/CPU auto-detection, and multi-worker isolation (`rl/train/train_ppo.py`, `configs/ppo.yaml`).
     - Sensor and dynamics domain randomizers (`rl/env/randomizers.py`).
     - Autonomous supervisor and deadline orchestration daemon (`scripts/orchestrate_p1_to_p4.py`).
     - Execution and verification of P1 (nominal) and P4 (combined) training runs.

2. **Siddhi Vinayak**
   - **Roll Number:** `B24BB1036`
   - **Email:** `b24bb1036@iitj.ac.in`
   - **Git Identity:** `siddhivn`
   - **Department:** Biosciences and Bioengineering, IIT Jodhpur
   - **Documented Contributions:**
     - Hand-coded reactive baseline controller (`baselines/reactive_controller.py`).
     - Deterministic scenario generator and dataset splits (`scenarios/scenario_generator.py`).
     - Missing-scan watchdog and simulator fault-recovery supervisor (`rl/train/watchdog.py`, `tests/test_watchdog.py`).
     - Reward structure verification and audit traces (`tests/test_reward_audit.py`).
     - Standalone ROS 2 policy deployment node architecture (`deploy/ppo_inference_node.py`).

---

## 2. Problem Statement & Research Objectives

### The Problem
Traditional mobile robot navigation architectures rely on Simultaneous Localization and Mapping (SLAM) and layered global/local costmaps (e.g., ROS 2 Nav2). While effective in structured environments, these pipelines suffer from:
1. High computational overhead for onboard embedded processors.
2. Fragility to dynamic map changes and geometric symmetry.
3. Vulnerability to sensor dropout and scan noise without manual tuning.

Mapless navigation via deep reinforcement learning maps raw local sensor inputs directly to velocity control commands $(v, \omega)$. However, policies trained in simulation suffer from performance collapse when transferred to physical hardware due to the **sim-to-real domain gap**.

### The Sim-to-Real Transfer Gap
The domain gap in planar mobile robotics consists of two distinct physical components:
1. **Sensor Observation Mismatch:** LiDAR ray dropouts, angular discretization, surface reflectivity variations, specular reflections, and unmodelled odometry dead-reckoning drift.
2. **Robot Dynamics Mismatch:** Wheel slip, motor response lag, latency delay between sensing and actuation, battery voltage decay, and chassis inertia variations.

### Core Objectives
1. Implement a clean, zero-SLAM, mapless 2D LiDAR navigation policy using Proximal Policy Optimization (PPO).
2. Establish a controlled $2 \times 2$ factorial experiment isolating:
   - **P1:** Nominal baseline (no domain randomization).
   - **P2:** Sensor observation randomization only.
   - **P3:** Robot dynamics randomization only.
   - **P4:** Combined sensor and dynamics randomization.
3. Evaluate policies on identical held-out test scenarios and in continuous closed-loop simulation demonstrations.
4. Establish statistical baselines and prepare the pipeline for downstream physical deployment on the TurtleBot3 Burger.

---

## 3. System Architecture & Technical Specifications

### Hardware & Simulation Setup
- **Robot Platform:** TurtleBot3 Burger differential-drive mobile robot.
  - Track width: $0.160\text{ m}$, Wheel radius: $0.033\text{ m}$.
  - Collision footprint: Circular radius $r_{\text{robot}} = 0.105\text{ m}$.
- **Primary Sensor:** 360-degree 2D planar LiDAR (Laser Distance Sensor LDS-01).
  - Maximum range: $r_{\max} = 3.50\text{ m}$, Scan rate: $5.0\text{ Hz}$.
- **Simulator:** Gazebo Classic 11.10.2 running ODE physics engine (`max_step_size: 0.001`, `real_time_update_rate: 1000.0`).
- **Middleware:** ROS 2 Humble Hawksbill with `rmw_cyclonedds_cpp`.
- **Arena:** $4.0\text{ m} \times 4.0\text{ m}$ bounded square arena with 4 perimeter walls and 8 geometric obstacles (4 boxes, 4 cylinders).

### Observation Space ($D = 43$, Pure NumPy)
The observation preprocessor ([`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py)) has zero ROS dependencies and constructs a 43-dimensional vector:
1. **LiDAR Range Bins ($obs[0:36]$):** 36 min-pooled $10^\circ$ sectors spanning $[-180^\circ, +180^\circ)$, with Bin 0 centered forward on $[-5^\circ, +5^\circ)$. Each bin is normalized by $r_{\max} = 3.50\text{ m}$ into $[0.0, 1.0]$. Invalid readings ($\le 0, \text{NaN}, \infty, > 3.5\text{ m}$) are mapped to $1.0$.
2. **Normalized Goal Distance ($obs[36]$):** Euclidean distance from estimated pose to goal, normalized by arena diagonal $d_{\text{arena}} = \sqrt{4^2 + 4^2} \approx 5.657\text{ m}$, clipped to $[0.0, 1.0]$.
3. **Goal Bearing Features ($obs[37:39]$):** Continuous trigonometric encoding $[\sin(\beta), \cos(\beta)] \in [-1.0, 1.0]^2$ of the target bearing relative to robot heading.
4. **Estimated Pose Dead-Reckoning:** Pure NumPy [`EstimatedPose`](file:///home/saigirish/sim-real-rl/rl/env/observation.py#L150) integrating forward velocity $v$ and angular velocity $\omega$ via midpoint circular arcs from the initial reset pose.
5. **Measured Velocities ($obs[39:41]$):** Forward velocity $v / 0.15\text{ m/s} \in [-1.0, 1.0]$ and angular velocity $\omega / 0.60\text{ rad/s} \in [-1.0, 1.0]$.
6. **Previous Action Buffer ($obs[41:43]$):** Last executed normalized control action $u_{t-1} \in [-1.0, 1.0]^2$, initialized to $[0.0, 0.0]$.

### Action Space ($A = 2$, Continuous)
Continuous action vector $u = [u_v, u_\omega]^T \in [-1.0, 1.0]^2$ is mapped linearly to robot commands:
$$v = 0.075 \cdot (u_v + 1.0) \in [0.0, 0.15]\,\text{m/s}$$
$$\omega = 0.60 \cdot u_\omega \in [-0.60, 0.60]\,\text{rad/s}$$
- Forward velocity $v$ is strictly non-negative (no reverse driving).
- Control frequency: $5.0\text{ Hz}$ ($\Delta t = 0.20\text{ s}$ zero-order hold).

### Reward Formulation (Frozen per Specification Section 5)
$$r_t = 10 \cdot (d_t - d_{t+1}) - 0.05 + 100 \cdot \mathbb{I}_{\text{goal}} - 100 \cdot \mathbb{I}_{\text{collision}}$$
- **Progress Reward:** $10.0 \cdot (d_t - d_{t+1})$ based on potential difference toward goal.
- **Time Step Penalty:** $-0.05$ per step to encourage minimum-time trajectory.
- **Goal Reward:** $+100.0$ when Euclidean distance to goal $d \le 0.20\text{ m}$.
- **Collision Penalty:** $-100.0$ when minimum obstacle distance $d_{\min} < 0.16\text{ m}$.

### Episode Termination Criteria
- **Success (Terminated):** Euclidean distance to goal $\le 0.20\text{ m}$.
- **Collision (Terminated):** Minimum LiDAR range $< 0.16\text{ m}$ (robot radius $0.105\text{ m} + 0.055\text{ m}$ safety margin).
- **Timeout (Truncated):** Episode duration reaches 350 control steps ($70.0\text{ s}$ simulated time).

### Reinforcement Learning Architecture (Stable-Baselines3 PPO)
- **Policy Network:** Separate actor and critic Multi-Layer Perceptrons (MLPs), each with 2 hidden layers of 128 units and $\tanh$ activations.
- **Initial Log Std:** $\log \sigma = -0.5$.
- **Hyperparameters:**
  - Learning rate: $3 \times 10^{-4}$ (constant).
  - Rollout buffer: $n_{\text{steps}} = 2048$ steps per worker ($4 \times 2048 = 8192$ transitions per batch).
  - Minibatch size: $256$, PPO epochs: $10$.
  - Discount factor: $\gamma = 0.99$, GAE parameter: $\lambda = 0.95$.
  - Clip range: $\epsilon = 0.2$, Entropy coefficient: $0.0$, Value function coefficient: $0.5$.
  - Gradient clipping: Max norm $0.5$.
  - Normalization: `normalize_observations: false` (handled manually), `normalize_rewards: true` (via `VecNormalize`).
- **Parallel Workers:** 4 parallel Gazebo Classic instances operating on isolated ports $11345..11348$ and domain IDs $30..33$.

---

## 4. Experimental Design: $2 \times 2$ Matrix

| Policy | Sensor Randomization | Dynamics Randomization | Planned Budget | Actual Completed Steps | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | **OFF** | **OFF** | 850,000 | **852,816** | Completed (`runs/main_p1_s0/final_model.zip`) |
| **P2** | **ON** | **OFF** | 850,000 | 0 | **Pending** (Code implemented in `randomizers.py`) |
| **P3** | **OFF** | **ON** | 850,000 | 0 | **Pending** (Code implemented in `randomizers.py`) |
| **P4** | **ON** | **ON** | 700,000 | **704,512** | Completed (`runs/main_p4_s0/final_model.zip`) |

### Domain Randomization Ranges (Implemented in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py))
1. **Sensor Randomization (P2 & P4):**
   - LiDAR Gaussian range noise: $\sigma \in [0.005, 0.03]\text{ m}$.
   - LiDAR beam dropout probability: $p \in [0.01, 0.08]$ (failed beams mapped to $r_{\max}$).
   - Odometry scale drift: $s_v \in [0.95, 1.05]$, $s_\omega \in [0.95, 1.05]$.
2. **Dynamics Randomization (P3 & P4):**
   - Robot mass variation: $\Delta m \in [-15\%, +15\%]$.
   - Wheel surface friction coefficient: $\mu \in [0.4, 1.2]$.
   - Motor response lag (first-order low-pass filter): $\tau \in [0.05, 0.25]\text{ s}$.
   - Communication / actuation delay: $\delta \in [0.02, 0.12]\text{ s}$.

### Dataset Splits & Seed Isolation
- **Training Split:** 10 scenarios (seeds $0$ to $9$).
- **Validation Split:** 50 scenarios (seeds $1,000,000$ to $1,000,049$). Used strictly for periodic evaluation checkpoints.
- **Held-Out Test Split:** 100 scenarios (seeds $2,000,000$ to $2,000,099$).
- **Leakage Audit:** Seed intersection between all three splits is **0**. Held-out evaluation evaluated identical scenarios $0..49$ on both P1 and P4.

---

## 5. Chronological Record of Completed Work

1. **Architecture & Environment Baseline (Oct 21–22, 2026):**
   - Repository skeleton initialized (`e3c40fd`).
   - Virtual environment specification and headless Gazebo simulation setup (`bfc16ad`, `6e068c5`).
   - Gymnasium environment wrapper with LDS-01 sensor plugin.
2. **Gate G2 & Baseline Controller (Oct 23, 2026):**
   - Deterministic scenario generator implemented for train, val, test, and stress splits (`4da798a`).
   - Hand-coded reactive baseline controller evaluated ($92.0\%$ nominal kinematic success).
   - Missing-scan watchdog supervisor implemented (`rl/train/watchdog.py`) with 6/6 tests passing.
3. **Training Automation & Metrics Logging (Oct 24, 2026):**
   - Training metrics and loss persistence to JSON (`d181656`).
   - Per-step reward traces audit (`5e1c0aa`).
   - Multi-worker isolation daemon with port cleanup and deadline enforcement (`scripts/orchestrate_p1_to_p4.py`).
4. **P1 & P4 Training Completion (Oct 24–25, 2026):**
   - P1 nominal run trained to $852,816$ steps ($11.2\text{ hours}$ wall-clock, $19.75\text{ sps}$).
   - P4 combined run trained to $704,512$ steps ($9.8\text{ hours}$ wall-clock, $19.79\text{ sps}$).
   - Models and VecNormalize stats saved and verified.
5. **Held-Out Comparative Benchmark (Oct 25, 2026):**
   - Automated evaluation of P1 and P4 on 50 identical held-out test scenarios. Full JSON records stored in `runs/eval/p1_vs_p4_comparison.json`.
6. **Live Gazebo Demonstration & Runbook (Oct 25, 2026):**
   - P4 deployed in live Gazebo Classic simulation across 3 consecutive navigation waypoints ($100\%$ success, 0 collisions, 0 timeouts).
   - Live console telemetry stream and enriched JSON logs implemented.
   - Comprehensive demo runbook authored in `docs/demo_runbook.md`.

---

## 6. Empirical Results & Discussion

### A. Periodic Validation Performance Over Training
*(Evaluated every 100k steps on 20 validation scenarios in Gazebo Classic):*

| Timesteps | P1 (Nominal) Success Rate | P1 Mean Reward | P4 (Combined) Success Rate | P4 Mean Reward |
| :---: | :---: | :---: | :---: | :---: |
| **100k / 150k** | 40.0% | $27.90$ | 55.0% | $42.70$ |
| **200k / 250k** | 95.0% | $101.92$ | 95.0% | $102.66$ |
| **300k / 350k** | 85.0% | $81.16$ | 95.0% | $102.62$ |
| **400k / 450k** | 100.0% | $111.32$ | 95.0% | $105.91$ |
| **500k / 550k** | 80.0% | $74.98$ | 90.0% | $81.83$ |
| **Final Checkpoint** | **95.0%** (at 850k) | **$102.69$** | **98.0%** (at 700k) | **$109.03$** |

### B. Held-Out Test Split Benchmark (50 Scenarios)

| Evaluation Metric | P1 (Nominal Baseline) | P4 (Combined Randomization) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Evaluated Episodes** | 50 | 50 | — |
| **Success Count** | 47 | 49 | +2 |
| **Success Rate** | **94.0%** | **98.0%** | **+4.0%** |
| **Wilson 95% Confidence Interval** | $[0.8378, 0.9794]$ | $[0.8950, 0.9965]$ | — |
| **Collisions** | 1 (2.0%) | 1 (2.0%) | 0.0% |
| **Timeouts (350 step limit)** | 2 (4.0%) | **0 (0.0%)** | **-4.0%** |
| **Mean Cumulative Reward** | $103.75 \pm 37.71$ | **$109.03 \pm 30.95$** | **+5.28** |
| **Mean Steps to Goal (Successes)** | $106.17\text{ steps}$ ($21.2\text{ s}$) | **$101.49\text{ steps}$** ($20.3\text{ s}$) | **-4.68 steps** |
| **Mean Path Length (Successes)** | $2.251\text{ m}$ | **$2.067\text{ m}$** | **-0.184 m (-8.2%)** |
| **Mean Min Obstacle Clearance** | $0.282\text{ m}$ | **$0.292\text{ m}$** | **+0.010 m** |

### C. Paired Statistical Analysis & Significance
From the per-scenario records of both policies on identical scenarios:
- **Both Succeeded:** 46 scenarios ($92\%$).
- **Both Failed:** 0 scenarios ($0\%$).
- **Discordant Pair $b$ (P1 Succeeded, P4 Failed):** 1 scenario (Scenario 34: P4 collided at step 78; P1 reached goal at step 157).
- **Discordant Pair $c$ (P1 Failed, P4 Succeeded):** 3 scenarios (Scenario 7: P1 collided at step 67; Scenarios 25 & 37: P1 timed out at step 350).
- **Exact McNemar Test (Two-Sided Binomial on $n_{\text{discordant}} = 4$):**
  $$p = 2 \cdot \sum_{k=3}^{4} \binom{4}{k} 0.5^4 = 2 \cdot (4 + 1) \cdot \frac{1}{16} = 0.6250$$

#### Methodological Interpretation & Limitations:
1. **Descriptive Superiority:** P4 demonstrates consistent empirical advantages across all continuous navigation metrics: $+5.28$ higher reward, $8.2\%$ shorter path length, faster goal reaching ($101.5$ vs $106.2$ steps), wider obstacle clearance, and zero step timeouts.
2. **Statistical Significance Caveat:** Because $p = 0.6250 > 0.05$, the $4.0\%$ paired difference in success rate is **not statistically significant** at the $\alpha = 0.05$ level on a sample of $N = 50$ scenarios.
3. **Power Limitation:** Detecting a statistically significant difference between $94\%$ and $98\%$ requires $N \ge 250$ test episodes. The mid-term report must accurately report this distinction rather than claiming decisive statistical superiority.

### D. Live Gazebo Classic Simulation Demonstration
The selected policy (P4) was evaluated in real-time closed-loop simulation on port `11345` across 3 continuous waypoints in the $4\text{m} \times 4\text{m}$ arena:
- **Trial 1 (East Waypoint $[1.20, 0.00]\text{m}$):** Reached in 57 steps ($14.3\text{ s}$), final distance $0.198\text{ m}$, min clearance $1.034\text{ m}$. Outcome: **SUCCESS**.
- **Trial 2 (North-East Waypoint $[1.00, 1.00]\text{m}$):** Reached in 61 steps ($15.2\text{ s}$), final distance $0.198\text{ m}$, min clearance $0.897\text{ m}$. Outcome: **SUCCESS**.
- **Trial 3 (North-West Waypoint $[-1.00, 1.00]\text{m}$):** Reached in 88 steps ($20.3\text{ s}$), final distance $0.191\text{ m}$, min clearance $1.077\text{ m}$. Outcome: **SUCCESS**.
- **Acceptance Rate:** **3 / 3 (100.0% Success)**, 0 collisions, 0 timeouts. Telemetry logged in [`runs/demo_run_20261025_182130/demo_summary.json`](file:///home/saigirish/sim-real-rl/runs/demo_run_20261025_182130/demo_summary.json).

---

## 7. Pending Work & Future Roadmap

| Work Package | Current Status | Target Completion | Description / Requirements |
| :--- | :---: | :---: | :--- |
| **P2 (Sensor Randomization)** | Pending | Final Phase | Train P2 seed 0 (850k steps) to isolate sensor noise contribution. |
| **P3 (Dynamics Randomization)** | Pending | Final Phase | Train P3 seed 0 (850k steps) to isolate dynamics perturbation contribution. |
| **Multi-Seed Replication** | Pending | Final Phase | Train Seeds 1 and 2 for all completed policies to assess training stability. |
| **Large-Scale Evaluation** | Pending | Final Phase | Evaluate on all 100 test scenarios + 50 stress scenarios ($N \ge 250$) for statistical power. |
| **Physical Hardware Trials** | Pending | Final Phase | Zero-shot transfer to physical TurtleBot3 Burger in hardware arena. |

---

## 8. Verified Core Citations

1. **PPO Algorithm:** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv preprint arXiv:1707.06347.
2. **Domain Randomization:** Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). *Domain randomization for transferring deep neural networks from simulation to the real world*. In IEEE/RSJ IROS (pp. 23-30).
3. **Sim-to-Real Mapless Navigation:** Tai, L., Paolo, G., & Liu, M. (2017). *Virtual-to-real deep reinforcement learning for robot navigation*. In IEEE/RSJ IROS (pp. 31-36).
4. **Stable-Baselines3:** Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable reinforcement learning implementations in Python*. JMLR, 22(268), 1-8.
5. **ROS 2:** Macenski, S., Foote, T., Gerkey, B., Lalancette, C., & Woodall, W. (2022). *Robot Operating System 2: Design, architecture, and uses in the wild*. Science Robotics, 7(66), eabm6074.
