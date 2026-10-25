# Sim-to-Real 2D LiDAR Navigation Pipeline (`tb3-rl-nav`)

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Gazebo Classic 11](https://img.shields.io/badge/Gazebo-Classic%2011-orange.svg)](http://gazebosim.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source reinforcement learning pipeline evaluating the independent and combined effects of **sensor observation randomization** and **robot dynamics domain randomization** for mapless 2D LiDAR navigation on the TurtleBot3 Burger.

Authoritative specification: [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf).

---

## 1. Research Overview & Objectives

### Problem Statement
Standard mobile navigation frameworks rely heavily on Simultaneous Localization and Mapping (SLAM) and layered costmaps (e.g., Nav2). While reliable in static mapped areas, they present high onboard computational burdens and can fail in featureless, dynamic, or unmapped environments. Mapless navigation using deep reinforcement learning maps raw local range sensing directly to motor actuation $(v, \omega)$. However, transferring simulation-trained policies to physical robots encounters the **sim-to-real transfer gap** caused by unmodelled sensing noise (beam dropouts, odometry drift) and vehicle dynamics (wheel slip, motor lag, latency).

### Research Question
In a controlled $2 \times 2$ factorial experiment, what are the individual and joint contributions of:
1. **Sensor Observation Randomization** (LiDAR noise, beam dropout, odometry drift)
2. **Robot Dynamics Randomization** (mass variance, surface friction, motor lag, communication delay)

in bridging the sim-to-real transfer gap for mapless planar navigation?

---

## 2. System Architecture

```
                  +----------------------------------------------+
                  |            TurtleBot3 Burger (Sim)           |
                  |     - LDS-01 LiDAR (360 beams, 5 Hz)         |
                  |     - Wheel Encoders (Odometry @ 30 Hz)      |
                  +----------------------+-----------------------+
                                         |  /scan, /odom
                                         v
                  +----------------------------------------------+
                  |   Observation Preprocessor (Pure NumPy)      |
                  |   - 36 Min-Pooled LiDAR Bins (10 deg, [0, 1])|
                  |   - 1 Goal Distance (Normalized by 5.657 m)  |
                  |   - 2 Goal Bearing [sin(th), cos(th)]        |
                  |   - 2 Measured Velocities (v, omega)         |
                  |   - 2 Previous Action Coordinates            |
                  +----------------------+-----------------------+
                                         |  State s_t (43-D)
                                         v
                  +----------------------------------------------+
                  |             PPO Policy (SB3 MLP)             |
                  |     Actor:  2 x 128 tanh  --> u in [-1, 1]^2 |
                  |     Critic: 2 x 128 tanh  --> V(s)           |
                  +----------------------+-----------------------+
                                         |  Continuous u_t
                                         v
                  +----------------------------------------------+
                  |         Action Mapping & Safety Layer        |
                  |   - cmd_v = 0.075 * (u[0] + 1) in [0, 0.15]  |
                  |   - cmd_omega = 0.60 * u[1]  in [-0.6, 0.6]  |
                  |   - Watchdog: Stale scan (>1.0s) -> Zero Stop|
                  +----------------------+-----------------------+
                                         |  /cmd_vel
                                         v
                  +----------------------------------------------+
                  |           Robot Actuation (Gazebo)           |
                  +----------------------------------------------+
```

### Observation & Action Interfaces
- **Observation Space ($D = 43$):**
  - $obs[0:36]$: 36 min-pooled LiDAR bins ($10^\circ$ each), normalized by $r_{\max} = 3.50\,\text{m}$ into $[0, 1]$. Bin 0 is centered forward $[-5^\circ, +5^\circ)$.
  - $obs[36]$: Normalized goal distance (clipped to $[0, 1]$ by arena diagonal $5.657\,\text{m}$).
  - $obs[37:39]$: Goal bearing $[\sin\beta, \cos\beta] \in [-1, 1]$ relative to robot heading computed via dead-reckoning estimated pose.
  - $obs[39:41]$: Normalized velocities $v / 0.15\,\text{m/s}$ and $\omega / 0.60\,\text{rad/s}$ in $[-1, 1]$.
  - $obs[41:43]$: Previous normalized continuous action $u_{t-1} \in [-1, 1]^2$.
- **Action Space ($A = 2$):**
  - Normalized action $u \in [-1, 1]^2$ mapped to $v = 0.075(u_v + 1)\,\text{m/s} \in [0.0, 0.15]\,\text{m/s}$ and $\omega = 0.60 u_\omega \in [-0.60, 0.60]\,\text{rad/s}$.
  - Strictly non-negative linear velocity (forward motion only). Control rate: $5.0\,\text{Hz}$ ($\Delta t = 0.20\,\text{s}$).
- **Reward Function:**
  $$r_t = 10 \cdot (d_t - d_{t+1}) - 0.05 + 100 \cdot \mathbb{I}_{\text{goal}} - 100 \cdot \mathbb{I}_{\text{collision}}$$
  - Goal arrival radius: $d \le 0.20\,\text{m}$. Collision threshold: $d_{\min} < 0.16\,\text{m}$. Max episode steps: $350$ ($70\,\text{s}$).

---

## 3. Experimental Design: $2 \times 2$ Matrix

| Policy | Policy Name | Sensor Randomization | Dynamics Randomization | Timesteps Trained | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **P1** | **Nominal Baseline** | OFF | OFF | **852,816** | Completed (`runs/main_p1_s0/`) |
| **P2** | **Sensor Only** | ON | OFF | — | Pending |
| **P3** | **Dynamics Only** | OFF | ON | — | Pending |
| **P4** | **Combined Randomization** | ON | ON | **704,512** | Completed (`runs/main_p4_s0/`) |

### Randomization Ranges
- **Sensor Observation Randomization (P2 & P4):**
  - LiDAR range Gaussian noise: $\sigma \in [0.005, 0.03]\,\text{m}$.
  - LiDAR beam dropout probability: $p \in [0.01, 0.08]$ (failed beams map to $r_{\max}$).
  - Odometry scale drift: $s_v, s_\omega \in [0.95, 1.05]$.
- **Robot Dynamics Randomization (P3 & P4):**
  - Robot mass variation: $\Delta m \in [-15\%, +15\%]$.
  - Ground friction coefficient: $\mu \in [0.4, 1.2]$.
  - Motor response lag (low-pass filter): $\tau \in [0.05, 0.25]\,\text{s}$.
  - Control actuation latency: $\delta \in [0.02, 0.12]\,\text{s}$.

---

## 4. Preliminary Results (Mid-Term Audit)

Both completed policies (**P1** and **P4**) were benchmarked on the identical 50 held-out test scenarios ([`scenarios/test/scenarios.json`](scenarios/test/scenarios.json), seeds $2,000,000$–$2,000,049$). Zero overlap exists between train, validation, and test split seeds.

### Held-Out Evaluation Summary ($N = 50$)

| Metric | P1 (Nominal Baseline) | P4 (Combined Randomization) | Empirical Delta |
| :--- | :---: | :---: | :---: |
| **Evaluated Scenarios** | 50 | 50 | — |
| **Goal Success Rate** | **94.0%** (47 / 50) | **98.0%** (49 / 50) | +4.0% |
| **Wilson 95% Confidence Interval** | $[0.838, 0.979]$ | $[0.895, 0.996]$ | — |
| **Collisions** | 1 (2.0%) | 1 (2.0%) | 0.0% |
| **Timeouts (350 step limit)** | 2 (4.0%) | **0 (0.0%)** | -4.0% |
| **Mean Cumulative Reward** | $103.75 \pm 37.71$ | **$109.03 \pm 30.95$** | +5.28 |
| **Mean Time-to-Goal (Successes)** | $21.2\,\text{s}$ ($106.2\,\text{steps}$) | **$20.3\,\text{s}$** ($101.5\,\text{steps}$) | -0.9 s |
| **Mean Path Length (Successes)** | $2.251\,\text{m}$ | **$2.067\,\text{m}$** | -0.184 m (-8.2%) |
| **Mean Min Obstacle Clearance** | $0.282\,\text{m}$ | **$0.292\,\text{m}$** | +0.010 m |

### Paired Statistical Comparison & Limitations
- **Contingency Table:** Both succeed in $46/50$ cases ($92\%$). P1 succeeds while P4 fails in $1$ case (Scenario 34 collision). P4 succeeds while P1 fails in $3$ cases (Scenario 7 collision, Scenarios 25 & 37 timeouts). Both fail in $0$ cases.
- **Exact McNemar Test:** With discordant pairs $b=1, c=3$, the two-sided exact binomial test yields **$p = 0.6250$**.
- **Statistical Interpretation:** While P4 is descriptively superior across reward, path efficiency, and timeout prevention, the $4.0\%$ difference is **not statistically significant** at the $\alpha = 0.05$ level on $N = 50$ test scenarios. Establishing statistical significance will require testing across larger sample sizes ($N \ge 250$).

---

## 5. Repository Structure

```text
sim-real-rl/
├── ros2_ws/
│   └── src/
│       ├── tb3_rl_navigation/   # ROS 2 deploy node, observation preprocessor
│       └── tb3_rl_sim/          # Gazebo world (arena_4x4.world), models, launch
├── rl/
│   ├── env/                     # Gymnasium env, estimated pose, domain randomizers
│   └── train/                   # PPO training loop, callbacks, watchdog supervisor
├── baselines/                   # Hand-coded reactive potential-field controller
├── configs/                     # YAML configuration files (single source of truth)
│   └── ppo.yaml
├── scenarios/                   # Deterministic scenario splits (train, val, test, stress)
│   ├── train/scenarios.json
│   ├── validation/scenarios.json
│   └── test/scenarios.json
├── deploy/                      # Standalone PPO policy inference node
│   └── ppo_inference_node.py
├── scripts/                     # Automation, orchestration, and demo runners
│   ├── orchestrate_p1_to_p4.py
│   ├── run_simulation_demo.py
│   └── verify_gazebo_integration.py
├── tests/                       # Pytest unit and integration test suites
│   ├── test_inference_correctness.py
│   ├── test_observation.py
│   └── test_watchdog.py
├── docs/                        # Specifications, setup logs, and mid-term report data
│   ├── demo_runbook.md
│   ├── midterm_report/
│   │   ├── project_fact_sheet.md
│   │   ├── evidence_map.md
│   │   ├── team_contributions.md
│   │   ├── references.bib
│   │   └── results_summary.json
│   └── Sim2Real_LiDAR_Navigation_Pipeline.pdf
└── runs/                        # Checkpoints, evaluation records, and logs (git-ignored)
    ├── main_p1_s0/final_model.zip
    ├── main_p4_s0/final_model.zip
    └── eval/p1_vs_p4_comparison.json
```

---

## 6. Installation & Quickstart

### Prerequisites
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill (`/opt/ros/humble`)
- Gazebo Classic 11.10.2
- Python 3.10 with PyTorch (CPU-pinned) and Stable-Baselines3 2.7.1

### Build ROS 2 Workspace
```bash
source /opt/ros/humble/setup.bash
cd ~/sim-real-rl/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Run Unit & Inference Parity Tests
```bash
cd ~/sim-real-rl
PYTHONPATH=. pytest -v tests/test_inference_correctness.py
```

---

## 7. Running the Simulation Demo

For full terminal steps and troubleshooting, see the [`docs/demo_runbook.md`](docs/demo_runbook.md).

### Automated 1-Command Demo Runner
Launches Gazebo Classic headless, executes 3 multi-waypoint navigation challenges in the arena, displays real-time console status, and cleanly shuts down:
```bash
# Run Selected Best Policy (P4 - Combined Domain Randomization)
python3 scripts/run_simulation_demo.py --policy P4

# Run Nominal Baseline Policy (P1)
python3 scripts/run_simulation_demo.py --policy P1
```

### Manual Execution with 3D GUI
Open two terminals:

**Terminal 1 (Gazebo Simulation):**
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30
export GAZEBO_MASTER_URI=http://localhost:11345

ros2 launch tb3_rl_sim headless_sim.launch.py &
gzclient
```

**Terminal 2 (Inference Node):**
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

python3 deploy/ppo_inference_node.py \
  --model-path runs/main_p4_s0/final_model.zip \
  --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl \
  --goal-x 1.2 --goal-y 0.0
```

---

## 8. Current Limitations & Future Work

- **Scope:** Current deliverables are **strictly simulation-only** inside Gazebo Classic. No physical hardware transfer claims are made at this mid-term stage.
- **Ablation Study:** Training P2 (sensor-only) and P3 (dynamics-only) models remains future work to complete the full $2 \times 2$ ablation grid.
- **Multi-Seed Variance:** Seeds 1 and 2 will be trained to assess initialization sensitivity.
- **Large-Scale Evaluation:** Expanding test set size ($N \ge 250$) across varying obstacle densities to establish statistical significance.
- **Hardware Validation:** Final transfer of the selected policy to a physical TurtleBot3 Burger in an identical hardware arena.

---

## 9. Contributors

- **Saigirish S** (`B24ME1066`, Department of Mechanical Engineering, IIT Jodhpur)
- **Siddhi Vinayak** (`B24BB1036`, Department of Biosciences and Bioengineering, IIT Jodhpur)

Detailed contribution breakdown and academic metadata are documented in [`docs/midterm_report/team_contributions.md`](docs/midterm_report/team_contributions.md).

---

## 10. References & Citation

See [`docs/midterm_report/references.bib`](docs/midterm_report/references.bib) for complete BibTeX entries.
- Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347, 2017.
- Tobin et al., *"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"*, IEEE/RSJ IROS, 2017.
- Tai et al., *"Virtual-to-Real Deep Reinforcement Learning for Robot Navigation"*, IEEE/RSJ IROS, 2017.
- Raffin et al., *"Stable-Baselines3: Reliable Reinforcement Learning Implementations in Python"*, JMLR 22(268), 2021.
- Macenski et al., *"Robot Operating System 2: Design, Architecture, and Uses in the Wild"*, Science Robotics 7(66), 2022.
