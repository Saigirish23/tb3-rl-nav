# Sim-to-Real 2D LiDAR Navigation Pipeline (`sim-real-rl`)

A controlled 2×2 sim-to-real reinforcement learning pipeline evaluating sensor observation and dynamics domain randomization for mapless 2D LiDAR navigation on the TurtleBot3 Burger.

Authoritative specification: [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf).

---

## 1. Project Structure

```text
sim-real-rl/
├── ros2_ws/
│   └── src/
│       ├── tb3_rl_navigation/   # Deploy-side package: shared observation, safety layer, deployment node (clean of sim deps)
│       ├── tb3_rl_sim/          # Simulation-only assets: worlds, model overrides, headless launch
│       └── (upstream TurtleBot3 repos only if needed; tracked via upstream.repos)
├── rl/
│   ├── env/                     # Gymnasium env, estimated pose, scan/dynamics models, Gazebo interface, reset manager
│   ├── train/                   # SB3 PPO training loops and callbacks
│   ├── evaluate/                # Evaluation runners across fixed test sets
│   └── analysis/                # Statistical comparison, plotting, table generation
├── baselines/                   # Hand-coded reactive controller baseline
├── configs/                     # YAML configuration files (env, ppo, rand_p2, rand_p3) - single source of truth
├── scenarios/                   # Reproducible scenario definitions
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── stress/
├── deploy/                      # Exported policy files and hardware deployment launch/config notes
├── tests/                       # Unit tests (pure Python, pytest)
├── runs/                        # Per-run outputs: configs, logs, checkpoints, CSVs (git-ignored)
├── docs/                        # Project specifications, setup logs, decision records
│   ├── Sim2Real_LiDAR_Navigation_Pipeline.pdf
│   ├── setup_log.md
│   ├── decisions.md
│   └── environment.md
├── .venv/                       # RL virtual environment with --system-site-packages (git-ignored)
├── .gitignore
└── README.md
```

---

## 2. Core Architecture & Design Decisions

### 2×2 Research Design
* **P1**: Nominal simulation (no domain randomization).
* **P2**: Sensor & observation randomization (LiDAR noise/dropout, odometry drift).
* **P3**: Robot dynamics randomization (mass, friction, motor lag, delay).
* **P4**: Full domain randomization (combined P2 + P3 ranges).

### Key Architectural Rules
* **Algorithm**: Stable-Baselines3 PPO, `MlpPolicy`, continuous actions, separate actor and critic MLPs with 2×128 tanh layers.
* **Observation (43 dims)**: 36 min-pooled 10° LiDAR bins, goal distance, sin/cos of goal bearing, measured $v$ and $\omega$, and previous normalized action (2).
* **Action**: $u \in [-1, 1]^2$, mapped to $v = 0.075(u_v + 1)\,\text{m/s}$ and $\omega = 0.6 u_\omega\,\text{rad/s}$. Zero-order hold synchronized to new LiDAR scans.
* **Shared Observation Module**: Pure NumPy module (`tb3_rl_navigation/observation.py`) with no ROS (`rclpy`) dependencies, shared identically between simulation and deployment.
* **Ground Truth**: Used exclusively for reward calculation, collision detection, and evaluation metrics. Never exposed in observations.

---

## 3. Environment & Prerequisites

* **OS**: Ubuntu 22.04 LTS (Jammy)
* **ROS**: ROS 2 Humble (`/opt/ros/humble`)
* **Simulator**: Gazebo Classic 11 (`gazebo_ros_pkgs`)
* **Robot**: TurtleBot3 Burger (LDS-01 or LDS-02)
* **Python**: Python 3.10 in `.venv` created with `--system-site-packages` (so `rclpy` is accessible), pinned to CPU-only PyTorch to minimize disk footprint.

For step-by-step setup, teammate onboarding, and operational records, see [`docs/handover.md`](docs/handover.md), [`docs/setup_log.md`](docs/setup_log.md), and [`docs/decisions.md`](docs/decisions.md).

