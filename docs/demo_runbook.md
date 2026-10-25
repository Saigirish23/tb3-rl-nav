# TurtleBot3 PPO Simulation Demo Runbook

This runbook documents the step-by-step procedures for launching, running, and verifying the autonomous PPO LiDAR navigation demonstration in Gazebo Classic 11 on ROS 2 Humble.

---

## 1. Prerequisites & Environment Setup

- **OS:** Ubuntu 22.04 LTS
- **ROS 2:** Humble Hawksbill (`/opt/ros/humble`)
- **Simulator:** Gazebo Classic 11 (`gzserver`, `gzclient`)
- **Robot Model:** TurtleBot3 Burger (`TURTLEBOT3_MODEL=burger`)
- **Workspace:** `~/sim-real-rl`
- **Python Environment:** System Python 3.10 / `.venv` (with `torch` CPU and `stable_baselines3`)

### Environment Initialization
In each terminal opened, initialize the ROS 2 and workspace overlays:
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30
export GAZEBO_MASTER_URI=http://localhost:11345
export PYTHONPATH=~/sim-real-rl:$PYTHONPATH
```

---

## 2. Quick Demo: Automated One-Command Demo Runner

The simplest and most reliable way to execute the demonstration is via the dedicated demo runner script [`scripts/run_simulation_demo.py`](file:///home/saigirish/sim-real-rl/scripts/run_simulation_demo.py). It automatically handles Gazebo Classic lifecycle, sensor verification, multi-waypoint scenario execution, live terminal status display, and clean shutdown.

### Run Primary Policy (P4 — Combined Domain Randomization)
```bash
cd ~/sim-real-rl
python3 scripts/run_simulation_demo.py --policy P4
```

### Run Nominal Baseline Policy (P1)
```bash
cd ~/sim-real-rl
python3 scripts/run_simulation_demo.py --policy P1
```

### Automated Run Options
- `--port 11345`: Specify custom Gazebo port (default `11345`).
- `--domain-id 30`: Specify custom ROS domain ID (default `30`).
- `--connect-existing`: Connects to an already-running Gazebo instance instead of starting a new simulator.
- Results are saved automatically to a new timestamped directory under `runs/demo_run_<timestamp>/`.

---

## 3. Terminal-by-Terminal Manual Launch (For Visual Inspection)

If you want to view the 3D robot moving in the Gazebo GUI or inspect topics interactively:

### Terminal 1: Launch Gazebo Simulation World & Robot
```bash
source /opt/ros/humble/setup.bash
source ~/sim-real-rl/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30
export GAZEBO_MASTER_URI=http://localhost:11345

# Headless server
ros2 launch tb3_rl_sim headless_sim.launch.py
```
*(Optional: In another terminal, run `gzclient` with `GAZEBO_MASTER_URI=http://localhost:11345` to open the 3D visualization window).*

### Terminal 2: Sensor Health & Topic Sanity Check
Verify that the simulated sensors and robot state publisher are publishing at expected rates:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

# 1. Verify active topics
ros2 topic list

# Expected topics:
#   /scan
#   /odom
#   /cmd_vel
#   /tf

# 2. Check LiDAR rate (~5 Hz)
ros2 topic hz /scan

# 3. Check Odometry rate (~30 Hz)
ros2 topic hz /odom

# 4. Check TF transform tree
ros2 run tf2_ros tf2_echo odom base_footprint
```

### Terminal 3: Launch Standalone PPO Inference Node
Deploy the trained policy to navigate to the target waypoint:

#### To Run Selected Primary Policy (P4):
```bash
cd ~/sim-real-rl
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

# Waypoint 1: Straight Forward (1.2m East)
python3 deploy/ppo_inference_node.py \
  --model-path runs/main_p4_s0/final_model.zip \
  --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl \
  --goal-x 1.2 --goal-y 0.0

# Waypoint 2: Diagonal Positive (1.0m, 1.0m North-East)
python3 deploy/ppo_inference_node.py \
  --model-path runs/main_p4_s0/final_model.zip \
  --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl \
  --goal-x 1.0 --goal-y 1.0

# Waypoint 3: Left Quadrant (-1.0m, 1.0m North-West)
python3 deploy/ppo_inference_node.py \
  --model-path runs/main_p4_s0/final_model.zip \
  --vecnorm-path runs/main_p4_s0/final_vecnormalize.pkl \
  --goal-x -1.0 --goal-y 1.0
```

#### To Switch to Nominal Policy (P1):
Simply point `--model-path` and `--vecnorm-path` to `runs/main_p1_s0/`:
```bash
python3 deploy/ppo_inference_node.py \
  --model-path runs/main_p1_s0/final_model.zip \
  --vecnorm-path runs/main_p1_s0/final_vecnormalize.pkl \
  --goal-x 1.2 --goal-y 0.0
```

---

## 4. Expected Demo Outputs

### Live Console Status Stream
During execution, `ppo_inference_node.py` outputs real-time navigation telemetry at 1-second intervals:
```text
[PPOInference] Successfully loaded PPO model: runs/main_p4_s0/final_model.zip (timesteps=704,512)
[PPOInference] Successfully loaded VecNormalize stats (count=704512.0001000001)
[PPOInference] Waiting for sensor topics (/scan, /odom)...
[PPOInference] Sensor topics received. Beginning navigation.
[PPOInference] Reset trial: Start=(0.00, -0.00, 0.00 rad) -> Goal=(1.20, 0.00)
[PPOInference] Step   5 ( 0.8s) | Pose: ( 0.07,  0.00,  0.01 rad) | Goal Dist:  1.13m | Min LiDAR:  1.99m | Cmd: (v=0.12, w=-0.03)
[PPOInference] Step  10 ( 1.8s) | Pose: ( 0.19,  0.00, -0.01 rad) | Goal Dist:  1.01m | Min LiDAR:  1.90m | Cmd: (v=0.12, w=-0.02)
...
[PPOInference] GOAL REACHED! (dist=0.198 m <= 0.200 m in 57 steps)
[PPOInference] Trial summary saved to: runs/demo_run_<timestamp>/inference_trial_<epoch>.json
```

### Acceptance Benchmark Targets

| Metric | Acceptance Criteria | Verified Performance |
| :--- | :---: | :---: |
| **Arrival Tolerance** | Distance $\le 0.20\text{ m}$ | Reached ($0.191\text{ m}$ – $0.198\text{ m}$) |
| **Safety Clearance** | Min range $\ge 0.16\text{ m}$ | Maintained ($> 0.89\text{ m}$) |
| **Collision Rate** | 0% | 0 collisions |
| **Timeout Rate** | 0% ($< 350\text{ steps}$) | 0 timeouts ($57$ to $88\text{ steps}$) |
| **Control Frequency** | $5.0\text{ Hz}$ zero-order hold | Enforced strictly at $0.20\text{ s}$ |

---

## 5. Troubleshooting & Clean Shutdown

### Port Conflicts or Lingering Simulators
If Gazebo fails to launch due to an occupied port or lingering background process:
```bash
# Check if port 11345 is occupied
lsof -i :11345

# Kill any leftover Gazebo Classic instances cleanly
pkill -9 gzserver 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true
pkill -9 -f "tb3_rl_sim" 2>/dev/null || true
```

### Stale Sensor Watchdog Alert
If the robot halts and outputs `WARNING: Scan data stale (> 1.0s)! Halting robot for safety`:
- Verify gzserver is running: `pgrep gzserver`
- Confirm `/scan` is receiving messages: `ros2 topic echo /scan --once`
- Ensure matching `ROS_DOMAIN_ID` is exported in all shells.

---

## 6. Verification Tests

To verify inference parity, observation layout, and action scaling before presenting:
```bash
cd ~/sim-real-rl
PYTHONPATH=. pytest -v tests/test_inference_correctness.py
```
*(Expected: 4 passed).*
