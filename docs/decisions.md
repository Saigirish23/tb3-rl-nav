# Architecture and Design Decisions: `sim-real-rl`

All fixed design decisions from the specification, provisional parameters, and local environment decisions are tracked here.

---

## 1. Fixed Design Decisions (Specification Frozen)

| Decision Area | Specification | Status | Rationale / Constraint |
| :--- | :--- | :--- | :--- |
| **RL Algorithm** | Stable-Baselines3 PPO, `MlpPolicy`, continuous actions | **Fixed** | Controlled 2×2 study; not comparing RL algorithms. |
| **Network Architecture** | Separate actor & critic MLPs, 2 hidden layers of 128 units each, `tanh` activation | **Fixed** | Pinned architecture across all four policies (P1-P4). |
| **Observation Space (43)** | 36 min-pooled 10° LiDAR bins, goal distance, $\sin$ & $\cos$ of goal bearing, measured $v$ & $\omega$, last normalized action (2) | **Fixed** | Compact, invariant, standardized across sim and real. |
| **Action Mapping** | $u \in [-1, 1]^2 \to v = 0.075(u_v + 1)\,\text{m/s} \in [0, 0.15]$, $\omega = 0.6 u_\omega\,\text{rad/s} \in [-0.6, 0.6]$ | **Fixed** | Forward-only bias avoids oscillation; matches TurtleBot3 Burger limits. |
| **Control Timing** | Zero-order hold (ZOH) per new LiDAR scan; control period = scan period (5 Hz / 0.2 s) | **Fixed** | Step-rate lockstep with physical sensor arrival. |
| **Shared Observation Code** | Single pure NumPy module (`tb3_rl_navigation/observation.py`) with zero ROS dependencies (`no rclpy`) | **Fixed** | Eliminates sim-to-real code divergence between Gymnasium env and Pi deployment node. |
| **Ground Truth Boundary** | Ground truth used *only* for reward, collision, and evaluation; never in observation | **Fixed** | Strictly preserves partial observability. |
| **Episode Terminations** | Goal & collision $\to$ `terminated`; Timeout $\to$ `truncated` | **Fixed** | Standard Gymnasium semantics; SB3 bootstraps values on truncation only. |
| **Collision Detection** | Ground-truth footprint collision geometry (URDF bounding geometry). Never LiDAR min range. | **Fixed** | LiDAR min range is susceptible to noise, dropout, and blind spots. |
| **Reward Function** | $r_t = 10(d_t - d_{t+1}) - 0.05 + 100 \cdot [\text{goal}] - 100 \cdot [\text{col}] - r_{\text{clear}} - r_{\text{smooth}}$ | **Fixed** | Progress reward uses ground-truth distance for all policies. Clearance & smoothness frozen after P1 pilot. |
| **Scan Distortion** | Per-beam motion distortion in shared simulator wrapper for all policies | **Fixed** | Physical sensor fidelity model (nominal fidelity, not randomization). |
| **Checkpoint Policy** | Final checkpoint of every run; validation set used for convergence monitoring only | **Fixed** | Avoids test-set bias / checkpoint cherry-picking. |
| **2×2 Research Design** | P1 (nominal), P2 (sensor/obs rand), P3 (dynamics rand), P4 (both) | **Fixed** | Identical architecture, reward, budget, and scenario distribution. Wheel radius excluded from P3 (modeled as odom scale error in P2). |
| **Safety Layer** | Independent node active in SITL and hardware; **inactive** during training | **Fixed** | Policy must learn autonomous obstacle avoidance. |

---

## 2. Provisional Values (Requires Explicit Approval to Modify)

| Parameter | Provisional Value | Evaluation / Verification Plan |
| :--- | :--- | :--- |
| **Arena Dimensions** | 4.0 m × 4.0 m square | Match physical test arena once available. |
| **Goal Tolerance** | 0.20 m ground-truth distance | Recheck during P1 pilot convergence. |
| **Episode Timeout** | 350 steps (70 s at 0.2 s control rate) | Compare against hand-coded reactive controller. |
| **P2 Randomization Ranges** | See `configs/rand_p2.yaml` | LiDAR dropout, gaussian noise, odom scale/bias. |
| **P3 Randomization Ranges** | See `configs/rand_p3.yaml` | Mass, friction, motor time constant, control delay. |
| **Evaluation Set Sizes** | 50 scenarios per evaluation set | Balance statistical power with compute time. |

---

## 3. Host Environment & Infrastructure Decisions

| Decision | Rationale | Status |
| :--- | :--- | :--- |
| **Python venv with `--system-site-packages`** | Allows `.venv` to import `rclpy` from ROS 2 Humble without breaking global system packages. | **Adopted** |
| **CPU-only PyTorch Wheel** | Machine has 11 GiB available disk space (< 15 GiB threshold). PyTorch CPU wheel saves ~2-4 GB vs CUDA wheel; spec dictates CPU training. | **Adopted** |
| **Gazebo Classic 11 Pin** | Spec requires Gazebo Classic with `gazebo_ros_init` and `gazebo_ros_state`. Modern Gazebo / `ros_gz` is prohibited. | **Adopted** |
| **Upstream Repos Isolation** | Upstream packages will not be installed or cloned into `ros2_ws/src` without explicit tracking in `upstream.repos` to avoid shadowing. | **Adopted** |
