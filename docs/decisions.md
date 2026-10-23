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

---

## 4. Simulation Backend Decision & Verification Requirements

| Decision Area | Specification / Status | Context & Constraints |
| :--- | :--- | :--- |
| **Specification Default** | Gazebo Classic 11.10.2 (`use_gazebo=True`) | Authoritative spec dictates training in Gazebo Classic simulation environment. |
| **Measured Gazebo Throughput** | 1 worker: 4.68 steps/s<br>2 workers: 9.71 steps/s<br>4 workers: 20.28 steps/s | Empirically benchmarked on host (16 cores, 15 GiB RAM). Linear scaling across isolated ports (`GAZEBO_MASTER_URI=11345+i`, `ROS_DOMAIN_ID=30+i`). |
| **Kinematic Backend Throughput** | ~657 steps/s (~130× faster than 1 Gazebo worker) | Pure NumPy 2D raycaster and differential-drive kinematic integration. Single run of 1M steps takes ~25.4 minutes; 12 runs take ~5.07 hours total. |
| **Backend Decision Status** | **Option A Selected by Saigirish; Siddhi Review & Formal Signoff Pending** | **Selected Strategy**: Train P1–P4 directly in Gazebo Classic (Option A).<br>**Kinematic Role**: Restricted strictly to development, reward debugging, and unit tests only.<br>**Weight Transfer Rule**: Do NOT initialize Gazebo policies from kinematic-trained weights.<br>**Outstanding Items**: Siddhi's formal agreement, independent reward/playback review, and formal G2 signoff remain pending. |
| **Gazebo G2 Verification** | **Executed & Verified** (`tests/test_g2_gazebo.py`) | 1,000 consecutive resets passed (0 failures, 204.0s wall-clock, 4.9 resets/s with full physics settle & 5 Hz scan arrival); scan monotonicity passed; scripted wall, goal, timeout passed; soak memory growth: `gzserver` -0.73 MB over 250 steps. |
| **PPO Normalization** | `norm_obs=False`, `norm_reward=True` | Fixed per spec. Observation normalization is handled manually in `rl/env/observation.py`. |
| **Reward Audit & Arithmetic** | **Empirically Verified & Reconciled** | Traces in `results/reward_audit_traces.json` match frozen Section 5 equation exactly. Every single episode return equals the sum of its 4 components to $< 10^{-12}$, and summary means are exact averages of those same episodes: Hand-coded (+101.86 mean = +90.00 goal [18/20 successes] + 17.73 progress - 5.87 step penalty [117.4 mean steps]), Spin (-17.50 = -350×0.05), Stationary (-17.50), Random (-107.65 = -100 collision - 1.38 progress - 6.27 step penalty). Historical values (+62.3, -34.1, -38.2, -39.8) are **strictly unverified** from repository artifacts. Siddhi's independent review remains pending. |
| **Check 8 & Check 11 Status** | **Passed on Kinematic Backend; Gazebo Dynamics Scope Bounded** | Evaluated on kinematic backend (`use_gazebo=False`). Check 8 reactive baseline: 92.0% nominal (92/100). Corner draws across real environment interfaces reconciled: 25-ep Combined sign=+1 (22/25 = 88.0%), Combined sign=-1 (21/25 = 84.0%); 50-ep runs on seeds 1,000,000..1,000,049 are a nested overlapping extension (first 25 identical bit-for-bit, second 25 are 24/25 and 21/25), yielding 46/50 (92.0%) and 42/50 (84.0%) with 88/100 (88.0%) aggregate; independent 50-ep runs on seeds 1,000..1,049 yield 40/50 (80.0%) and 40/50 (80.0%). All configurations exceed 73.6% non-collapse threshold. Check 11 randomizers verified: P1 nominal, P2 sensor-only ($d_{\text{p1}} == d_{\text{p2}}$), P3 dynamics-only ($d_{\text{p1}} \ne d_{\text{p3}}$), P4 latency constraint ($t_{\text{lat}} + d_{\text{del}} \le 0.40$s). Live-Gazebo boundary: command gains, lag, delay, and acceleration limits are plumbed into `/cmd_vel` Twist; sensor perturbations are applied to `/scan` and `/odom`. However, SDF link physics (`mass_multiplier`, `friction_multiplier`) are **unsupported in Gazebo Classic at runtime** and are excluded from verified dynamics coverage. |
| **Check 9 Live Gazebo Soak** | **PASSED & Fully Executed** | Kinematic 100k soak passed (+0.01% RSS). 100,000 continuous live Gazebo steps completed across 4 isolated workers (`results/gazebo_soak_checkpoint.json`): 5,215.3 s elapsed (86.9 mins, 19.77 sps aggregate), 1,018 episodes, gzserver RSS 1,264.3 MB $\to$ 1,213.2 MB (-51.15 MB, -4.05% growth, passes $< 5\%$ limit), Python RSS 61.4 MB $\to$ 63.4 MB, 0 NaNs/Infs, 0 crashes. Baseline 1k checkpoint archived in `results/gazebo_soak_checkpoint_1k_baseline.json`. |
| **Recovery & Watchdog (Check 10)** | **PASSED & Fully Implemented** (`rl/train/watchdog.py`, `tests/test_watchdog.py`) | Implemented 2-second monotonic clock missing-scan watchdog (`watchdog_timeout_sec: 2.0` from `configs/ppo.yaml`). On timeout: suppresses actions (zero Twist), cleanly terminates owned process group via SIGTERM/SIGKILL, restarts simulator, verifies fresh scan arrival, supports checkpoint restoration without duplicate accounting, and bounds retries (max 3, fails safely with `WatchdogRecoveryError`). All 6 automated tests in `tests/test_watchdog.py` passed, including controlled live Gazebo fault injection (SIGSTOP on owned worker PID 289858 detected after 2.0s $\to$ killed $\to$ restarted as PID 290061 $\to$ fresh scan verified $\to$ stepped cleanly; diagnostics saved to `results/watchdog_diagnostics.json`). Non-training SB3 checkpoint save/reload parameter parity (`atol=1e-6`) verified in `tests/test_checkpoint_resume.py`. |
| **Process Ownership Audit** | **Forensic Verification & Process-Group Enforced** | PID 257023 was forensically verified via `/proc/257023/environ` as an orphaned `gzserver` process spawned by this agent task in step 1079 (`ANTIGRAVITY_SOURCE_METADATA` matched task ID). All runner, watchdog, and benchmark scripts now enforce process-group SIGTERM/SIGKILL termination. |

### Measured Gazebo Stepping Throughput & Projected Training Budget (Host: 16-core / 24-thread i7-14650HX, 15 GiB RAM)

| Setup | Measured Aggregate Stepping Rate | Measured Per-Worker Rate | Real-Time Factor (RTF) | Host CPU Utilization (Normalized) | Total `gzserver` RSS | Projected 1M-Step Run Duration | Projected 12 × 1M-Step Total Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Worker (Benchmark)** | 4.68 steps/s | 4.68 steps/s | 0.94× | ~2.5% (0.6 cores) | 317.3 MB | ~59.4 hours (2.48 days) | ~712.5 hours (29.7 days) |
| **2 Workers (Benchmark)** | 9.71 steps/s | 4.86 steps/s | 0.97× | ~6.7% (1.6 cores) | 628.8 MB | ~28.6 hours (1.19 days) | ~343.2 hours (14.3 days) |
| **4 Workers (100k Soak)** | **19.77 steps/s** | **4.94 steps/s** | **0.99×** | **~13.3% (3.2 cores)** | **1,213.2 MB** | **~14.05 hours (0.59 days)** | **~168.6 hours (7.03 days)** |
| *(Kinematic Dev)* | *657.2 steps/s* | *657.2 steps/s* | *131.4×* | *4.2% (1 core)* | *N/A* | *0.42 hours (25.4 mins)* | *5.07 hours* |

*Note on Throughput vs. Training Duration*: The 19.77 steps/s figure represents **empirically measured stepping throughput during the 100k continuous random soak**. The 1M and 12M durations are **projections**, not measurements of active PPO policy training. Actual PPO training includes backpropagation (10 epochs over minibatches of 256 every 2048 steps), periodic evaluation callbacks every 100k steps, and checkpoint serialization, which will yield slightly lower throughput (~18.5–19.5 steps/s), making ~168.6 hours (~7.0 days) an estimated lower bound.


