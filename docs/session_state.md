# Session Snapshot: `sim-real-rl` Project Setup

* **Date**: 2026-10-23
* **Current State**: **Option A selected by Saigirish (direct Gazebo training for P1–P4). Kinematic backend restricted to dev/unit tests only (no weight transfer). Siddhi's agreement, independent review, and formal G2 signoff pending.**
* **P1 Training Status**: **STRICTLY ON HOLD** — Do not launch P1 or any long training run until formal co-owner review and signoff.

---

## 1. Project & Workspace Status

* **Repository Root**: `/home/saigirish/sim-real-rl`
* **Specification**: Frozen per [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](file:///home/saigirish/sim-real-rl/docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf).
* **Tracked Decisions**: Recorded in [`docs/decisions.md`](file:///home/saigirish/sim-real-rl/docs/decisions.md) (Section 4 Backend Decision pending).
* **Setup History**: Chronologically logged in [`docs/setup_log.md`](file:///home/saigirish/sim-real-rl/docs/setup_log.md).
* **Environment Spec & Scan Audit**: Pinned in [`configs/env.yaml`](file:///home/saigirish/sim-real-rl/configs/env.yaml) and [`docs/environment.md`](file:///home/saigirish/sim-real-rl/docs/environment.md).

---

## 2. Parallel Gazebo Throughput Benchmark Results

Controlled multi-worker benchmarks run on host (Intel Core i7-14650HX, 16 cores / 24 threads, 15 GiB RAM) using isolated Gazebo master ports (`GAZEBO_MASTER_URI=http://localhost:11345+i`) and DDS domains (`ROS_DOMAIN_ID=30+i`):

| Workers | Aggregate Throughput | Per-Worker Rate | RTF | Host CPU % (Normalized) | System RAM RSS | Total `gzserver` RSS | Est. 1M Steps | Est. 12 × 1M Steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Worker** | **4.68 steps/s** | 4.68 steps/s | 0.94× | ~2.5% (0.6 cores) | 4,534 MB | 317.3 MB (+0.21 MB) | 59.4 hrs (2.48 d) | **712.5 hrs (29.7 days)** |
| **2 Workers** | **9.71 steps/s** | 4.86 steps/s | 0.97× | ~6.7% (1.6 cores) | 4,685 MB | 628.8 MB (+0.39 MB) | 28.6 hrs (1.19 d) | **343.2 hrs (14.3 days)** |
| **4 Workers** | **20.28 steps/s** | 5.07 steps/s | 1.01× | ~13.3% (3.2 cores) | 4,775 MB | 1,272.7 MB (+0.61 MB) | 13.7 hrs (0.57 d) | **164.3 hrs (6.85 days)** |
| *(Kinematic)* | *657.2 steps/s* | *657.2 steps/s* | *131.4×* | *4.2% (1 core)* | *744 MB* | *N/A* | *0.42 hrs (25.4 m)* | *5.07 hours* |

*Note: No cross-worker interference observed. Memory growth remained negligible (< 1 MB per worker over 50 steps). CPU utilization measured continuously during active stepping.*

---

## 3. Gazebo-Specific G2 Verification (`tests/test_g2_gazebo.py`)

Executed against live headless Gazebo Classic 11.10.2:
1. **1,000 Consecutive Resets (`TestGazebo1000Resets`)**:
   - Fixed ODE penetration repulsion and stale-scan caching: pause physics, teleport obstacles first, place robot at $z=0.012$, unpause physics, and explicitly wait for strictly newer scan post-teleport.
   - Result: **0 failures across 1,000 resets** (204.0s wall-clock, 4.9 resets/s). All poses within 2 cm and 1°, residual velocity zeroed (< 0.05 m/s), valid goal geometry ($1.5 \le d \le 3.5$), zero obstacle overlap. **PASSED**.
2. **Strictly Newer Scan Timing (`TestGazeboScanTiming`)**:
   - 50 consecutive steps verified strictly monotonic scan timestamps ($t_{i} - t_{i-1} \in [0.10, 0.45]\,\text{s}$). Synchronized step loop waits for physical scan publication. **PASSED**.
3. **Scripted Terminations (`TestGazeboTerminationSemantics`)**:
   - Wall collision: Driven forward into arena wall $\to$ `terminated=True, is_collision=True, reward <= -90`. **PASSED**.
   - Goal success: Placed 0.10 m from goal $\to$ `terminated=True, is_success=True, reward >= +90`. **PASSED**.
   - Timeout: Stationary for 350 steps $\to$ `truncated=True, terminated=False, is_collision=False, is_success=False`. **PASSED**.
4. **Live Gazebo Soak & Stability (`TestGazeboSoakAndMemory`)**:
   - 250 live steps: 4.88 steps/s, `gzserver` RSS 319.4 MB $\to$ 318.6 MB (-0.73 MB), Python RSS 747.0 MB $\to$ 747.2 MB (+0.15 MB), NaN/Inf count: 0. **PASSED**.

---

## 4. Controller & Scenario Audit Findings

1. **Observation & Ground-Truth Boundary**:
   - `baselines/reactive_controller.py` strictly accesses `obs[0:36]` (LiDAR bins), `obs[37]` ($\sin\theta$), and `obs[38]` ($\cos\theta$). It has zero access to ground truth (`robot_x`, `robot_y`, `goal_x`, etc.).
2. **Test Isolation**:
   - Benchmark used on-the-fly random seeds `1_000_000 + i`, which are completely disjoint from the held-out test scenarios in `scenarios/test/` (seeds 2,000,000–2,000,099). Zero data leakage.
3. **Controller State Reset**:
   - `controller.reset()` explicitly resets `stuck_steps = 0`, `escape_steps = 0`, and `escape_dir = 1.0` between episodes.
4. **Benchmark Metrics & Confidence Intervals (Kinematic Backend)**:
   - Nominal scenarios: 92/100 successes (92.0%, Wilson 95% CI: **[85.0%, 95.9%]**).
   - Randomization corners (evaluated strictly on kinematic backend, sampling protocol reconciled):
     - 25-episode runs (seeds 1,000,000..1,000,024): Combined sign=+1 (22/25 = **88.0%**), Combined sign=-1 (21/25 = **84.0%**), Sensor sign=+1 (19/25 = **76.0%**), Sensor sign=-1 (21/25 = **84.0%**).
     - 50-episode runs (seeds 1,000,000..1,000,049): Nested overlapping extension (first 25 identical bit-for-bit; second 25 are 24/25 = 96.0% and 21/25 = 84.0%), yielding Combined sign=+1 (46/50 = **92.0%**) and Combined sign=-1 (42/50 = **84.0%**), aggregate 88/100 = **88.0%**.
     - Separate runs on seeds 1,000..1,049: Combined sign=+1 (40/50 = **80.0%**), Combined sign=-1 (40/50 = **80.0%**); earlier uncommitted run: 40/50 (80.0%) and 39/50 (78.0%).
     - All tested configurations comfortably exceed the $80\% \times 92\% = 73.6\%$ non-collapse threshold. Samples are nested/separate and must NOT be combined as 75 episodes.
   - Clarification: The "84%" figure in earlier logs stemmed from both an initial 25-episode upper-corner draw run and the pytest test completion progress indicator `test_goal_reached_terminated PASSED [ 84%]`.
5. **Structural Stress Scenarios**:
   - Implemented `generate_corridor_scenario`, `generate_narrow_gap_scenario` (0.5 m gap), and `generate_dead_end_scenario` in `scenarios/scenario_generator.py`.
   - Regenerated validation (50), test (100), stress (10), and train (10) layouts. Determinism and disjoint seed checks passed.
6. **Reward Audit Traces & Mathematical Proof**:
   - Empirical traces generated and saved to `results/reward_audit_traces.json`. Confirmed that every single episode return equals the sum of its 4 components to $< 10^{-12}$, and summary means are exact averages of those same episodes:
     - Hand-Coded: $+101.86 \pm 37.95$ (90% success, 0% collision, 10% timeout). Exact arithmetic breakdown:
       - Goal reward: $\frac{18 \times 100 + 2 \times 0}{20} = \mathbf{+90.00}$
       - Progress reward: $\mathbf{+17.73}$
       - Step penalty: $117.4 \text{ mean steps} \times (-0.05) = \mathbf{-5.87}$
       - Collision penalty: $\mathbf{0.00}$
       - Sum: $90.00 + 17.73 - 5.87 = \mathbf{+101.86}$ (matches recorded mean to $1.4 \times 10^{-14}$).
     - Spinning: $-17.50 \pm 0.00$ ($350 \times -0.05 = -17.50$, 100% timeout)
     - Stationary: $-17.50 \pm 0.00$ ($350 \times -0.05 = -17.50$, 100% timeout)
     - Random: $-107.65 \pm 8.84$ ($-100.00$ collision + $-1.38$ progress + $-6.27$ step penalty)
   - Historical values (+62.3, -34.1, -38.2, -39.8): **Strictly unverified** from repository artifacts (no code, logs, or git commits contain these figures; they appeared only in external prompt text). Under frozen Section 5 formulas ($k_{\text{progress}}=10, c_{\text{step}}=0.05, r_{\text{goal}}=100, r_{\text{col}}=100, \text{max\_steps}=350$), the empirical values match mathematical truth. Siddhi's independent review remains pending.
7. **Forensic Process Review**:
   - PID 257023 was forensically confirmed via `/proc/257023/environ` JSON metadata as an orphaned `gzserver` process spawned by this agent task in step 1079. Indiscriminate `kill -9` was removed; all runners now use process-group signals (`os.killpg(..., signal.SIGTERM)`).

---

## 5. Current Gate Status & Next Steps

* **Gate G2 Status**: **INCOMPLETE — P1 ON HOLD** — Checks 1–7, 9–10 passed; Check 8 passed on kinematic; co-owner review and backend signoff remain required.
* **Separately Executed Test Suites**:
  - **Full Unit & Integration Suite (Kinematic + Watchdog)**: **46 passed, 1 skipped, 2 deselected** (`tests/test_observation.py` [7], `tests/test_env.py` [8], `tests/test_randomizers.py` [7], `tests/test_checkpoint_resume.py` [1], `tests/test_watchdog.py` [5 passed, 1 live-skipped during non-sim batch], `tests/test_g2_gate.py` [18]) in 75.29s.
  - **Targeted Soak Runner Logic Suite**: **5 passed, 0 failed** (`pytest tests/test_soak_runner_logic.py -v`: atomic checkpointing, corrupt load, resume target override, delta math, linear elapsed time) in 0.14s.
  - **Watchdog Live Gazebo Test Suite**: **6 passed, 0 failed** (`pytest tests/test_watchdog.py -v -s` including controlled live Gazebo fault injection PID 290515 -> 290721) in 11.93s.
  - **Live Gazebo G2 Suite**: **6 passed, 0 failed** (`python3 tests/run_gazebo_g2.py` / `tests/test_g2_gazebo.py`: 1,000 resets [203.7s, 0 failures], scan monotonicity, wall collision, goal success, timeout truncation, 250-step soak) in 268.85s.
  - **Live Gazebo 4-Worker 100k Soak**: **100,000 steps completed across 4 workers** (`python3 tests/run_gazebo_soak.py --steps 100000 --workers 4 --resume`: 5,215.3s / 86.9 mins, 19.77 sps aggregate, 1,018 episodes, gzserver RSS 1,264.3 $\to$ 1,213.2 MB [-4.05%], 0 NaNs, 0 crashes).
  - **Live Gazebo Kill-and-Resume Soak Runner**: **1 passed** (`tests/run_gazebo_soak.py --test-kill-resume`: 300 steps, interrupted, saved, resumed to 600 steps, 0 NaNs, 0 crashes) in 325.7s.
* **Check 8 Status**: **PASSED on Kinematic Backend** (92.0% nominal, 76.0%–92.0% across corner configurations; live-Gazebo corner runs not performed).
* **Check 9 Status**:
  - Kinematic Soak: **PASSED** (100k steps, +0.01% RSS, 0 NaNs).
  - Gazebo Soak: **PASSED** (100,000 continuous live Gazebo steps across 4 workers: 19.77 sps aggregate, 5,215.3s, 1,018 episodes, gzserver RSS -4.05% growth [passes $< 5\%$], 0 NaNs, 0 crashes). Baseline 1k archived in `results/gazebo_soak_checkpoint_1k_baseline.json`.
* **Check 10 & Recovery Status**: **PASSED** (`tests/test_checkpoint_resume.py` verified parameter/action parity with `atol=1e-6`; `rl/train/watchdog.py` and `tests/test_watchdog.py` verified 2s missing-scan detection, bounded retries, and live Gazebo fault recovery).
* **Check 11 Status**: **PASSED on Kinematic Backend; Gazebo Dynamics Scope Bounded** (Factor isolation verified through real `reset()` and `step()` interfaces: P1 nominal, P2 sensor-only, P3 dynamics-only, P4 combined with $t_{\text{lat}} + d_{\text{del}} \le 0.40$s; live-Gazebo physics link properties `mass_multiplier` and `friction_multiplier` are unsupported in Gazebo Classic at runtime and excluded from verified dynamics coverage).
* **P1 Training Status**: **STRICTLY ON HOLD** pending Siddhi's agreement on Option A (Gazebo training), independent reward review, and formal G2 co-owner signoff.
* **Co-Owner Action**: Siddhi to review reward/termination traces (`results/reward_audit_traces.json`), observe Gazebo playback episodes, and co-sign Gate G2 closure. Both co-owners (Saigirish & Siddhi) must formally sign off before P1 training can commence.


