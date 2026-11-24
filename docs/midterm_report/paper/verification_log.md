# Complete IEEE Midterm Report Evidence Verification Log

**Audited Repository:** `https://github.com/Saigirish23/tb3-rl-nav.git`  
**Local Working Directory:** `/home/saigirish/sim-real-rl`  
**Audited Base Branch:** `midterm-docs-audit`  
**Audited Commit Hash:** `9bf431db4f664bdc67a16137dd075e2bdff31f0d`  
**Working Branch Created for Audit:** `midterm-report-draft`  
**Audit Date:** October 25, 2026 (Local Time: ~21:10 IST)  

---

## 1. Files Inspected & Availability Audit

| Target File / Artifact | Location in Repo | Inspected Status | SHA-256 / Checksum / Commit Reference |
| :--- | :--- | :---: | :--- |
| `README.md` | `README.md` | Present & Inspected | Audited at `9bf431d` |
| `demo_runbook.md` | `docs/demo_runbook.md` | Present & Inspected | Audited at `9bf431d` |
| `project_fact_sheet.md` | `docs/midterm_report/project_fact_sheet.md` | Present & Inspected | Audited at `9bf431d` |
| `evidence_map.md` | `docs/midterm_report/evidence_map.md` | Present & Inspected | Audited at `9bf431d` |
| `team_contributions.md` | `docs/midterm_report/team_contributions.md` | Present & Inspected | Audited at `9bf431d` |
| `references.bib` | `docs/midterm_report/references.bib` | Present & Inspected | Audited at `9bf431d` |
| `results_summary.json` | `docs/midterm_report/results_summary.json` | Present & Inspected | Audited at `9bf431d` |
| Observation Module | `rl/env/observation.py` | Present & Inspected | Pure NumPy, 43-D, `EstimatedPose` |
| Environment Module | `rl/env/tb3_navigation_env.py` | Present & Inspected | Gymnasium Env, 5 Hz control rate |
| Randomizers Module | `rl/env/randomizers.py` | Present & Inspected | Sensor, Dynamics, Combined classes |
| PPO Training Script | `rl/train/train_ppo.py` | Present & Inspected | SB3 PPO wrapper, multi-worker setup |
| Inference Deployment | `deploy/ppo_inference_node.py` | Present & Inspected | ROS 2 rclpy node, deterministic inference |
| Training Config (PPO) | `configs/ppo.yaml` | Present & Inspected | Hyperparameters, policy net arch |
| Environment Config | `configs/env.yaml` | Present & Inspected | Gazebo Classic 11.10.2, LDS-01 sensor |
| Test Scenarios | `scenarios/test/scenarios.json` | Present & Inspected | `d9a6a25cef080fa05a19627b64d8da72ea0c9f201c149733f604eb9195b0b4bf` |
| P1 Nominal Eval | `runs/eval/p1_test/eval_P1_nominal.json` | Present & Inspected | `3a9a9282c270fc869c6d1857db5d2313e6ac1f5c1cf4caefefb9e0ca806e8d50` |
| P4 Nominal Eval | `runs/eval/p4_test/eval_P4_nominal.json` | Present & Inspected | `5a1f30de3f1114744012cb5daaa236471b46f90dd01a5b923a1b22e97b407a8e` |
| P1 vs P4 Eval Comparison | `runs/eval/p1_vs_p4_comparison.json` | Present & Inspected | `2e286491b06a463c317f577d97e45a781d0de5b67cebac9932b95550d74423af` |
| P1 Checkpoint & VecNorm | `runs/main_p1_s0/final_model.zip`, `final_vecnormalize.pkl` | Present & Inspected | Model: `b693de19...`, VecNorm: `5cffbd65...` |
| P4 Checkpoint & VecNorm | `runs/main_p4_s0/final_model.zip`, `final_vecnormalize.pkl` | Present & Inspected | Model: `d8842fd3...`, VecNorm: `1b3a346e...` |
| Live Gazebo Demo Summary | `runs/demo_run_20261025_182130/demo_summary.json` | Present & Inspected | `a76618496274fe3729ec8cb6214664ae8ce10a40b454fa1989237b603ccf85ae` |
| Reward Audit Script & Data | `tests/generate_reward_audit.py`, `results/reward_audit_traces.json` | Present & Inspected | Tracked in commit `5294e6b` by `siddhivn` |

---

## 2. Claim-by-Claim Verification Table

| ID | Claim | Reported Value or Wording | Evidence Inspected | Exact Path and Line / JSON Key / Command | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **C01** | Simulator and Robot Platform | TurtleBot3 Burger in Gazebo Classic 11.10.2 under ROS 2 Humble | `configs/env.yaml`, `rl/env/tb3_navigation_env.py` | `configs/env.yaml:3-4`, `rl/env/tb3_navigation_env.py:270-305` | **verified** | Confirmed ROS 2 Humble packages and Gazebo Classic 11 interface. |
| **C02** | Observation Vector Dimension | 43 continuous dimensions | `rl/env/observation.py`, `rl/env/tb3_navigation_env.py`, `deploy/ppo_inference_node.py` | `rl/env/observation.py:29, 314-322`, `rl/env/tb3_navigation_env.py:200-210`, `deploy/ppo_inference_node.py:148` | **verified** | Layout: 36 LiDAR bins, 1 goal dist, 2 goal bearing (sin/cos), 1 linear vel, 1 angular vel, 2 previous action. No $c_{\text{flag}}$ in vector. |
| **C03** | Observation Layout Inconsistency ($c_{\text{flag}}$) | Prior summary claimed 43 dimensions including a collision flag $c_{\text{flag}}$ | `rl/env/observation.py` | `rl/env/observation.py:314-322` | **contradicted** | Vector is 43-D, but does NOT contain $c_{\text{flag}}$. The 43 dimensions are: 36 bins + 1 dist + 2 bearing + 1 linear vel + 1 angular vel + 2 previous action $= 43$. (Earlier claim that 36+1+2+1+1+2=42 was an arithmetic error in earlier summary; $36+7=43$). |
| **C04** | Action Space Dimensionality | Continuous 2-D action in $[-1, 1]$ | `rl/env/tb3_navigation_env.py`, `configs/ppo.yaml` | `rl/env/tb3_navigation_env.py:188-195` | **verified** | Continuous Box space of shape (2,) in $[-1.0, 1.0]$. |
| **C05** | Velocity Scaling Discrepancy | Earlier proposal stated $v \in [0, 0.15]\,\text{m/s}, \omega \in [-0.6, 0.6]\,\text{rad/s}$, while some audit summaries reported $v \in [0, 0.22]\,\text{m/s}, \omega \in [-2.0, 2.0]\,\text{rad/s}$ | `rl/env/tb3_navigation_env.py`, `deploy/ppo_inference_node.py` | `rl/env/tb3_navigation_env.py:488-490`, `deploy/ppo_inference_node.py:196-197` | **contradicted** (Audit summaries contradicted; code verified) | Code implements: $v = 0.075 \cdot (u_0 + 1) \in [0.0, 0.15]\,\text{m/s}$ and $\omega = 0.60 \cdot u_1 \in [-0.60, 0.60]\,\text{rad/s}$. The $[0, 0.22]$ and $[-2.0, 2.0]$ figures in earlier summaries were erroneous TB3 hardware specification limits, NOT the active environment mapping! |
| **C06** | Control and Sensor Frequency | 5.0 Hz control loop ($dt = 0.20\,\text{s}$), zero-order hold | `rl/env/tb3_navigation_env.py`, `configs/env.yaml` | `rl/env/tb3_navigation_env.py:166, 475`, `configs/env.yaml:12` | **verified** | Control rate is exactly 5 Hz ($dt = 0.2\,\text{s}$), matching LDS-01 scan rate (5 Hz). |
| **C07** | Mapless / No-SLAM Navigation | Policy acts directly from LiDAR + odometry dead-reckoning; SLAM and Nav2 are absent | `rl/env/observation.py`, `rl/env/tb3_navigation_env.py`, `deploy/ppo_inference_node.py` | `rl/env/observation.py:173-228`, `deploy/ppo_inference_node.py:249-251` | **verified** | No map, SLAM, global costmap, or Nav2 planner in the action loop. Goal is tracked in the reset odometry frame via `EstimatedPose`. |
| **C08** | PPO Policy Network Architecture | MLP `[128, 128]` for actor ($\pi$) and critic ($vf$) with `tanh` activation | `configs/ppo.yaml`, `runs/main_p1_s0/run_meta.json`, `runs/main_p4_s0/run_meta.json` | `configs/ppo.yaml:11-16`, `runs/main_p1_s0/run_meta.json:14-25` | **contradicted** (Prior summary claimed `[64, 64]`; actual config is `[128, 128]`) | Actual effective architecture logged during training is two hidden layers of 128 units (`pi: [128, 128]`, `vf: [128, 128]`) with `tanh` activations. |
| **C09** | PPO Hyperparameters | $\text{lr}=3\times 10^{-4}, n_{\text{steps}}=2048, \text{batch\_size}=256, n_{\text{epochs}}=10, \gamma=0.99, \lambda=0.95, \text{clip}=0.2, c_{\text{ent}}=0.0, c_{\text{vf}}=0.5$ | `configs/ppo.yaml`, `runs/main_p1_s0/run_meta.json` | `configs/ppo.yaml:18-29`, `runs/main_p1_s0/run_meta.json:26-37` | **verified** | Confirmed from frozen YAML and training run metadata. Note: batch size is 256 (prior summary claimed 64). |
| **C10** | Observation & Reward Normalization | Observations: unnormalized in VecNormalize (`norm_obs=False`); Rewards: normalized (`norm_reward=True`) | `configs/ppo.yaml`, `runs/main_p1_s0/run_meta.json`, `tests/test_inference_correctness.py` | `configs/ppo.yaml:35-36`, `runs/main_p1_s0/run_meta.json:39-40`, `deploy/ppo_inference_node.py:91-105` | **verified** | Manual fixed scaling in `observation.py` avoids double-normalization; VecNormalize only tracks returns. |
| **C11** | Implemented Reward Function | $R_t = 10 \cdot (d_{t-1} - d_t) - 0.05 + 100 \cdot [\text{goal}] - 100 \cdot [\text{collision}]$ | `rl/env/tb3_navigation_env.py` | `rl/env/tb3_navigation_env.py:180-185, 521-528` | **verified** | $k_{\text{progress}}=10.0$ (not 2.0 as stated in some earlier summaries!), $c_{\text{step}}=0.05, r_{\text{goal}}=100.0, r_{\text{collision}}=100.0$. |
| **C12** | Clearance and Smoothness Reward Activation | Clearance penalty and action smoothness penalty active in training environment | `rl/env/tb3_navigation_env.py` | `rl/env/tb3_navigation_env.py:184-185, 521-528` | **contradicted** | In `tb3_navigation_env.py`, `self.r_clear_coef = 0.0` and `self.r_smooth_coef = 0.0`. They are NOT added to the step reward! The effective reward consists strictly of progress, step penalty, goal bonus, and collision penalty. |
| **C13** | Termination & Truncation Semantics | Goal ($d \le 0.20\,\text{m}$ in eval / $0.15\,\text{m}$ in env default) or collision ($d_{\min} \le 0.16\,\text{m}$ or obstacle bounding box) $\to$ terminated; step $\ge 350 \to$ truncated | `rl/env/tb3_navigation_env.py`, `deploy/ppo_inference_node.py` | `rl/env/tb3_navigation_env.py:513-518`, `deploy/ppo_inference_node.py:141, 169, 177` | **verified** | Conforms strictly to Gymnasium distinction between terminated (absorbing terminal state) and truncated (finite horizon timeout). |
| **C14** | Domain Randomization Design (P1–P4) | Planned $2 \times 2$ factorial matrix: P1 (nominal), P2 (sensor only), P3 (dynamics only), P4 (combined) | `configs/ppo.yaml`, `rl/env/randomizers.py` | `configs/ppo.yaml:49-68`, `rl/env/randomizers.py:1-25` | **verified** | Classes implemented in `rl/env/randomizers.py`: `SensorRandomizer`, `DynamicsRandomizer`, `CombinedRandomizer`. |
| **C15** | P4 Joint Rejection Constraint | Sensor latency + control delay $\le 0.40\,\text{s}$ | `rl/env/randomizers.py` | `rl/env/randomizers.py:291-305` | **verified** | Rejection sampling enforces sum of sensor delivery latency and control delay does not exceed 2 control steps ($0.40\,\text{s}$). |
| **C16** | Completed Policy Training Runs | P1 completed, P4 completed; P2 and P3 pending | `runs/main_p1_s0/training_summary.json`, `runs/main_p4_s0/training_summary.json` | `runs/main_p1_s0/training_summary.json`, `runs/main_p4_s0/training_summary.json` | **verified** | Only P1 and P4 runs exist in `runs/`. No P2 or P3 directories exist. |
| **C17** | Training Timestep Budgets | P1: 852,816 timesteps; P4: 704,512 timesteps | `runs/main_p1_s0/training_summary.json`, `runs/main_p4_s0/training_summary.json` | `runs/main_p1_s0/training_summary.json:5`, `runs/main_p4_s0/training_summary.json:5` | **verified** | Confirmed from JSON summaries. P1 completed 852,816 steps; P4 completed 704,512 steps. |
| **C18** | Unequal Training Budgets Confound | P1 and P4 training budgets are unequal (852k vs 704k) | `runs/main_p1_s0/training_summary.json`, `runs/main_p4_s0/training_summary.json` | `runs/main_p1_s0/training_summary.json:5`, `runs/main_p4_s0/training_summary.json:5` | **verified** | Major methodological limitation. P1 received ~148k (+21%) more training steps than P4 due to execution deadline constraints. |
| **C19** | Number of Independent Training Seeds | Only seed 0 evaluated for P1 and P4 | `runs/main_p1_s0/run_meta.json`, `runs/main_p4_s0/run_meta.json` | `runs/main_p1_s0/run_meta.json:3`, `runs/main_p4_s0/run_meta.json:3` | **verified** | Single seed ($s=0$) for both policies. No inter-seed variance or standard error of training is available. |
| **C20** | Held-Out Evaluation Dataset | 50 identical held-out test scenarios | `scenarios/test/scenarios.json`, `runs/eval/p1_test/eval_P1_nominal.json`, `runs/eval/p4_test/eval_P4_nominal.json` | `scenarios/test/scenarios.json` (`d9a6a25c...`), `runs/eval/p1_vs_p4_comparison.json:4` | **verified** | Both policies evaluated on the exact same 50 scenarios (IDs 0–49, seeds 2000000–2000049). |
| **C21** | Held-Out Success Counts | P1: 47 / 50 (94.0%), P4: 49 / 50 (98.0%) | `runs/eval/p1_test/eval_P1_nominal.json`, `runs/eval/p4_test/eval_P4_nominal.json` | `runs/eval/p1_test/eval_P1_nominal.json:5`, `runs/eval/p4_test/eval_P4_nominal.json:5` | **verified** | Direct tally of per-episode records: P1 has 47 True, 3 False; P4 has 49 True, 1 False. |
| **C22** | Failure Mode Breakdown | P1: 1 collision (Scen 7), 2 timeouts (Scen 25, 37); P4: 1 collision (Scen 34), 0 timeouts | `runs/eval/p1_test/eval_P1_nominal.json`, `runs/eval/p4_test/eval_P4_nominal.json` | `runs/eval/p1_test/eval_P1_nominal.json`, `runs/eval/p4_test/eval_P4_nominal.json` | **verified** | Note: Prior summary swapped scenario failure labels (claimed Scen 25 was collision and Scen 7 was timeout; raw data proves Scen 7 is Collision, Scen 25 and 37 are Timeouts for P1). |
| **C23** | Paired McNemar Test Significance | Reported exact two-sided $p = 0.625$; difference is NOT statistically significant | Independent computation from raw episode records | `scipy.stats.binomtest(1, 4, 0.5, alternative='two-sided').pvalue` | **verified** | Contingency table: $a=46$ (both succ), $b=1$ (P1 only), $c=3$ (P4 only), $d=0$ (both fail). Discordant pairs $= 4$. Exact two-sided binomial $p = 0.6250 > 0.05$. |
| **C24** | 95% Wilson Score Confidence Intervals | P1: [83.45%, 97.97%]; P4: [89.51%, 99.65%] | Statistical calculation on $N=50$ | Recomputed via Wilson score interval formula | **verified** | Overlapping confidence intervals confirm lack of statistical separation at $N=50$. |
| **C25** | Evaluation Episode Duration (Successes) | Mean time-to-goal: P1 = 21.23 s, P4 = 20.30 s (at 5 Hz control rate: 106.17 steps vs 101.49 steps) | `runs/eval/p1_test/eval_P1_nominal.json`, `runs/eval/p4_test/eval_P4_nominal.json` | Computed from `per_episode` success entries | **contradicted** (Prior summary claimed ~10.6 s / 9.9 s by dividing steps by 10 instead of 5 Hz!) | At 5 Hz ($dt=0.2\,\text{s}$), 106.17 steps corresponds to $21.23\,\text{s}$, and 101.49 steps corresponds to $20.30\,\text{s}$. (Earlier summary erroneously divided by 10 Hz). |
| **C26** | Live Gazebo Classic Simulation Demo | 3 out of 3 trials successful in Gazebo Classic on port 11345; 0 collisions, 0 timeouts | `runs/demo_run_20261025_182130/demo_summary.json` | `runs/demo_run_20261025_182130/demo_summary.json:1-25` | **verified** | Goals reached: (1.2, 0.0) in 57 steps, (1.0, 1.0) in 61 steps, (-1.0, 1.0) in 88 steps. Qualitative demonstration only. |
| **C27** | Physical Robot Hardware Validation | Navigation demonstrated on physical TurtleBot3 hardware | Entire repository | Workspace audit | **contradicted** (No physical hardware runs exist) | All experiments were performed in Gazebo Classic simulation or fast kinematic simulation. Hardware deployment remains future work. |
| **C28** | Test Suite Status | 11 inference and observation tests pass | Terminal execution of `pytest` | `tests/test_inference_correctness.py`, `tests/test_observation.py` | **verified** | Executed: `PYTHONPATH=. pytest -v tests/test_inference_correctness.py tests/test_observation.py`. 11 passed in 1.75s. |
| **C29** | Reward Audit Traces | 321,557 lines of per-step reward traces generated across 4 baseline behaviors | `results/reward_audit_traces.json`, commit `5294e6b` | `git show 5294e6b`, `tests/generate_reward_audit.py` | **verified** | Traces committed by Siddhi Vinayak on remote `origin/main`. |
| **C30** | Equal Contribution Declaration | Both authors contributed equally with complementary responsibilities | `docs/midterm_report/team_contributions.md`, Git history | `docs/midterm_report/team_contributions.md:1-40`, git commit logs | **verified** | Author declaration confirmed. Task distribution verified against code and test artifacts. |

---

## 3. Detailed Technical Contradictions & Audit Findings

### A. Observation Vector Dimensions & Structure
- **Finding:** The observation vector is strictly **43-dimensional**.
- **Breakdown:**
  1. LiDAR Bins: Indices `0:36` (36 features) — $10^\circ$ min-pooling, normalized by $3.50\,\text{m}$ to $[0, 1]$.
  2. Goal Distance: Index `36` (1 feature) — normalized by arena diagonal ($5.657\,\text{m}$) to $[0, 1]$.
  3. Goal Bearing: Indices `37:39` (2 features) — $[\sin(\phi), \cos(\phi)]$, range $[-1, 1]$.
  4. Measured Linear Velocity: Index `39` (1 feature) — $v / 0.15\,\text{m/s}$, clipped to $[-1, 1]$.
  5. Measured Angular Velocity: Index `40` (1 feature) — $\omega / 0.60\,\text{rad/s}$, clipped to $[-1, 1]$.
  6. Previous Action Buffer: Indices `41:43` (2 features) — $[u_{v, t-1}, u_{\omega, t-1}]$, range $[-1, 1]$.
- **Inconsistency Resolved:** Earlier reports had conflicting arithmetic claiming 42 dimensions or suggesting an extra $c_{\text{flag}}$ collision bit. No collision flag is present in the observation vector. $36 + 1 + 2 + 1 + 1 + 2 = 43$.

### B. Action Mapping and Velocity Limits
- **Finding:** The active control mapping in `tb3_navigation_env.py` (lines 488–490) and `deploy/ppo_inference_node.py` (lines 196–197) is:
  $$\text{cmd\_v} = 0.075 \cdot (u_0 + 1.0) \in [0.0, 0.15]\,\text{m/s}$$
  $$\text{cmd\_\omega} = 0.60 \cdot u_1 \in [-0.60, 0.60]\,\text{rad/s}$$
- **Inconsistency Resolved:** Prior audit summaries reported linear velocity up to $0.22\,\text{m/s}$ and angular velocity up to $\pm 2.0\,\text{rad/s}$. Those are the factory maximum physical specifications of the TurtleBot3 Burger, but the reinforcement learning action space and inference node explicitly scale actions to $[0, 0.15]\,\text{m/s}$ and $[-0.6, 0.6]\,\text{rad/s}$ to ensure safe operation.

### C. Reward Function Formulation
- **Finding:** The implemented reward function in `rl/env/tb3_navigation_env.py` (lines 521–528) is:
  $$R_t = 10.0 \cdot (d_{t-1} - d_t) - 0.05 + 100.0 \cdot \mathbb{I}_{\text{goal}} - 100.0 \cdot \mathbb{I}_{\text{collision}}$$
- **Inconsistency Resolved:**
  - Progress multiplier is **$10.0$**, not $2.0$.
  - Obstacle clearance penalty and action smoothness penalty are defined in code but their coefficients are set to **$0.0$** (`self.r_clear_coef = 0.0`, `self.r_smooth_coef = 0.0`). They were NOT active during P1 or P4 training.

### D. PPO Policy Network Architecture
- **Finding:** The network architecture in `configs/ppo.yaml` and `runs/main_p1_s0/run_meta.json` is:
  - Policy network ($\pi$): `[128, 128]` with `tanh` activations.
  - Value network ($vf$): `[128, 128]` with `tanh` activations.
  - Initial log std: $-0.5$.
  - Batch size: 256 minibatches (rollout 2048 per env across 4 workers).
- **Inconsistency Resolved:** Earlier summaries listed `[64, 64]` and batch size 64; the actual training metadata confirms `[128, 128]` and batch size 256.

### E. Evaluation Episode Duration
- **Finding:** Control rate is $5.0\,\text{Hz}$ ($dt = 0.20\,\text{s}$).
  - P1 average successful steps: $106.17 \to \mathbf{21.23\,\text{s}}$ average duration.
  - P4 average successful steps: $101.49 \to \mathbf{20.30\,\text{s}}$ average duration.
- **Inconsistency Resolved:** Earlier summaries reported $\sim 10.6\,\text{s}$ and $\sim 9.9\,\text{s}$ because steps were erroneously multiplied by $0.1\,\text{s}$ instead of $0.2\,\text{s}$.

---

## 4. Empirical Evaluation & Statistical Rigor

### A. Raw Episode Verification ($N = 50$ Identical Test Scenarios)
- **Dataset Hash:** `scenarios/test/scenarios.json` (`sha256: d9a6a25cef080fa05a19627b64d8da72ea0c9f201c149733f604eb9195b0b4bf`)
- **P1 Raw Records:** `runs/eval/p1_test/eval_P1_nominal.json` (`sha256: 3a9a9282c270fc869c6d1857db5d2313e6ac1f5c1cf4caefefb9e0ca806e8d50`)
- **P4 Raw Records:** `runs/eval/p4_test/eval_P4_nominal.json` (`sha256: 5a1f30de3f1114744012cb5daaa236471b46f90dd01a5b923a1b22e97b407a8e`)

#### Contingency Matrix (Paired Outcomes across 50 Scenarios)

| | P4 Succeeded | P4 Failed | Total |
| :--- | :---: | :---: | :---: |
| **P1 Succeeded** | 46 | 1 (Scenario 34) | **47** |
| **P1 Failed** | 3 (Scenarios 7, 25, 37) | 0 | **3** |
| **Total** | **49** | **1** | **50** |

#### Discordant Scenarios Detail
- **Scenario 7:** P1 Collided (step 68); P4 Succeeded (step 69, $t=13.8\,\text{s}$).
- **Scenario 25:** P1 Timed Out (step 350); P4 Succeeded (step 89, $t=17.8\,\text{s}$).
- **Scenario 34:** P1 Succeeded (step 85, $t=17.0\,\text{s}$); P4 Collided (step 78).
- **Scenario 37:** P1 Timed Out (step 350); P4 Succeeded (step 88, $t=17.6\,\text{s}$).

#### Statistical Significance
- **Exact McNemar Test (Two-Sided Binomial):**
  $$p = \sum_{k=0}^{1} \binom{4}{k} 0.5^4 + \sum_{k=3}^{4} \binom{4}{k} 0.5^4 = 2 \times (1 + 4) \times \frac{1}{16} = \frac{10}{16} = \mathbf{0.6250}$$
- **Conclusion:** $p = 0.6250 \gg 0.05$. The observed difference (+4.0% success rate) is **not statistically significant** on an evaluation sample of $N=50$.

#### 95% Wilson Score Confidence Intervals
- **P1 (47/50, 94.0%):** $[83.45\%, 97.97\%]$
- **P4 (49/50, 98.0%):** $[89.51\%, 99.65\%]$

---

## 5. Methodological & Experimental Limitations

1. **Unequal Training Budgets:**
   - P1: 852,816 transitions.
   - P4: 704,512 transitions (17.4% fewer steps).
   - *Implication:* P1 had longer gradient exposure, creating a confounding variable when comparing asymptotic performance.
2. **Single Training Seed ($s=0$):**
   - Both policies were trained with a single random seed (`seed=0`).
   - *Implication:* No inter-seed variance or standard error of training is available. The report must state that results reflect a single training run per condition.
3. **Missing P2 and P3 Ablations:**
   - Planned factorial design includes P2 (sensor noise only) and P3 (dynamics perturbation only).
   - Neither P2 nor P3 has been trained. The report must clearly designate P2 and P3 as *pending ablation studies*.
4. **Sample Size for Statistical Power:**
   - With $N=50$, a 4% difference has statistical power $< 0.15$. Establishing statistical significance would require $N \ge 250$ test scenarios.
5. **No Physical Hardware Validation:**
   - The project is strictly simulation-based (ROS 2 Humble + Gazebo Classic 11.10.2).
   - No hardware runs on physical TurtleBot3 robots were performed. Sim-to-real claims must be phrased as *domain randomization pipeline design for future hardware transfer*.

---

## 6. Bibliographic Verification (`docs/midterm_report/references.bib`)

| BibTeX Key | Citation Details | Verified Primary Source | Status |
| :--- | :--- | :--- | :---: |
| `schulman2017proximal` | Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347 (2017) | arXiv:1707.06347 | **verified** |
| `tobin2017domain` | Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017 | DOI: 10.1109/IROS.2017.8202133 | **verified** |
| `tai2017virtual` | Tai et al., "Virtual-to-Real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation", IROS 2017 | DOI: 10.1109/IROS.2017.8202134 | **verified** |
| `raffin2021stable` | Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations in Python", JMLR 22(268):1-8 (2021) | JMLR v22/20-1364 | **verified** |
| `macenski2022robot` | Macenski et al., "Robot Operating System 2: Design, Architecture, and Uses in the Wild", Science Robotics 7(66):eabm6074 (2022) | DOI: 10.1126/scirobotics.abm6074 | **verified** |
| `koenig2004design` | Koenig & Howard, "Design and Use Paradigms for Gazebo, an Open-Source Multi-Robot Simulator", IROS 2004 | DOI: 10.1109/IROS.2004.1389727 | **verified** |
| `peng2018sim` | Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA 2018 | DOI: 10.1109/ICRA.2018.8460528 | **verified** |
| `towers2023gymnasium` | Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments", arXiv:2407.17032 (2024) | arXiv:2407.17032 | **verified** |

---

## 7. Author Contribution Evidence Audit

### Author Declaration
> *“Both authors contributed equally to the project; their responsibilities were complementary.”*

| Author | Roll Number | Verified Tasks in Code / Git History | Source Evidence |
| :--- | :--- | :--- | :--- |
| **Saigirish S** | `B24ME1066` | Project architecture, simulation environment (`tb3_navigation_env.py`), observation preprocessor (`observation.py`), domain randomizers (`randomizers.py`), PPO training pipeline (`train_ppo.py`), P1 and P4 model training, evaluation suites, live Gazebo demo runbook. | Git commits `e3c40fd`, `6e068c5`, `4da798a`, `d181656`, `9bf431d`. |
| **Siddhi Vinayak** | `B24BB1036` | Reward-function audit harness, per-step reward trace logging, baseline behavior evaluation (reactive, random, stationary, spin-in-place), reward decomposition unit testing. | Git commit `5294e6b`, `tests/generate_reward_audit.py`, `results/reward_audit_traces.json`. |

---

## 8. Toolchain Availability

- **Command Tested:** `kpsewhich IEEEtran.cls`
- **Result:** Exited with code 127 (`kpsewhich: command not found`).
- **Additional Compilers Checked:** `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `typst`, `pandoc` are **NOT installed** in the system PATH.
- **Action Taken:** In accordance with instructions, no system packages or TeX Live distributions were installed. Phase 1 produces purely evidence documentation.

---

## 9. Unresolved Questions Requiring Author Confirmation

Before drafting the formal IEEE paper in Phase 2, the following administrative metadata must be confirmed by the authors:
1. **Course Title & Code:** (e.g., *CS/ME xxx: Reinforcement Learning in Robotics*)
2. **Academic Department:** (e.g., *Department of Mechanical Engineering / Bioscience & Bioengineering*)
3. **Faculty Advisor / Course Instructor:** Exact title and name for acknowledgement or header.
4. **Academic Term:** (e.g., *Autumn Semester 2026*).
