# Phase 2 Stress-Testing Benchmark Specification (Final Finalized)

**Project:** Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation  
**Robot Platform:** TurtleBot3 Burger ($L = 0.138\,\text{m}, W = 0.178\,\text{m}, R_{\text{bound}} = 0.113\,\text{m}$)  
**Environment:** ROS 2 Humble + Gazebo Classic 11 ($5.0 \times 5.0\,\text{m}$ expanded warehouse arena)  
**Status:** Implementation-Ready / Pilot Pending Execution  

---

## 1. Executive Summary & Factorial Architecture

This benchmark evaluates four trained **Gen1 PPO policies** ($P_1, P_2, P_3, P_4$) across nine targeted stress conditions to diagnose failure modes along three orthogonal generalization axes: geometric layout complexity, perceptual degradation, and actuation lag/mismatch.

```
                                  2x2 FACTORIAL POLICY DESIGN
                                       Sensor Randomization
                                    None                   Sampled
                           +----------------------+----------------------+
               Nominal     |          P1          |          P2          |
Dynamics                   |   852,816 steps      |   704,512 steps      |
Randomization              |   Best ckpt: 450k    |   Best ckpt: 300k    |
                           +----------------------+----------------------+
               Sampled     |          P3          |          P4          |
               Command-    |   704,512 steps      |   704,512 steps      |
               Response    |   Best ckpt: 600k    |   Best ckpt: 400k    |
                           +----------------------+----------------------+
```

### 1.1 Policy Architecture & Verified Observation Space
* **Policy Network Topology**: Multi-Layer Perceptron (MLP) with **two hidden layers of 128 units each (`[128, 128]`)** using `tanh` activations. Output layer maps to continuous action $\mathbf{u} \in [-1.0, 1.0]^2$.
* **Observation Vector ($D = 43$ continuous dimensions)**:
  - `obs[0:36]`: 36 min-pooled LiDAR range bins ($10^\circ$ per bin, normalized to $[0.0, 1.0]$ over $[0.12, 3.50]\,\text{m}$). Bin 0 is forward-centered on $[-5^\circ, +5^\circ)$.
  - `obs[36]`: Normalized Euclidean distance to goal: $d_{\text{goal}} / D_{\text{diag}}$, where $D_{\text{diag}} = \sqrt{4.0^2 + 4.0^2} \approx 5.6569\,\text{m}$.
  - `obs[37:39]`: Relative goal bearing unit vector $[\sin\theta_{\text{rel}}, \cos\theta_{\text{rel}}]$, where $\theta_{\text{rel}} = \text{wrap\_to\_pi}(\theta_{\text{goal}} - \theta_{\text{robot}})$.
  - `obs[39:41]`: Normalized measured velocities from odometry: $v_{\text{meas}} / 0.15\,\text{m/s}$ and $\omega_{\text{meas}} / 0.60\,\text{rad/s}$ (clipped to $[-1, 1]$).
  - `obs[41:43]`: Previous normalized action command $\mathbf{u}_{t-1} \in [-1.0, 1.0]^2$ (initialized to $[0.0, 0.0]$ at reset).
* **Action Mapping**:
  $$v = 0.075 \cdot (u_v + 1.0) \in [0.0, 0.15]\,\text{m/s}, \quad \omega = 0.60 \cdot u_\omega \in [-0.60, +0.60]\,\text{rad/s}$$
* **Training Reward Function**:
  $$r_t = 10.0 \cdot (d_t - d_{t+1}) - 0.05 + 100.0 \cdot \mathbb{I}[\text{goal}] - 100.0 \cdot \mathbb{I}[\text{collision}], \quad \text{Goal Radius } r_{\text{goal}} = 0.20\,\text{m}$$

---

## 2. Reconciled Gen1 Training Metadata & Randomization Regimes

### 2.1 Recorded Training Budgets vs. Selected Checkpoints
* **$P_1$ (Nominal Baseline)**: Configured target: $850,000$ steps; **actual recorded interactions: $852,816$ steps**. Deployed checkpoint: **$450,000$ steps** (best eval reward $+324.5$).
* **$P_2$ (Sensor Randomization)**: Configured target: $700,000$ steps; **actual recorded interactions: $704,512$ steps**. Deployed checkpoint: **$300,000$ steps** (best eval reward $+298.1$).
* **$P_3$ (Dynamics Randomization)**: Configured target: $700,000$ steps; **actual recorded interactions: $704,512$ steps**. Deployed checkpoint: **$600,000$ steps** (best eval reward $+287.4$).
* **$P_4$ (Combined Randomization)**: Configured target: $700,000$ steps; **actual recorded interactions: $704,512$ steps**. Deployed checkpoint: **$400,000$ steps** (best eval reward $+271.8$).

> [!NOTE]
> All policies fully completed their configured training budgets. Deployed checkpoints reflect peak validation performance under periodic evaluations; earlier checkpoint selection in $P_2$ ($300\text{k}$) and $P_4$ ($400\text{k}$) does not represent training truncation or failure.

### 2.2 Exact Training Randomization Formulations
* **$P_2$ Sensor Randomization**:
  Per-episode sampled parameters from `rl/env/randomizers.py`:
  $$a \sim U(0.003, 0.015)\,\text{m}, \quad b \sim U(0.000, 0.010), \quad \sigma(r) = a + b \cdot r$$
  $$r_{\text{noisy}} = \text{clip}(r + \mathcal{N}(0, \sigma^2(r)), 0.12, 3.50)$$
  - Independent beam dropout: $p_{\text{beam}} \sim U(0.00, 0.05)$ (reads $3.50\,\text{m}$).
  - Contiguous block dropout: $p_{\text{block}} \sim U(0.00, 0.30)$, $N_{\text{blocks}} \sim U_{\text{int}}(1, 2)$, $L_{\text{block}} \sim U_{\text{int}}(3, 10)$ beams.
  - Scan delivery latency: $L_{\text{scan}} \in \{0, 1\}$ scans ($0.0\,\text{s}$ or $0.20\,\text{s}$ buffer delay).
  - Mounting offsets: yaw $\Delta \psi \sim U(-3^\circ, +3^\circ)$, position $\Delta x, \Delta y \sim U(-0.02, +0.02)\,\text{m}$.
  - Odometry perturbations: velocity scale $k_v^{\text{odom}} \sim U(0.95, 1.05)$, yaw bias $b_\omega \sim U(-0.005, +0.005)\,\text{rad/s}$, velocity noise $\sigma_v \sim U(0.000, 0.005)\,\text{m/s}, \sigma_\omega \sim U(0.000, 0.020)\,\text{rad/s}$.
* **$P_3$ Command-Response Dynamics Randomization**:
  - Command gains: $k_v \sim U(0.85, 1.15)$, $k_\omega \sim U(0.85, 1.15)$.
  - Actuator lag time constant: $\tau \sim U(0.00, 0.30)\,\text{s}$.
  - Control delay: $d \sim U(0.00, 0.20)\,\text{s}$.
  - Acceleration limits: $a_{\text{lin}} \sim U(0.20, 0.60)\,\text{m/s}^2$, $a_{\text{ang}} \sim U(1.00, 3.00)\,\text{rad/s}^2$.
  - **Important Physics Grounding**: Sampled friction multipliers ($\mu \in [0.70, 1.30]$) and mass multipliers ($m \in [0.90, 1.20]$) are **never applied to Gazebo physics**. The robot's simulated mass ($1.0\,\text{kg}$) and surface friction ($\mu = 1.0$) remain at nominal defaults. $P_3/P_4$ exhibit **command-response dynamics robustness**, not physical tribological/inertial robustness.
* **$P_4$ Combined Randomization**:
  Joint sampling of $P_2$ and $P_3$ under total latency rejection constraint:
  $$L_{\text{scan}} \cdot 0.20\,\text{s} + d \le 0.40\,\text{s}$$

---

## 3. Verified Static World Geometry (`arena_4x4.world`)

All object dimensions and poses match the physical SDF definitions:

| Static Model | Geometry | Center Pose $(x, y, z)$ | Size / Extents $(dx, dy, dz)$ or $(r, l)$ | Navigable Bounding Box |
| :--- | :---: | :---: | :---: | :---: |
| **Arena Perimeter Walls** | Box | $(0.0, \pm 2.55, 0.25)$ & $(\pm 2.55, 0.0, 0.25)$ | $5.20 \times 0.10 \times 0.50\,\text{m}$ | Inner boundary: $[-2.50, +2.50]^2\,\text{m}$ |
| **`warehouse_rack_1`** | Box | $(-1.15, +0.65, 0.45)$ | $0.40 \times 1.50 \times 0.90\,\text{m}$ | $x \in [-1.35, -0.95], y \in [-0.10, +1.40]$ |
| **`warehouse_rack_2`** | Box | $(+1.15, -0.65, 0.45)$ | $0.40 \times 1.50 \times 0.90\,\text{m}$ | $x \in [+0.95, +1.35], y \in [-1.40, +0.10]$ |
| **`warehouse_shelf_north`** | Box | $(0.00, +2.25, 0.40)$ | $1.40 \times 0.30 \times 0.80\,\text{m}$ | $x \in [-0.70, +0.70], y \in [+2.10, +2.40]$ |
| **`pallet_stack`** | Box | $(-1.95, +1.95, 0.175)$ | $0.50 \times 0.50 \times 0.35\,\text{m}$ | $x \in [-2.20, -1.70], y \in [+1.70, +2.20]$ |
| **`storage_container`** | **Cylinder** | $(+1.95, -1.95, 0.225)$ | **$r = 0.22\,\text{m}, l = 0.45\,\text{m}$** | $x \in [+1.73, +2.17], y \in [-2.17, -1.73]$ |

* **Layout Composition**:
  - Overlaid Warehouse (Cases 1, 2, 5, 6, 7A, 7B, 8): Retains all five static objects.
  - Isolated Layout (Cases 3 & 4): Retains boundary walls, North shelf, pallet stack, and container, but **removes `warehouse_rack_1` and `warehouse_rack_2`** to isolate turning mechanics without spatial interference.

---

## 4. Specification of the Nine Benchmark Stress Conditions

### Case 1: Ultra-Narrow Corridor Stress Test
* **Objective**: Measure lateral centering stability and heading drift damping in a narrow $0.450\,\text{m}$ corridor.
* **Geometry**: Parallel walls along $y \in [-1.00, +1.00]\,\text{m}$ (length $D = 2.00\,\text{m}$), inner faces at $x = \pm 0.225\,\text{m}$.
* **Clearance**: Robot width $W = 0.178\,\text{m}$ gives $0.225 - 0.089 = \mathbf{0.136\,\text{m}}$ ($13.6\,\text{cm}$) centered clearance.
* **Kinematic Drift**: At $v = 0.15\,\text{m/s}$ and $\Delta t = 0.20\,\text{s}$ ($67\,\text{steps}$ to traverse $2.0\,\text{m}$), solving $2.0\tan\theta + 0.5(0.178\cos\theta + 0.138\sin\theta) = 0.225\,\text{m}$ yields critical heading error:
  $$\theta_{\text{crit}} = \mathbf{3.77^\circ} \quad (0.0657\,\text{rad})$$
  Per-step lateral displacement: $\Delta x = 0.03\sin(3.77^\circ) = \mathbf{1.97\,\text{mm/step}}$.
* **Poses**: Start $(0.00, -1.40, \pi/2)$, Goal $(0.00, +1.40)$.

### Case 2: Deep Concave U-Trap Escape
* **Objective**: Evaluate escape from non-convex local minima requiring counter-intuitive retreat away from the goal.
* **Geometry**: South back wall $x \in [-0.50, +0.50], y \in [-0.60, -0.40]$; West wall $x \in [-0.50, -0.30], y \in [-0.40, +0.50]$; East wall $x \in [+0.30, +0.50], y \in [-0.40, +0.50]$. Open mouth at North ($y = +0.50$).
* **Poses**: Start $(0.00, 0.00, -\pi/2)$ (inside trap facing South), Goal $(0.00, -1.80)$ (behind South wall).
* **Escape Route Status**:
  - **Route 2A (Analytically Designed Choke-Centered Path)**: Centerline at $x = -0.725\,\text{m}$ provides exactly **$13.6\,\text{cm}$ centered clearance** from U-trap West wall ($x = -0.50$) and Rack 1 East face ($x = -0.95$). Waypoints: $(0.0, 0.0) \to (0.0, 0.65) \to (-0.725, 0.65) \to (-0.725, -0.65) \to (0.0, -1.80)$. Piecewise linear length: **$4.034\,\text{m}$** (fillet-smoothed: **$3.903\,\text{m}$**).
  - **Route 2B (Provisional Grid-Planner Reference)**: 8-connected $2\,\text{cm}$ grid A\* computes a path of length **$3.718\,\text{m}$** ($167\,\text{steps}$), passing at $x \approx -0.62\,\text{m}$ ($3.1\,\text{cm}$ clearance to U-trap lip). *Status: Labeled strictly as a provisional discrete grid-planner reference; diagonal cell-to-cell transitions cut corners and require Gazebo dynamic validation.*

### Case 3: Overlapping Baffle Chicane (S-Curve)
* **Objective**: Evaluate continuous non-holonomic maneuvering through an overlapping chicane with $100\%$ occluded direct line-of-sight.
* **Geometry**: Barrier 1 at $x \in [-2.50, +0.25], y \in [-0.50, -0.30]$; Barrier 2 at $x \in [-0.25, +2.50], y \in [+0.30, +0.50]$. Baffles overlap across $[-0.25, +0.25]\,\text{m}$ ($0.50\,\text{m}$ overlap). Centerline $x = 0$ is blocked by solid wall material.
* **Poses**: Start $(0.00, -1.80, \pi/2)$, Goal $(0.00, +1.80)$.
* **Reference Trajectories (Status: Analytically Designed / Provisional)**:
  - Path 3A (Circumscribed Circle Design): Passes tips at $x = \pm 0.370\,\text{m}$ ($+0.7\,\text{cm}$ circle clearance, $+3.1\,\text{cm}$ chassis clearance, length $4.357\,\text{m}$).
  - Path 3B (Safety-Inflated Design): Passes tips at $x = \pm 0.410\,\text{m}$ ($+3.7\,\text{cm}$ circle clearance, $+6.5\,\text{cm}$ chassis clearance, length $4.494\,\text{m}$).
  - *Status: Both paths are classified as provisional analytical reference trajectories requiring complete swept-footprint validation in Gazebo pilot trials.*

### Case 4: Blind $90^\circ$ L-Junction
* **Objective**: Measure corner-cutting tendencies and reactive turning around an occluded right-angle vertex.
* **Geometry**: Inner vertex $V_{\text{in}} = (-0.15, +0.30)\,\text{m}$, channel width $0.550\,\text{m}$.
* **Poses**: Start $(+0.125, -1.000, \pi/2)$, Goal $(-1.000, +0.575)$.
* **Analytical Sweep**: Centerline turn radius $R_c = 0.275\,\text{m}$. Inner clearance: $0.275 - 0.089 = \mathbf{18.6\,\text{cm}}$. Outer swept radius $R_{\text{sweep}} = \sqrt{(0.275+0.089)^2 + 0.069^2} = 0.3705\,\text{m}$, yielding outer straight wall clearance of $0.550 - 0.3705 = \mathbf{18.0\,\text{cm}}$.
* *Status: Analytically designed reference geometry; full approach-turn-exit swept footprint requires Gazebo pilot validation.*

### Case 5: Dynamic Obstacle Crossing / Intercept Avoidance
* **Objective**: Evaluate dynamic deceleration, yielding, and collision avoidance under unmodeled cross-traffic.
* **Obstacle Setup & Tangency Condition**:
  - Moving cylinder radius $R_{\text{obs}} = 0.15\,\text{m}$, height $H = 0.50\,\text{m}$.
  - Initial Position: $(x_{\text{obs}}, y_{\text{obs}}) = (-0.80, 0.00)\,\text{m}$.
  - **Initial Rack 1 Clearance**: Rack 1 East face is at $x = -0.95\,\text{m}$ ($y \in [-0.10, +1.40]$). At $y = 0.00$, cylinder West edge is at $x = -0.80 - 0.15 = -0.95\,\text{m}$.
  - **Initial Clearance is exactly 0 cm (tangency)**.
  - *Critical Flag*: This initial tangency requires explicit Gazebo pilot validation to ensure contact sensors do not register false collision impulses at spawn. The trajectory must not be claimed collision-free until experimentally validated.
* **Deterministic Motion Schedule & Stopping Clearance**:
  - Stationary from $t = 0.00\,\text{s}$ to $t = 6.00\,\text{s}$.
  - At $t = 6.00\,\text{s}$, moves East at $v_{\text{obs}} = +0.20\,\text{m/s}$.
  - Crosses centerline $x = 0.00$ at $t = 6.00 + 0.80 / 0.20 = \mathbf{10.00\,\text{s}}$.
  - **Stopping Position**: Halts at $(x, y) = (\mathbf{+0.70}, 0.00)\,\text{m}$ at $t = 13.50\,\text{s}$ ($1.50\,\text{m}$ total transit).
  - **Clearance to Rack 2**: Rack 2 West face is at $x = +0.95\,\text{m}$. Cylinder leading edge at stop is $x = 0.70 + 0.15 = 0.85\,\text{m}$, leaving **$10.0\,\text{cm}$ face clearance** and **$11.9\,\text{cm}$ corner clearance** to Rack 2 corner $(0.95, 0.10)$.
* **Robot Collision Encounter Window**:
  - Robot starts at $(0.00, -1.50, \pi/2)$.
  - Ideal crossing time at constant $0.15\,\text{m/s}$: $t = 1.50 / 0.15 = \mathbf{10.00\,\text{s}}$.
  - Actual Gazebo arrival accounting for startup acceleration ramp ($a_{\text{lin}} \le 0.60\,\text{m/s}^2$): $t_{\text{acc}} = 0.25\,\text{s}$ ($s_{\text{acc}} = 0.019\,\text{m}$), cruise time: $1.481 / 0.15 = 9.875\,\text{s} \implies t_r = \mathbf{10.125\,\text{s}}$.
  - Relative velocity: $\mathbf{v}_{\text{rel}} = (0.20, -0.15)\,\text{m/s}$ (speed $0.25\,\text{m/s}$).
  - Contact collision threshold ($R_{\text{obs}} + R_{\text{bound}} = 0.263\,\text{m}$) defines collision window:
    $$t \in [8.95, 11.05]\,\text{s} \quad (\Delta t = 2.10\,\text{s})$$

### Case 6: Stepped Funnel with Bottleneck Throat
* **Objective**: Evaluate speed regulation and chassis alignment down a progressively constricting corridor.
* **Geometry**: 4 symmetric box pairs over $y \in [-0.50, +0.50]\,\text{m}$:
  - Step 1 ($y \in [-0.50, -0.25]$): Width $0.900\,\text{m}$, centered clearance: $\mathbf{36.1\,\text{cm}}$.
  - Step 2 ($y \in [-0.25, 0.00]$): Width $0.700\,\text{m}$, centered clearance: $\mathbf{26.1\,\text{cm}}$.
  - Step 3 ($y \in [0.00, +0.25]$): Width $0.500\,\text{m}$, centered clearance: $\mathbf{16.1\,\text{cm}}$.
  - Step 4 Throat ($y \in [+0.25, +0.50]$): Width **$0.350\,\text{m}$**, centered clearance: $\mathbf{8.6\,\text{cm}}$ (Circle: $\mathbf{6.2\,\text{cm}}$).
* **Throat Critical Drift Angle**: $\theta_{\text{crit}} = \mathbf{15.77^\circ}$ ($0.2752\,\text{rad}$).
* **Poses**: Start $(0.00, -1.20, \pi/2)$, Goal $(0.00, +1.20)$.

### Case 7A: Distance-Dependent Sensor Degradation
* **Perturbation**: Out-of-distribution noise applied to all 360 LiDAR beams:
  $$\sigma(r) = 0.025 + 0.015 r \quad (4.0\,\text{cm noise at } 1.0\,\text{m}; 7.8\,\text{cm noise at } 3.5\,\text{m})$$
* **Layout**: Nominal warehouse layout across matched scenario seeds.

### Case 7B: Forward $30^\circ$ Blind Sector
* **Perturbation**: Forward sector ($[-15^\circ, +15^\circ]$) masked to $3.50\,\text{m}$ ($1.0$ normalized). Covers 30 beams across Bins 35, 0, and 1 with 0 unaffected beams.
* **Layout**: Nominal warehouse layout across matched scenario seeds.

### Case 8: Actuation Lag & Dynamics Stress Test
* **Perturbations Applied**:
  - Command delay: $d = 0.20\,\text{s} = \Delta t$ (1-step delay: $a_t^{\text{delayed}} = a_{t-1}^{\text{cmd}}$).
  - Out-of-distribution command gains: $k_v = 0.80, k_\omega = 1.20$ (outside training range $[0.85, 1.15]$).
  - Acceleration limits: $a_{\text{lin}} = 0.20\,\text{m/s}^2, a_{\text{ang}} = 1.00\,\text{rad/s}^2$.
  - First-order actuator lag: $\tau = 0.30\,\text{s}$ ($\alpha = \frac{0.20}{0.30 + 0.20} = 0.40$).
  - Velocity output clipping: $v \in [0.0, 0.15]\,\text{m/s}, \omega \in [-0.60, +0.60]\,\text{rad/s}$.
  - Buffer reset: `prev_action = [0.0, 0.0]`, delay buffer cleared, `actual_v = 0.0, actual_omega = 0.0`.
* **Layout**: Nominal warehouse layout across matched scenario seeds.

---

## 5. Evaluation Protocol, Metrics & Statistical Methodology

### 5.1 Episode Allocation & Execution Modes
* **Pilot Mode (Default)**: $9\,\text{conditions} \times 4\,\text{policies} \times 1\,\text{seed} = \mathbf{36\,\text{episodes}}$ (technical verification of resets, spawns, and logging).
* **Main Benchmark Mode**: $9\,\text{conditions} \times 4\,\text{policies} \times 20\,\text{matched seeds} = \mathbf{720\,\text{episodes}}$.
* **Status**: Both pilot and benchmark modes are **planned and pending runtime execution**.

### 5.2 Performance Metrics & Selection Bias Treatment
1. **Success ($SR$)**: $d_{\text{goal}} \le 0.25\,\text{m}$ without collision. *(Goal tolerance is $0.25\,\text{m}$ for evaluation vs $0.20\,\text{m}$ during training to account for robot physical diameter of $0.226\,\text{m}$).*
2. **Collision ($CR$)**: Dual-source collision detection:
   - `collision_geometric`: minimum distance from physical chassis footprint boundary to any obstacle $\le 0.0\,\text{m}$.
   - `collision_sensor`: Gazebo contact trigger OR raw `/scan` range $\le 0.120\,\text{m}$.
   - *Source of Truth*: An episode is classified as collision if **either** source indicates collision. Disagreements are logged (`collision_disagreement: 1`), with geometric clearance serving as ground truth.
3. **Timeout ($TO$)**: Reaching $T_{\text{max}} = 500\,\text{steps}$ ($100.0\,\text{s}$) without goal arrival or collision.
4. **Penalized Completion Time ($T_{\text{penalized}}$)**: Full-cohort metric assigning $T_{\text{penalized}} = T_{\text{goal}}$ for successes, and $T_{\text{max}} = 100.0\,\text{s}$ for collisions and timeouts.
5. **Conditional Trajectory Metrics**:
   - Path length ($L$), completion time ($T_{\text{goal}}$), and action smoothness ($\text{TV}_{\text{norm}}(\omega)$) are **conditional on successful episodes only**.
   - *Selection Bias Statement*: Success-conditioned metrics describe kinematic efficiency among episodes that solved the task, but do not represent all episodes. They must always be evaluated alongside full-cohort $SR$, $CR$, $TO$, and $T_{\text{penalized}}$.

### 5.3 Statistical Comparison Protocol
* **Binary Outcomes ($SR, CR$)**: Paired **McNemar's test** across matched seeds with Bonferroni correction for family-wise error control across the 6 policy pairs ($\alpha_{\text{adjusted}} = 0.05 / 6 \approx 0.0083$). Unadjusted tests are labeled exploratory.
* **Continuous Outcomes ($T_{\text{goal}}, L, \text{TV}_{\text{norm}}$)**: Paired **two-sided Wilcoxon signed-rank test** evaluated strictly over the subset of mutually successful seed pairs. **The exact number of common successful seeds ($N_{\text{both}}$) must be reported for every comparison**.
* **Statistical Power Notice**: With $N = 20$ matched seeds per condition, statistical power is inherently limited. Small effect sizes cannot be reliably distinguished from sample noise; claims of definitive policy superiority are avoided in favor of diagnostic failure-mode characterization.
