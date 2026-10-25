# Team Contributions & Project Metadata

**Project:** Sim-to-Real 2D LiDAR Navigation Pipeline using PPO and Domain Randomization  
**Repository:** `https://github.com/Saigirish23/tb3-rl-nav.git`  
**Milestone:** Mid-Term Progress Audit (October 2026)  

---

## 1. Academic & Institutional Metadata

- **Institution:** Indian Institute of Technology Jodhpur (IIT Jodhpur)  
  *(Evidence: `@iitj.ac.in` domain on all Git commit authors and team correspondence)*
- **Department(s):**
  - Department of Mechanical Engineering (ME)
  - Department of Biosciences and Bioengineering (BB)
- **Course Number / Title:** `NOT FOUND — USER INPUT REQUIRED`  
  *(Placeholder: Please insert exact course code, e.g., CSxxx / MExxx / AIxxx)*
- **Project Supervisor / Guide:** `NOT FOUND — USER INPUT REQUIRED`  
  *(Placeholder: Please insert Faculty Advisor / Supervisor Name & Designation)*
- **Academic Term / Session:** 2026–2027 Academic Year (Semester: `NOT FOUND — USER INPUT REQUIRED`)

---

## 2. Team Members & Detailed Work Division

### Member 1: Saigirish S
- **Roll Number:** `B24ME1066`
- **Email:** `b24me1066@iitj.ac.in`
- **GitHub Username:** `Saigirish23`
- **Program / Department:** Mechanical Engineering, IIT Jodhpur
- **Core Responsibilities:** Simulation Architecture, RL Environment, Training Infrastructure, and Multi-Worker Orchestration.
- **Specific Evidence-Backed Contributions:**
  1. **Gazebo Simulation Infrastructure:** Configured Gazebo Classic 11.10.2 simulation world (`ros2_ws/src/tb3_rl_sim/worlds/arena_4x4.world`), custom physics parameters, and headless launch pipelines (`ros2_ws/src/tb3_rl_sim/launch/headless_sim.launch.py`).
  2. **Gymnasium Environment Wrapper:** Developed [`rl/env/tb3_navigation_env.py`](file:///home/saigirish/sim-real-rl/rl/env/tb3_navigation_env.py), establishing the continuous action mapping ($v \in [0, 0.15]\,\text{m/s}$, $\omega \in [-0.60, 0.60]\,\text{rad/s}$), reward computation, and Gymnasium reset/step lifecycle.
  3. **Multi-Worker Training & Isolation:** Designed parallel PPO training loop with 4 isolated Gazebo Classic instances operating across master ports $11345..11348$ and ROS domain IDs $30..33$ to eliminate DDS crosstalk.
  4. **Domain Randomization Engine:** Co-developed sensor observation noise models (LiDAR noise, dropout, odometry drift) and robot dynamics perturbations (mass, wheel friction, motor lag, communication delay) in [`rl/env/randomizers.py`](file:///home/saigirish/sim-real-rl/rl/env/randomizers.py).
  5. **Orchestration & Deadline Protection Daemon:** Implemented [`scripts/orchestrate_p1_to_p4.py`](file:///home/saigirish/sim-real-rl/scripts/orchestrate_p1_to_p4.py), providing autonomous phase management, model promotion, graceful termination (`.stop_requested`), and progress logging.
  6. **Model Execution:** Supervised training of P1 nominal baseline ($852,816$ steps) and P4 combined policy ($704,512$ steps).

### Member 2: Siddhi Vinayak
- **Roll Number:** `B24BB1036`
- **Email:** `b24bb1036@iitj.ac.in`
- **GitHub Username:** `siddhivn`
- **Program / Department:** Biosciences and Bioengineering, IIT Jodhpur
- **Core Responsibilities:** Baseline Controllers, Evaluation Suite, Verification Harnesses, Safety Watchdogs, and Deployment Node.
- **Specific Evidence-Backed Contributions:**
  1. **Reactive Baseline Controller:** Authored [`baselines/reactive_controller.py`](file:///home/saigirish/sim-real-rl/baselines/reactive_controller.py), implementing potential-field obstacle avoidance and goal attraction reaching $92.0\%$ nominal success rate.
  2. **Scenario Generation & Split Isolation:** Designed deterministic pseudo-random scenario generator ([`scenarios/scenario_generator.py`](file:///home/saigirish/sim-real-rl/scenarios/scenario_generator.py)) ensuring zero seed leakage across train ($0..9$), validation ($1\text{M}..1\text{M}+49$), and test ($2\text{M}..2\text{M}+99$) splits.
  3. **Simulator Missing-Scan Watchdog:** Implemented [`rl/train/watchdog.py`](file:///home/saigirish/sim-real-rl/rl/train/watchdog.py) using monotonic clock supervision to detect missing scans ($>2.0\text{ s}$), cleanly terminate owned process groups via SIGTERM/SIGKILL, restart Gazebo, and reload checkpoints without accounting corruption.
  4. **Reward Audit Traces & Verification Tests:** Developed verification suites ([`tests/test_reward_audit.py`](file:///home/saigirish/sim-real-rl/tests/test_reward_audit.py), [`tests/test_watchdog.py`](file:///home/saigirish/sim-real-rl/tests/test_watchdog.py)) ensuring exact parity of reward components across corner-case robot behaviors.
  5. **Standalone Deployment Architecture:** Designed the zero-ROS pure NumPy observation pipeline ([`rl/env/observation.py`](file:///home/saigirish/sim-real-rl/rl/env/observation.py)) and ROS 2 deployment node ([`deploy/ppo_inference_node.py`](file:///home/saigirish/sim-real-rl/deploy/ppo_inference_node.py)) for downstream sim-to-real transfer.

---

## 3. Contribution Matrix Summary

| Component / Task | Primary Lead | Supporting Contributor | Evidence File(s) |
| :--- | :---: | :---: | :--- |
| **Gazebo Simulator & Robot Spawner** | Saigirish S | Siddhi Vinayak | `ros2_ws/src/tb3_rl_sim/` |
| **Gymnasium Env & Observation Preprocessor** | Saigirish S | Siddhi Vinayak | `rl/env/tb3_navigation_env.py`, `rl/env/observation.py` |
| **Reactive Controller Baseline** | Siddhi Vinayak | Saigirish S | `baselines/reactive_controller.py` |
| **Missing-Scan Watchdog & Process Supervisor** | Siddhi Vinayak | Saigirish S | `rl/train/watchdog.py`, `tests/test_watchdog.py` |
| **PPO Training Engine & Device Fallback** | Saigirish S | Siddhi Vinayak | `rl/train/train_ppo.py`, `configs/ppo.yaml` |
| **Domain Randomizers (P2/P3/P4)** | Saigirish S | Siddhi Vinayak | `rl/env/randomizers.py` |
| **Scenario Generator & Test Split Isolation** | Siddhi Vinayak | Saigirish S | `scenarios/scenario_generator.py` |
| **Multi-Worker Autonomous Orchestrator** | Saigirish S | Siddhi Vinayak | `scripts/orchestrate_p1_to_p4.py` |
| **Held-Out Evaluation Suite & Metrics** | Siddhi Vinayak | Saigirish S | `scripts/evaluate.py`, `runs/eval/` |
| **ROS 2 Inference Node & Demo Automation** | Siddhi Vinayak | Saigirish S | `deploy/ppo_inference_node.py`, `scripts/run_simulation_demo.py` |
