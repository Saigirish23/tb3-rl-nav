# Unverified Claims and Items Requiring Author Confirmation

**Project:** Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation  
**Repository:** [https://github.com/Saigirish23/tb3-rl-nav](https://github.com/Saigirish23/tb3-rl-nav)  
**Date:** October 25, 2026  

This document explicitly identifies any claims, administrative metadata, or experimental conditions that could not be verified directly from the code, logs, and evaluation files in the workspace, along with the exact actions or data required to resolve them.

---

## 1. Administrative / Course Metadata (Excluded from Body)

The following fields were not present in any tracked repository configuration files or Git metadata. In accordance with instructions, they were omitted or left as standard academic placeholders rather than fabricated:

1. **Course Name & Code:**
   - *Status:* `NOT FOUND IN REPOSITORY`
   - *Action Required:* If required by your professor or course rubric, insert the formal course code (e.g., *CS/ME xxx: Autonomous Mobile Robotics*) into the author affiliation block or title footnote.
2. **Faculty Advisor / Course Instructor:**
   - *Status:* `NOT FOUND IN REPOSITORY`
   - *Action Required:* If required, add the instructor's name and title in an Acknowledgements section.
3. **Academic Term / Semester:**
   - *Status:* Date confirmed (October 2026); semester designation (e.g., *Autumn Semester 2026-27*) not formally declared in repo files.

---

## 2. Experimental Claims Excluded or Explicitly Qualified

1. **Hardware Sim-to-Real Transfer:**
   - *Claim:* "Demonstrated sim-to-real transfer on physical TurtleBot3."
   - *Status:* **EXCLUDED / LABELED AS FUTURE WORK**.
   - *Evidence Gap:* No rosbag logs, hardware telemetry, physical sensor recordings, or physical testing videos exist in the repository. All completed experiments used Gazebo Classic simulation.
   - *Resolution in Report:* The paper explicitly states in Sections 2, 5, 6, and 7 that physical deployment is a planned post-midterm milestone.
2. **Intermediate Ablation Policies (P2 and P3):**
   - *Claim:* "Empirical evaluation of sensor-only (P2) and dynamics-only (P3) policies."
   - *Status:* **EXCLUDED / LABELED AS PENDING**.
   - *Evidence Gap:* No checkpoints, logs, or directories exist for P2 or P3 in `runs/`. Only P1 and P4 were trained and evaluated.
   - *Resolution in Report:* The $2 \times 2$ matrix is detailed conceptually in Section 4, but P2 and P3 are explicitly reported as pending ablations in the results and roadmap.
3. **Statistical Superiority of P4 over P1:**
   - *Claim:* "P4 is statistically superior to P1 based on 98% vs 94% success."
   - *Status:* **CONTRADICTED BY EXACT STATISTICAL TEST**.
   - *Evidence Finding:* A paired exact two-sided binomial McNemar test on the 50 held-out scenarios yields $p = 0.6250 > 0.05$.
   - *Resolution in Report:* The paper reports the observed descriptive improvement (+4.0%) alongside the rigorous caveat that it is not statistically significant at $N=50$, establishing the need for an expanded test suite ($N \ge 250$).
4. **Hardware-in-the-Loop Demonstration:**
   - *Claim:* "Hardware-in-the-loop qualitative demonstration."
   - *Status:* **TERMINOLOGY CORRECTED**.
   - *Evidence Finding:* The 3-trial demo was executed in Gazebo Classic simulation on port 11345 via `deploy/ppo_inference_node.py`. No physical controller hardware was in the loop.
   - *Resolution in Report:* Terminology corrected to *"simulation-based navigation demonstration in Gazebo Classic."*
