# Official IEEE Midterm Progress Report Rebuild & Scientific Audit Summary

**Project:** Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation  
**Repository:** [https://github.com/Saigirish23/tb3-rl-nav](https://github.com/Saigirish23/tb3-rl-nav)  
**Working Branch:** `midterm-report-draft` (Tracking `origin/midterm-docs-audit`)  
**Output Directory:** `docs/midterm_report/ieee_final/`  
**Date:** October 25, 2026  

---

## 1. Executive Summary of Rebuild & Audit

In accordance with user directives, the mid-term progress report was rebuilt from the ground up to comply with official IEEE conference layout standards, correct scientific terminology and statistical claims, provide explicit reconciliation of training throughput and Wilson confidence intervals, and ensure strict technical integrity.

All files are delivered in an isolated, self-contained directory: [`docs/midterm_report/ieee_final/`](file:///home/saigirish/sim-real-rl/docs/midterm_report/ieee_final). No previous files were overwritten.

---

## 2. Itemized Verification of Required Corrections

### 1. Official IEEE Conference Template & Formatting
- **LaTeX Source:** Created [`docs/midterm_report/ieee_final/main.tex`](file:///home/saigirish/sim-real-rl/docs/midterm_report/ieee_final/main.tex) using `\documentclass[conference]{IEEEtran}`.
- **Official Class File:** Downloaded and verified the official IEEE class file [`IEEEtran.cls`](file:///home/saigirish/sim-real-rl/docs/midterm_report/ieee_final/IEEEtran.cls) (281,957 bytes) into the project directory for 100% self-contained compilation.
- **Typography & Section Headings:** Enforced Times-Roman serif typography, Roman-numeral section headings (`I. INTRODUCTION AND OBJECTIVES`, `II. PLANNED METHODOLOGY...`), uppercase IEEE table captions above tables, and figure captions below figures.
- **Author Block:** Correctly structured author blocks with roll numbers and institutional affiliation (IIT Jodhpur). Author departments were omitted pending formal verification to prevent unverified claims.

### 2. Collision Definition Discrepancy Disclosed
- **Audit Finding:**
  - In `rl/env/tb3_navigation_env.py` (lines 599–615), collision is checked via *ground-truth geometry* (`_check_collision()`), evaluating whether the circular footprint ($r_{\text{robot}} = 0.105\,\text{m}$) intersects arena walls ($|x|, |y| \ge 1.95 - 0.105 = 1.845\,\text{m}$) or active obstacle bounding boxes.
  - In `deploy/ppo_inference_node.py` (line 136), autonomous inference lacks ground-truth pose and instead employs a *LiDAR-range collision proxy* ($\min(\text{ranges}) < 0.16\,\text{m}$).
- **Correction Applied:** Explicitly disclosed in Section II.D (*Collision Detection Definition Discrepancy*) and identified as an operational limitation: while ground-truth geometry governs simulation training, deployment relies on a range proxy that can trigger in narrow traversable spaces without actual physical contact.

### 3. Scope of Domain-Randomization Evaluation & 4x4 Matrix
- **Audit Finding:** The held-out evaluation on $N=50$ scenarios was conducted under **nominal evaluation conditions** (unperturbed sensor feedback and nominal dynamics). It demonstrates nominal competence, but does *not* prove out-of-distribution robustness or sim-to-real transfer.
- **Corrections Applied:**
  - Abstract and Section IV explicitly specify: *"under nominal evaluation conditions on scenarios with static obstacles."*
  - Added **TABLE III: Planned $4 \times 4$ Policy / Environment Cross-Evaluation Matrix** in Section V, detailing the evaluation of policies P1–P4 against Nominal, Sensor DR, Dynamics DR, and Combined DR conditions.
  - Physical robot deployment is explicitly designated as future work, conditional on physical robot access, lab availability, and safety approval.

### 4. Training Throughput Reconciled
- **Audit Finding:**
  - **P1:** 852,816 transitions over 40,496.33 seconds (11.2490 hours). Overall wall-clock throughput $= 852,816 / 40,496.33 = \mathbf{21.06\,\text{steps/s}}$. The previously stated $19.75\,\text{steps/s}$ represents the running mean steps-per-second across update batches.
  - **P4:** 704,512 transitions over 35,551.54 seconds (9.8754 hours). Overall wall-clock throughput $= 704,512 / 35,551.54 = \mathbf{19.82\,\text{steps/s}}$. The previously stated $19.69\,\text{steps/s}$ represents the running mean steps-per-second.
- **Correction Applied:** Both metrics are reported with technical precision in Section III: *"Completed 852,816 transitions over 40,496.33 seconds (11.25 hours). The overall wall-clock throughput is 21.06 steps/s (19.75 steps/s running average logged in run metadata)."*

### 5. Exact Wilson 95% Confidence Intervals Recalculated
Using the exact standard Wilson score formula without continuity correction:
$$\tilde{p} = \frac{k + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \quad \text{CI} = \tilde{p} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}, \quad z = 1.959964$$
- **P1 ($k=47, n=50$):** Centre $= 0.9086$, Margin $= 0.0708 \implies \mathbf{[83.78\%, 97.94\%]}$ (matches `eval_P1_nominal.json` stored values `0.837827, 0.979385`).
- **P4 ($k=49, n=50$):** Centre $= 0.9458$, Margin $= 0.0507 \implies \mathbf{[89.50\%, 99.65\%]}$ (matches `eval_P4_nominal.json` stored values `0.895043, 0.996461`).
- **Correction Applied:** Updated Table II, Abstract, and Fig. 2 to report $[83.78\%, 97.94\%]$ and $[89.50\%, 99.65\%]$.

### 6. Software Versions and Test Counts Contextualized
- **PyTorch Version:** Verified directly from `runs/main_p1_s0/pip_freeze.txt` (line 45): `torch==2.11.0`. The mention of `2.14.1+cpu` in `docs/environment.md` refers to an alternative CPU-only build specification; the active training run executed on PyTorch 2.11.0. This distinction is documented in Section III.
- **Test Count Scope:** Collected 69 tests across 12 test modules in `tests/`. The "11/11 tests" refers specifically to the core targeted unit tests (`tests/test_inference_correctness.py` [4 tests] and `tests/test_observation.py` [7 tests]). This scope is explicitly clarified in Section III.

### 7. Statistical Power Claim Corrected
- **Correction Applied:** Removed the unsupported claim that $N \ge 250$ guarantees $80\%$ statistical power.
- **Power Analysis Context:** Documented in Section IV.A that sample-size requirements depend fundamentally on the test design and discordance frequency. An unpaired comparison for $94\%$ vs. $98\%$ at $\alpha=0.05$ with $80\%$ power requires approximately 370 episodes per policy. Under paired testing, power is governed by the frequency of discordant pairs ($b$ and $c$). $N=250$ is presented as an expanded benchmark to improve estimation precision.
- **Paired McNemar Result Preserved:** Exact two-sided binomial $p = 0.6250$ on discordant counts $b=1, c=3$ verified and reported.

### 8. Moderated Scientific Wording
- Replaced *"indicates sample efficiency"* with *"is consistent with, but does not definitively establish, improved sample efficiency"* (Section V).
- Replaced *"reproducible foundation"* with *"an initial implementation and evaluation baseline"* (Abstract and Conclusion).
- Replaced *"100% success rate"* with *"3/3 successful trials"* (Section III).
- Clarified that obstacles in the evaluation suite are *static* (Abstract and Section IV).
- Confirmed that clearance and smoothness rewards were inactive in training (`r_clear_coef = 0.0`, `r_smooth_coef = 0.0`) and avoided claiming they were enforced by the reward function.

### 9. Individual Contributions & Equal Contribution Confirmation
- Section VI accurately attributes:
  - **Saigirish S:** RL pipeline architecture, 43-D observation preprocessor, dead-reckoning module, domain randomizers, multi-worker Gazebo orchestration, P1 and P4 training, ROS 2 standalone inference node.
  - **Siddhi Vinayak:** Reward decomposition audit framework, per-step reward trace logging, unit test execution and baseline validation, participation in experimental design and hardware deployment preparation.
- Explicitly stated that both authors must confirm the final declaration upon submission.

### 10. Nav2 Literature Citation Added
- Verified primary citation added: **S. Macenski, F. Martín, R. White, and J. G. Clavero, "The Marathon 2: A Navigation System," in *Proc. IEEE/RSJ IROS*, 2020, pp. 2718–2725** (DOI: `10.1109/IROS45743.2020.9341207`).
- Correctly cited in Section I alongside ROS 2 \cite{macenski2022robot} for the classical navigation stack.

### 11. Figure Quality & Layout Repaired
- **Fig. 1 (Architecture):** Re-rendered with generous spacing, distinct colored process containers, and non-overlapping arrow callouts (`figures/fig1_system_architecture.pdf`).
- **Fig. 2 (Outcomes & CIs):** Moved legend to upper center horizontal layout (`ncol=3`), eliminating bar label occlusion. Wilson error bars display exact recalculated bounds $[83.78\%, 97.94\%]$ and $[89.50\%, 99.65\%]$ (`figures/fig2_evaluation_results.pdf`).
- **Fig. 3 (Efficiency Boxplots):** Sized and formatted with clear axis labels and IEEE font hierarchy (`figures/fig3_trajectory_metrics.pdf`).

---

## 3. Visual & Compilation Acceptance Testing

1. **LaTeX Source:** Complete, valid `IEEEtran` document in `docs/midterm_report/ieee_final/main.tex` with matching `references.bib` and vector figures in `figures/`.
2. **Direct PDF Compilation:** Generated via `docs/midterm_report/ieee_final/generate_pdf.py` producing `docs/midterm_report/ieee_final/midterm_report.pdf` (392 KB).
3. **Page Flow & Visual Check:**
   - **Page 1:** Title, authors, abstract, index terms, Section I (Introduction & Objectives). Objectives conclude cleanly on page 1 with zero orphaned text.
   - **Page 2:** Section II (Methodology), Fig. 1 (Architecture), Subsections II.A--II.D (Observation, Action, Reward, Collision discrepancy), Table I (Hyperparameters).
   - **Page 3:** Section III (Implementation & GitHub link), Section IV (Preliminary Results), Table II (Held-out Results).
   - **Page 4:** Fig. 2 (Results & CIs), Fig. 3 (Boxplots), Subsection IV.A (Statistical Rigor), Section V (Limitations), Table III ($4 \times 4$ Planned Matrix).
   - **Page 5:** Subsection V.A (Roadmap), Section VI (Contributions), Section VII (Conclusion), References [1]–[9].
   - **Formatting:** Clean running headers, page numbers ("1", "2"...), crisp vector figures, and zero text clipping.
