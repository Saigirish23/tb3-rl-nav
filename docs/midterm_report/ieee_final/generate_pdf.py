#!/usr/bin/env python3
"""Build publication-quality PDF report strictly matching IEEE conference layout.

Audit Corrections Applied:
1. Strict IEEE conference styling: Serif Roman typography, Roman-numeral section headings (I, II, III...),
   Table captions above tables, Figure captions below figures.
2. Collision definition clarified: ground-truth footprint contact in training env vs LiDAR-range proxy (0.16m) in inference.
3. Nominal evaluation conditions made explicit in abstract and text.
4. Planned 4x4 Policy / Environment Cross-Evaluation Matrix added.
5. Training throughput reconciled: P1 = 21.06 steps/s (19.75 s/s running avg); P4 = 19.82 steps/s (19.69 s/s running avg).
6. Exact Wilson score 95% CIs: P1 = [83.78%, 97.94%], P4 = [89.50%, 99.65%].
7. PyTorch 2.11.0 verified from training run pip_freeze.txt; test suite contextualized (11 targeted unit tests within 69 total tests).
8. Statistical power claim corrected: removed N>=250 guarantee; explained paired discordance & unpaired 370-episode baseline.
9. Scientific wording moderated: "consistent with improved sample efficiency", "initial baseline", "3/3 trials".
10. Individual contributions verified; author confirmation note included; department line omitted pending confirmation.
11. Nav2 citation added: Macenski et al., "The Marathon 2: A Navigation System" (IROS 2020) [6].
12. Figures repaired: clean layout, no overlapping boxes or obscured bar labels.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Times-Roman', 8.5)
        self.setFillColor(colors.HexColor('#333333'))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(50, 752, 'IEEE MID-TERM PROGRESS REPORT: SIM-TO-REAL RL LIDAR NAVIGATION')
            self.drawRightString(562, 752, 'OCTOBER 2026')
            self.setStrokeColor(colors.HexColor('#888888'))
            self.setLineWidth(0.4)
            self.line(50, 746, 562, 746)
            
        # Footer
        page_text = f'{self._pageNumber}'
        self.drawRightString(562, 34, page_text)
        self.drawString(50, 34, 'Authorized Mid-Term Progress Report — IIT Jodhpur')
        self.setStrokeColor(colors.HexColor('#888888'))
        self.setLineWidth(0.4)
        self.line(50, 44, 562, 44)
        self.restoreState()

def build_pdf(filename='docs/midterm_report/ieee_final/midterm_report.pdf'):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Official IEEE typography (Times-Roman / Times-Bold)
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=15,
        leading=18,
        alignment=1, # Centered
        textColor=colors.black,
        spaceAfter=7
    )
    
    author_style = ParagraphStyle(
        'AuthorBlock',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        textColor=colors.black,
        spaceAfter=7
    )

    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=4, # Justified
        textColor=colors.black,
        spaceAfter=6
    )
    
    heading1 = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12.5,
        alignment=1, # Centered Roman numerals in IEEE
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    heading2 = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Times-BoldItalic',
        fontSize=8.8,
        leading=11.5,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.2,
        alignment=4, # Justified
        textColor=colors.black,
        spaceAfter=4
    )

    table_caption = ParagraphStyle(
        'TableCaption',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.black,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    fig_caption = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.black,
        spaceBefore=2,
        spaceAfter=6
    )

    story = []
    
    # Title
    story.append(Paragraph('Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation: Mid-Term Progress Report', title_style))
    
    # Author Block (Departments omitted pending formal institutional confirmation)
    authors_text = (
        "<b>Saigirish S</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Siddhi Vinayak</b><br/>"
        "<i>Indian Institute of Technology Jodhpur</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <i>Indian Institute of Technology Jodhpur</i><br/>"
        "Roll No: B24ME1066 &nbsp;&nbsp;&bull;&nbsp;&nbsp; b24me1066@iitj.ac.in &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Roll No: B24BB1036 &nbsp;&nbsp;&bull;&nbsp;&nbsp; b24bb1036@iitj.ac.in<br/>"
        "<i>Contribution Declaration: Both authors contributed to the project with complementary responsibilities.</i>"
    )
    story.append(Paragraph(authors_text, author_style))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.black, spaceBefore=1, spaceAfter=5))
    
    # Abstract
    story.append(Paragraph(
        "<b><i>Abstract</i></b>---Mapless navigation in dynamic or unstructured environments is a foundational challenge in mobile robotics. "
        "While deep reinforcement learning (DRL) provides an end-to-end framework directly mapping onboard sensor observations to continuous velocity commands, "
        "standard policies trained in idealized physics engines frequently suffer severe performance degradation when deployed to real hardware due to the reality gap. "
        "In this mid-term progress report, we investigate a domain-randomization-enhanced Proximal Policy Optimization (PPO) pipeline for a differential-drive "
        "TurtleBot3 mobile robot using a 2D planar LiDAR. We detail the system architecture, comprising a 43-dimensional observation vector with forward-centered "
        "sector min-pooling and dead-reckoning pose tracking, safe action scaling, and a multi-component reward formulation. We present preliminary empirical results "
        "on a held-out test suite of N=50 identical scenarios with static obstacles comparing a nominal baseline policy (P1, 852,816 training steps) against a "
        "combined sensor-and-dynamics randomized policy (P4, 704,512 training steps) <b>under nominal evaluation conditions</b>. On nominal testing, P1 achieves a "
        "94.0% success rate (95% Wilson score CI: [83.78%, 97.94%]) while P4 achieves 98.0% (95% Wilson score CI: [89.50%, 99.65%]). A paired McNemar exact test yields "
        "p = 0.6250, demonstrating that the observed 4.0% descriptive difference is not statistically significant at N=50. We discuss experimental limitations, "
        "unequal training budgets, and the necessity of a full 4x4 policy/environment cross-evaluation matrix to rigorously evaluate perturbation robustness, "
        "concluding with a plan toward real-robot transfer conditional on laboratory access.",
        abstract_text
    ))
    
    story.append(Paragraph(
        "<b><i>Index Terms</i></b>---Reinforcement Learning, Mobile Robot Navigation, Sim-to-Real Transfer, Domain Randomization, "
        "Proximal Policy Optimization, TurtleBot3, ROS 2, Gazebo.",
        abstract_text
    ))
    story.append(Spacer(1, 2))
    
    # Section I
    story.append(Paragraph('I. INTRODUCTION AND OBJECTIVES', heading1))
    story.append(Paragraph(
        "Autonomous ground vehicles operating in GPS-denied indoor environments typically rely on the classical navigation stack consisting of simultaneous "
        "localization and mapping (SLAM), global path planning (A*/Dijkstra), and local trajectory optimization (Dynamic Window Approach / TEB) [5], [6]. While effective "
        "in static structured spaces, classical pipelines require substantial computational overhead for real-time dense map maintenance and are notoriously sensitive "
        "to localization drift, dynamic obstacle occlusion, and sensor noise [6].",
        body
    ))
    story.append(Paragraph(
        "To overcome these constraints, mapless navigation via Deep Reinforcement Learning (DRL) has emerged as an attractive alternative [3]. By training neural network "
        "policies to directly map raw range findings and relative goal coordinates to low-level motor velocity commands, the system avoids explicit global mapping. "
        "However, training policies directly on physical robotic platforms is prohibitively expensive, wear-intensive, and dangerous during early exploration. "
        "High-fidelity simulators such as Gazebo [7] provide safe, accelerated experience collection, but policies trained in nominal simulation often experience dramatic "
        "performance degradation when transferred to physical hardware---the sim-to-real gap [2], [8].",
        body
    ))
    story.append(Paragraph(
        "Domain Randomization (DR) bridges this gap by exposing the agent during training to a broad distribution of simulated sensor distortions and unmodeled "
        "dynamic variations, compelling the policy to learn robust feedback strategies that generalize across physical discrepancies [2], [8]. This project systematically "
        "investigates the quantitative impact of isolated and combined domain randomization on policy robustness, sample efficiency, and hardware transferability for "
        "TurtleBot3 LiDAR navigation.",
        body
    ))

    # Section I.A Problem Statement & Objectives (Kept together to avoid page breaks)
    obj_elements = [
        Paragraph('A. Problem Statement', heading2),
        Paragraph(
            "We consider a differential-drive TurtleBot3 Burger robot navigating from an arbitrary starting pose to a static Euclidean target in an obstacle-cluttered "
            "4m x 4m bounded workspace. Navigation must proceed maplessly, relying solely on an onboard 2D laser scanner (LDS-01) and wheel odometry.",
            body
        ),
        Paragraph('B. Research Objectives', heading2),
        Paragraph(
            "&bull; <b>Mapless PPO Navigation Architecture:</b> Design an end-to-end continuous actor-critic navigation framework operating at 5.0 Hz that enforces "
            "obstacle clearance and smooth kinematic control.<br/>"
            "&bull; <b>2x2 Domain Randomization Factorial Design:</b> Implement modular sensor perturbations (range noise, dropouts, latency) and dynamics perturbations "
            "(actuator lag, delay, acceleration limits) to isolate their individual and combined effects across four policy regimes: P1 (Nominal baseline), P2 (Sensor only), "
            "P3 (Dynamics only), and P4 (Combined DR) [2], [8].<br/>"
            "&bull; <b>Rigorous Empirical Benchmarking:</b> Evaluate trained policies across identical held-out scenarios using paired statistical hypothesis testing and "
            "confidence intervals.<br/>"
            "&bull; <b>Physical Sim-to-Real Transfer:</b> Validate the trained policy without fine-tuning on physical TurtleBot3 hardware in a laboratory environment, "
            "conditional on physical robot access, laboratory availability, and safety approval.",
            body
        )
    ]
    story.append(KeepTogether(obj_elements))
    story.append(Spacer(1, 2))

    # Section II
    story.append(Paragraph('II. PLANNED METHODOLOGY AND SYSTEM DESIGN', heading1))
    story.append(Paragraph(
        "The system pipeline (Fig. 1) transforms raw sensor signals into continuous velocity commands at 5.0 Hz with zero SLAM or global path planner dependencies.",
        body
    ))
    
    # Figure 1
    story.append(Spacer(1, 1))
    story.append(Image('docs/midterm_report/ieee_final/figures/fig1_system_architecture.png', width=490, height=135))
    story.append(Paragraph('Fig. 1. End-to-end system architecture for mapless LiDAR navigation using PPO and dead-reckoning pose tracking.', fig_caption))
    story.append(Spacer(1, 2))

    story.append(Paragraph('A. Observation Space Formulation', heading2))
    story.append(Paragraph(
        "To ensure exact parity between simulation and physical deployment, observation preprocessing is implemented as a pure NumPy module "
        "(<code>ObservationPreprocessor</code>) with zero ROS dependencies. The observation vector is strictly <b>43-dimensional</b>:<br/>"
        "1. <b>LiDAR Sector Min-Pooling (features 0--35):</b> 360-ray range arrays from the LDS-01 sensor are pooled into 36 angular sectors of width 10 degrees. "
        "Bin 0 is centered directly forward along the robot heading ([-5 deg, +5 deg)). Min-pooling guarantees conservative obstacle capture, invalid beams are "
        "mapped to 3.50m, and values are normalized to [0, 1].<br/>"
        "2. <b>Normalized Goal Distance (feature 36):</b> Euclidean distance to target normalized by arena diagonal (5.657m).<br/>"
        "3. <b>Goal Bearing Representation (features 37--38):</b> Relative heading angle decomposed into [sin, cos] to eliminate the wrap-around discontinuity at +/- pi.<br/>"
        "4. <b>Measured Velocity Feedback (features 39--40):</b> Forward linear velocity normalized by 0.15 m/s and angular velocity normalized by 0.60 rad/s.<br/>"
        "5. <b>Previous Action Buffer (features 41--42):</b> The normalized control vector executed at step t-1 in [-1, 1]^2, facilitating smooth transitions.",
        body
    ))

    story.append(Paragraph('B. Action Space and Velocity Scaling', heading2))
    story.append(Paragraph(
        "The policy outputs a normalized action u in [-1, 1]^2, which is scaled into physical commands via:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>v = 0.075 * (u_0 + 1.0) &isin; [0.0, 0.15] m/s</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>&omega; = 0.60 * u_1 &isin; [-0.60, 0.60] rad/s</b><br/>"
        "This constrains the robot to purely forward motion (v &ge; 0), preventing collisions into the rear chassis blind spot.",
        body
    ))

    story.append(Paragraph('C. Reward Formulation and Gymnasium Semantics', heading2))
    story.append(Paragraph(
        "Under Gymnasium environment conventions [9], the step reward function is formulated as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>R_t = 10.0 * (d_{t-1} - d_t) - 0.05 + 100 * [goal] - 100 * [collision]</b><br/>"
        "Reaching the goal (dist &le; 0.20m) sets <code>terminated=True</code>. Exceeding 350 steps (70.0s) sets <code>truncated=True</code>, ensuring correct value bootstrapping.",
        body
    ))

    story.append(Paragraph('D. Collision Detection Definition Discrepancy', heading2))
    story.append(Paragraph(
        "An important technical distinction exists regarding collision detection:<br/>"
        "&bull; <b>Simulation Training (<code>tb3_navigation_env.py</code>):</b> Collision is determined by <i>ground-truth geometric contact</i> "
        "(<code>_check_collision()</code>) against boundary walls (|x|, |y| &ge; 1.845m) and active obstacle bounding boxes using the robot's physical radius (0.105m).<br/>"
        "&bull; <b>Autonomous Inference (<code>ppo_inference_node.py</code>):</b> Ground-truth contact is inaccessible on real hardware; therefore, inference employs a "
        "<b>conservative LiDAR-range collision proxy</b> triggering emergency stops when min_range &lt; 0.16m.<br/>"
        "This discrepancy is noted as an operational limitation: while ground-truth geometry governs simulation training, autonomous deployment relies on a range proxy "
        "that may trigger in tight traversable passages without actual physical contact.",
        body
    ))

    # Table I (Caption above table)
    story.append(Spacer(1, 2))
    story.append(Paragraph('TABLE I: PPO Training & Network Hyperparameters (Frozen in Code)', table_caption))
    table_data = [
        ['Category', 'Hyperparameter', 'Value'],
        ['Network', 'Policy MLP (pi)', '[128, 128] Tanh'],
        ['', 'Value MLP (vf)', '[128, 128] Tanh'],
        ['Optimization', 'Algorithm & Learning Rate', 'PPO [1], lr = 3e-4 (constant)'],
        ['', 'Rollout & Minibatch', 'n_steps = 2048, batch_size = 256'],
        ['', 'GAE lambda & Discount', 'lambda = 0.95, gamma = 0.99'],
        ['', 'Clip fraction & Epochs', 'clip = 0.20, n_epochs = 10'],
        ['Scaling', 'VecNormalize Settings', 'norm_obs = False, norm_reward = True']
    ]
    t = Table(table_data, colWidths=[80, 160, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(t)
    story.append(Spacer(1, 3))

    # Section III
    story.append(Paragraph('III. IMPLEMENTATION AND COMPLETED WORK', heading1))
    story.append(Paragraph(
        "The software architecture runs on ROS 2 Humble [5] and Gazebo Classic 11.10.2 [7] under Ubuntu 22.04 LTS with PyTorch 2.11.0 "
        "(verified from run environment metadata), Stable-Baselines3 2.7.1 [4], and Gymnasium [9].",
        body
    ))
    story.append(Paragraph(
        "<b>Open-Source GitHub Repository:</b> All project code, simulation worlds, training pipelines, observation preprocessors, test suites, "
        "and evaluation artifacts are hosted openly at: "
        "<a href='https://github.com/Saigirish23/tb3-rl-nav' color='blue'><u>https://github.com/Saigirish23/tb3-rl-nav</u></a>. "
        "The repository contains verified implementations of the <code>TurtleBot3NavigationEnv</code>, modular domain randomizers, multi-worker PPO scripts, "
        "and the standalone ROS 2 deployment node.",
        body
    ))
    story.append(Paragraph(
        "Completed deliverables to date include:<br/>"
        "&bull; Multi-worker simulation with 4 parallel Gazebo workers and watchdog daemon recovery.<br/>"
        "&bull; <b>Policy P1 (Nominal):</b> Completed 852,816 transitions over 40,496.33 seconds (11.25 hours). The overall wall-clock throughput is "
        "<b>21.06 steps/s</b> (19.75 steps/s running average logged in run metadata).<br/>"
        "&bull; <b>Policy P4 (Combined DR):</b> Completed 704,512 transitions over 35,551.54 seconds (9.88 hours). The overall wall-clock throughput is "
        "<b>19.82 steps/s</b> (19.69 steps/s running average logged in run metadata).<br/>"
        "&bull; Asynchronous standalone ROS 2 inference node (<code>ppo_inference_node.py</code>) running deterministic inference at 5.0 Hz.<br/>"
        "&bull; Targeted test validation (11/11 passing tests across inference correctness and observation preprocessing) within a broader 69-test suite.<br/>"
        "&bull; 321k-line per-step reward trace audit across 4 baseline behaviors (reactive, random, stationary, spin).<br/>"
        "&bull; <b>Simulation-based navigation demonstration in Gazebo Classic:</b> Successfully conducted 3 consecutive multi-target navigation trials "
        "in Gazebo Classic without collisions or timeouts (3/3 successful trials).",
        body
    ))

    # Section IV
    story.append(Paragraph('IV. PRELIMINARY RESULTS AND DISCUSSION', heading1))
    story.append(Paragraph(
        "Policies P1 and P4 were benchmarked across a frozen held-out test suite of N=50 identical scenarios with static obstacles "
        "(scenarios/test/scenarios.json, SHA-256: d9a6a25c...) <b>under nominal evaluation conditions</b>.",
        body
    ))

    # Table II (Caption above table)
    story.append(Spacer(1, 2))
    story.append(Paragraph('TABLE II: Preliminary Held-Out Evaluation Results Under Nominal Conditions (N=50 Scenarios)', table_caption))
    results_data = [
        ['Metric', 'P1 (Nominal)', 'P4 (Combined DR)', 'Relative Delta'],
        ['Training Steps Completed', '852,816', '704,512', '-148k (-17.4%)'],
        ['Evaluation Scenarios (N)', '50', '50', 'Identical paired set'],
        ['Success Count & Rate', '47 / 50 (94.0%)', '49 / 50 (98.0%)', '+4.0% (+2 successes)'],
        ['95% Wilson Score CI', '[83.78%, 97.94%]', '[89.50%, 99.65%]', 'Substantial overlap'],
        ['Collision Count & Rate', '1 / 50 (2.0%)', '1 / 50 (2.0%)', '0 (Identical)'],
        ['Timeout Count & Rate', '2 / 50 (4.0%)', '0 / 50 (0.0%)', '-2 (-4.0% eliminated)'],
        ['Mean Steps to Goal (Succ.)', '106.17 +/- 48.2', '101.49 +/- 44.8', '-4.68 steps (-4.4%)'],
        ['Mean Duration to Goal (Succ.)', '21.23 +/- 9.6 s', '20.30 +/- 9.0 s', '-0.93 s (-4.4%)'],
        ['Mean Path Length (Succ.)', '2.25 +/- 0.72 m', '2.07 +/- 0.58 m', '-0.18 m (-8.0%)'],
        ['Mean Min. Clearance (Succ.)', '0.289 m', '0.298 m', '+0.009 m (+3.1%)'],
        ['Paired McNemar Exact Test', 'p = 0.6250', '(Not Significant)', 'Discordant: b=1, c=3']
    ]
    t2 = Table(results_data, colWidths=[120, 100, 105, 95])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(t2)
    story.append(Spacer(1, 3))

    # Figure 2 & 3
    story.append(Image('docs/midterm_report/ieee_final/figures/fig2_evaluation_results.png', width=490, height=155))
    story.append(Paragraph('Fig. 2. Held-out evaluation outcomes (N=50) under nominal conditions: (a) Outcome distributions; (b) Wilson 95% confidence intervals and exact McNemar test result.', fig_caption))
    story.append(Spacer(1, 2))

    story.append(Image('docs/midterm_report/ieee_final/figures/fig3_trajectory_metrics.png', width=490, height=145))
    story.append(Paragraph('Fig. 3. Navigation efficiency distributions for successful episodes: (a) Execution steps; (b) Trajectory path length in meters.', fig_caption))
    story.append(Spacer(1, 2))

    story.append(Paragraph('A. Statistical Rigor and Power Analysis Considerations', heading2))
    story.append(Paragraph(
        "A critical requirement of scientific reporting is distinguishing descriptive gains from statistical significance. On the paired test set, "
        "46 scenarios succeeded under both policies. Discordant outcomes comprised: Scenario 34 (P1 succeeded, P4 collided), Scenario 7 (P1 collided, P4 succeeded), "
        "and Scenarios 25 & 37 (P1 timed out, P4 succeeded). Evaluating these discordant pairs (b=1, c=3) via the exact two-sided binomial McNemar test yields "
        "<b>p = 0.6250 &gt;&gt; 0.05</b>. Therefore, the 4.0% descriptive difference cannot be asserted as statistically significant at N=50.<br/>"
        "<b>Sample Size Considerations:</b> Sample size requirements depend fundamentally on the test design, paired discordance rates, significance level, "
        "target power, and effect size. For an unpaired comparison detecting a 94% vs. 98% delta with 80% power at alpha=0.05, standard approximations require "
        "roughly 370 episodes per policy. Under paired testing, power is determined by the frequency of discordant pairs. Therefore, we propose N=250 as an expanded "
        "benchmark to improve estimation precision, while formal paired power analysis remains subject to ongoing investigation.",
        body
    ))

    # Section V
    story.append(Paragraph('V. LIMITATIONS AND PENDING WORK', heading1))
    story.append(Paragraph(
        "&bull; <b>Unequal Training Budgets:</b> P1 received 852k steps while P4 received 704k steps (-17.4%) due to execution deadline constraints. That P4 achieves parity "
        "despite fewer gradient steps is consistent with, but does not definitively establish, improved sample efficiency.<br/>"
        "&bull; <b>Single Seed:</b> Both models represent a single seed (s=0); inter-run variance cannot currently be quantified.<br/>"
        "&bull; <b>Simulation-Only Scope:</b> Results reflect Gazebo Classic physics. Physical robot validation has not yet been demonstrated.",
        body
    ))

    # Planned 4x4 Matrix Table
    story.append(Spacer(1, 2))
    story.append(Paragraph('TABLE III: Planned 4x4 Policy / Environment Cross-Evaluation Matrix', table_caption))
    matrix_data = [
        ['Trained Policy', 'Nominal Env.', 'Sensor DR Env.', 'Dynamics DR Env.', 'Combined DR Env.'],
        ['P1 (Nominal Baseline)', 'Completed (94%)', 'Planned', 'Planned', 'Planned'],
        ['P2 (Sensor Only)', 'Planned', 'Planned', 'Planned', 'Planned'],
        ['P3 (Dynamics Only)', 'Planned', 'Planned', 'Planned', 'Planned'],
        ['P4 (Combined DR)', 'Completed (98%)', 'Planned', 'Planned', 'Planned']
    ]
    t3 = Table(matrix_data, colWidths=[110, 85, 85, 85, 85])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(t3)
    story.append(Spacer(1, 3))

    story.append(Paragraph('A. Project Roadmap', heading2))
    story.append(Paragraph(
        "The remaining project roadmap includes:<br/>"
        "1. <b>Intermediate Ablations (P2 and P3):</b> Train P2 (sensor noise only) and P3 (dynamics only) with equalized budgets of 850k steps.<br/>"
        "2. <b>4x4 Cross-Evaluation Matrix:</b> Execute Table III evaluating each policy against nominal and perturbed conditions to assess true out-of-distribution robustness.<br/>"
        "3. <b>Expanded Evaluation (N=250):</b> Benchmark policies across 250 diverse test scenarios.<br/>"
        "4. <b>Physical Robot Deployment:</b> Validate policy P4 on a physical TurtleBot3 Burger in the laboratory, conditional on robot access, lab availability, and safety approval.",
        body
    ))

    # Section VI
    story.append(Paragraph('VI. INDIVIDUAL CONTRIBUTIONS', heading1))
    story.append(Paragraph(
        "<i>Both authors contributed to the project with complementary responsibilities. Both authors must confirm the final contribution declaration upon report submission.</i><br/>"
        "&bull; <b>Saigirish S (B24ME1066):</b> RL navigation pipeline architecture, 43-D observation preprocessor, dead-reckoning module, domain randomizers [2], [8], "
        "multi-worker Gazebo orchestration, P1 and P4 training, ROS 2 standalone inference node.<br/>"
        "&bull; <b>Siddhi Vinayak (B24BB1036):</b> Reward audit framework, per-step reward trace logging, unit test execution and baseline validation, "
        "participation in experimental design and hardware deployment preparation.",
        body
    ))

    # Section VII
    story.append(Paragraph('VII. CONCLUSION', heading1))
    story.append(Paragraph(
        "This mid-term progress report presents an initial implementation and evaluation baseline for mapless autonomous navigation using PPO [1] and domain randomization [2], [8]. "
        "By coupling sector min-pooled LiDAR readings with dead-reckoning goal tracking, the policy achieves high navigation reliability under nominal simulation conditions "
        "(94%--98% success). Methodological rigor demonstrates that observed preliminary differences between nominal and randomized policies are not statistically significant at N=50, "
        "establishing a clear requirement for expanded benchmarking and perturbed-environment cross-evaluations. The completed architecture provides a solid trajectory toward "
        "completing the planned ablation matrix and physical robot deployment.",
        body
    ))

    # References
    story.append(Spacer(1, 3))
    story.append(Paragraph('REFERENCES', heading1))
    refs = [
        "[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
        "[2] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, \"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World,\" in Proc. IEEE/RSJ IROS, 2017, pp. 23--30.",
        "[3] L. Tai, G. Paolo, and M. Liu, \"Virtual-to-Real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation,\" in Proc. IEEE/RSJ IROS, 2017, pp. 31--36.",
        "[4] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, \"Stable-Baselines3: Reliable Reinforcement Learning Implementations in Python,\" JMLR, vol. 22, no. 268, pp. 1--8, 2021.",
        "[5] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, \"Robot Operating System 2: Design, Architecture, and Uses in the Wild,\" Science Robotics, vol. 7, no. 66, p. eabm6074, 2022.",
        "[6] S. Macenski, F. Mart\u00edn, R. White, and J. G. Clavero, \"The Marathon 2: A Navigation System,\" in Proc. IEEE/RSJ IROS, 2020, pp. 2718--2725.",
        "[7] N. Koenig and A. Howard, \"Design and Use Paradigms for Gazebo, an Open-Source Multi-Robot Simulator,\" in Proc. IEEE/RSJ IROS, vol. 3, 2004, pp. 2149--2154.",
        "[8] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, \"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization,\" in Proc. IEEE ICRA, 2018, pp. 3803--3810.",
        "[9] M. Towers et al., \"Gymnasium: A Standard Interface for Reinforcement Learning Environments,\" arXiv:2407.17032, 2024."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', parent=styles['Normal'], fontName='Times-Roman', fontSize=7.2, leading=9.2, spaceAfter=2)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {filename}")

if __name__ == '__main__':
    build_pdf()
