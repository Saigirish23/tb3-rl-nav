#!/usr/bin/env python3
"""Build publication-quality PDF report matching IEEE conference layout."""

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
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#555555'))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, 'IEEE Mid-Term Progress Report: Sim-to-Real RL LiDAR Navigation')
            self.drawRightString(558, 750, 'IIT Jodhpur (Oct 2026)')
            self.setStrokeColor(colors.HexColor('#cccccc'))
            self.setLineWidth(0.5)
            self.line(54, 745, 558, 745)
            
        # Footer
        page_text = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, 'tb3-rl-nav: Saigirish S & Siddhi Vinayak')
        self.setStrokeColor(colors.HexColor('#cccccc'))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()

def build_pdf(filename='docs/midterm_report/paper/midterm_report.pdf'):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1, # Centered
        textColor=colors.HexColor('#111111'),
        spaceAfter=8
    )
    
    author_style = ParagraphStyle(
        'AuthorBlock',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#222222'),
        spaceAfter=12
    )

    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11.5,
        alignment=4, # Justified
        textColor=colors.HexColor('#222222'),
        spaceAfter=8
    )
    
    heading1 = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#002B49'), # Deep Navy
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    heading2 = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1976D2'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        alignment=4, # Justified
        textColor=colors.HexColor('#111111'),
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#444444'),
        spaceBefore=3,
        spaceAfter=8
    )

    story = []
    
    # Title
    story.append(Paragraph('Sim-to-Real Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation: Mid-Term Progress Report', title_style))
    
    # Author Block
    authors_text = (
        "<b>Saigirish S</b> (Roll No: B24ME1066, Email: b24me1066@iitj.ac.in) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Siddhi Vinayak</b> (Roll No: B24BB1036, Email: b24bb1036@iitj.ac.in)<br/>"
        "<i>Indian Institute of Technology Jodhpur</i><br/>"
        "<b>Contribution Declaration:</b> <i>Both authors contributed equally to the project; their responsibilities were complementary.</i>"
    )
    story.append(Paragraph(authors_text, author_style))
    story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#002B49'), spaceBefore=2, spaceAfter=8))
    
    # Abstract
    story.append(Paragraph(
        "<b><i>Abstract</i></b>---Mapless navigation in dynamic or unstructured environments is a foundational challenge in mobile robotics. "
        "While deep reinforcement learning (DRL) provides an end-to-end framework directly mapping onboard sensor observations to continuous velocity commands, "
        "standard policies trained in idealized physics engines frequently suffer catastrophic failure when deployed to real hardware due to the reality gap. "
        "In this mid-term progress report, we investigate a domain-randomization-enhanced Proximal Policy Optimization (PPO) pipeline for a differential-drive "
        "TurtleBot3 mobile robot using a 2D planar LiDAR. We detail the system architecture, comprising a 43-dimensional observation vector with forward-centered "
        "sector min-pooling and dead-reckoning pose tracking, safe action scaling, and a multi-component reward formulation. We present preliminary empirical results "
        "on a held-out test suite of N=50 identical scenarios comparing a nominal policy (P1, 852k steps) against a combined sensor-and-dynamics randomized policy "
        "(P4, 704k steps). On nominal evaluation, P1 achieves a 94.0% success rate while P4 achieves 98.0%. A paired McNemar exact test yields p = 0.6250, indicating "
        "that the observed 4% difference is not statistically significant at N=50 and highlighting critical sample-size requirements. We conclude with a discussion "
        "of experimental limitations, unequal training budgets, and an explicit roadmap toward completing intermediate ablations (P2, P3) and real-robot deployment.",
        abstract_text
    ))
    
    story.append(Paragraph(
        "<b><i>Index Terms</i></b>---Reinforcement Learning, Mobile Robot Navigation, Sim-to-Real Transfer, Domain Randomization, "
        "Proximal Policy Optimization, TurtleBot3, ROS 2, Gazebo.",
        abstract_text
    ))
    story.append(Spacer(1, 4))
    
    # Section 1
    story.append(Paragraph('1. Introduction and Motivation', heading1))
    story.append(Paragraph(
        "Autonomous ground vehicles operating in GPS-denied indoor environments typically rely on the classical navigation stack consisting of simultaneous "
        "localization and mapping (SLAM), global path planning (A*/Dijkstra), and local trajectory optimization (Dynamic Window Approach / TEB). While effective "
        "in static structured spaces, classical pipelines require substantial computational overhead for real-time dense map maintenance and are notoriously sensitive "
        "to localization drift, dynamic obstacle occlusion, and sensor noise.",
        body
    ))
    story.append(Paragraph(
        "To overcome these constraints, mapless navigation via Deep Reinforcement Learning (DRL) has emerged as a promising alternative. By training neural network "
        "policies to directly map raw range findings and relative goal coordinates to low-level motor velocity commands, the system avoids explicit global mapping. "
        "However, training policies directly on physical robotic platforms is prohibitively expensive, wear-intensive, and dangerous during early exploration. "
        "High-fidelity simulators such as Gazebo provide safe, accelerated experience collection, but policies trained in nominal simulation often experience dramatic "
        "performance degradation when transferred to physical hardware---the sim-to-real gap.",
        body
    ))
    story.append(Paragraph(
        "Domain Randomization (DR) bridges this gap by exposing the agent during training to a broad distribution of simulated sensor distortions and unmodeled "
        "dynamic variations, compelling the policy to learn robust feedback strategies that generalize to real-world physics. This project systematically investigates "
        "the quantitative impact of isolated and combined domain randomization on policy robustness, sample efficiency, and hardware transferability for TurtleBot3 "
        "LiDAR navigation.",
        body
    ))

    # Section 2
    story.append(Paragraph('2. Problem Statement and Objectives', heading1))
    story.append(Paragraph(
        "We consider a differential-drive TurtleBot3 Burger robot navigating from an arbitrary starting pose to a static Euclidean target in an obstacle-cluttered "
        "4m x 4m bounded workspace. Navigation must proceed maplessly, relying solely on an onboard 2D laser scanner (LDS-01) and wheel odometry. The core objectives are:",
        body
    ))
    story.append(Paragraph(
        "&bull; <b>Mapless PPO Navigation Architecture:</b> Design an end-to-end continuous actor-critic navigation framework operating at 5.0 Hz that enforces "
        "obstacle clearance and smooth kinematic control.<br/>"
        "&bull; <b>2x2 Domain Randomization Factorial Design:</b> Implement modular sensor perturbations (range noise, dropouts, latency) and dynamics perturbations "
        "(actuator lag, delay, acceleration limits) to isolate their individual and combined effects across four policy regimes: P1 (Nominal baseline), P2 (Sensor only), "
        "P3 (Dynamics only), and P4 (Combined DR).<br/>"
        "&bull; <b>Rigorous Empirical Benchmarking:</b> Evaluate trained policies across identical held-out scenarios using paired statistical hypothesis testing and "
        "confidence intervals.<br/>"
        "&bull; <b>Physical Sim-to-Real Transfer:</b> Validate the trained policy without fine-tuning on physical TurtleBot3 hardware in a laboratory environment.",
        body
    ))

    # Section 3
    story.append(Paragraph('3. Proposed System Architecture', heading1))
    story.append(Paragraph(
        "The system pipeline (Fig. 1) transforms raw sensor signals into continuous velocity commands at 5.0 Hz with zero SLAM or global path planner dependencies.",
        body
    ))
    
    # Image 1: System Architecture
    story.append(Spacer(1, 2))
    story.append(Image('docs/midterm_report/paper/figures/system_architecture.png', width=490, height=150))
    story.append(Paragraph('Fig. 1. End-to-end system architecture for mapless LiDAR navigation using PPO and dead-reckoning pose tracking.', caption_style))
    story.append(Spacer(1, 2))

    story.append(Paragraph('3.1 Observation Space Formulation', heading2))
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

    story.append(Paragraph('3.2 Action Space and Velocity Scaling', heading2))
    story.append(Paragraph(
        "The policy outputs a normalized action u in [-1, 1]^2, which is scaled into physical commands via:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>v = 0.075 * (u_0 + 1.0) &isin; [0.0, 0.15] m/s</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>&omega; = 0.60 * u_1 &isin; [-0.60, 0.60] rad/s</b><br/>"
        "This constrains the robot to purely forward motion (v &ge; 0), preventing collisions into the rear chassis blind spot.",
        body
    ))

    story.append(Paragraph('3.3 Reward Formulation and Gymnasium Semantics', heading2))
    story.append(Paragraph(
        "The step reward function is formulated as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>R_t = 10.0 * (d_{t-1} - d_t) - 0.05 + 100 * [goal] - 100 * [collision]</b><br/>"
        "Reaching the goal (dist &le; 0.20m) or colliding (min_range &le; 0.16m) sets <code>terminated=True</code>. Exceeding 350 steps (70.0s) sets "
        "<code>truncated=True</code>, ensuring correct value bootstrapping.",
        body
    ))

    # Table 1: Hyperparameters
    table_data = [
        ['Category', 'Hyperparameter', 'Value'],
        ['Network', 'Policy MLP (pi)', '[128, 128] Tanh'],
        ['', 'Value MLP (vf)', '[128, 128] Tanh'],
        ['Optimization', 'Algorithm & Learning Rate', 'PPO, lr = 3e-4 (constant)'],
        ['', 'Rollout & Minibatch', 'n_steps = 2048, batch_size = 256'],
        ['', 'GAE lambda & Discount', 'lambda = 0.95, gamma = 0.99'],
        ['', 'Clip fraction & Epochs', 'clip = 0.20, n_epochs = 10'],
        ['Scaling', 'VecNormalize Settings', 'norm_obs = False, norm_reward = True']
    ]
    t = Table(table_data, colWidths=[80, 160, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(Spacer(1, 4))
    story.append(t)
    story.append(Paragraph('TABLE I: PPO Training & Network Hyperparameters (Frozen in Code)', caption_style))
    story.append(Spacer(1, 4))

    # Section 4
    story.append(Paragraph('4. Domain Randomization Methodology', heading1))
    story.append(Paragraph(
        "Modular randomizers sample physical perturbations per episode:<br/>"
        "&bull; <b>Sensor Randomization (P2/P4):</b> Distance-dependent range noise (sigma = a + b*r), individual beam dropout (p &le; 0.05), "
        "contiguous block dropouts (1--2 blocks), delivery latency (0 or 0.20s), and odometry scale/bias drift.<br/>"
        "&bull; <b>Dynamics Randomization (P3/P4):</b> Command gains (0.85--1.15), first-order actuator lag (tau &le; 0.30s), control transport delay "
        "(d &le; 0.20s), and acceleration bounds.<br/>"
        "&bull; <b>Rejection Constraint:</b> Enforces sensor latency + control delay &le; 0.40s in P4 to guarantee physical traversability.",
        body
    ))

    # Section 5
    story.append(Paragraph('5. Implementation and Completed Work', heading1))
    story.append(Paragraph(
        "The stack runs on ROS 2 Humble and Gazebo Classic 11.10.2 under Ubuntu 22.04 LTS with PyTorch 2.11.0 and Stable-Baselines3 2.7.1. Completed deliverables include:<br/>"
        "&bull; Multi-worker simulation with 4 parallel Gazebo workers and watchdog daemon recovery.<br/>"
        "&bull; Policy P1 (Nominal): Completed 852,816 transitions over 11.25 hours (19.75 steps/s).<br/>"
        "&bull; Policy P4 (Combined DR): Completed 704,512 transitions over 9.88 hours (19.69 steps/s).<br/>"
        "&bull; Asynchronous standalone ROS 2 inference node (<code>ppo_inference_node.py</code>) running deterministic inference at 5.0 Hz.<br/>"
        "&bull; Comprehensive test harnesses with 11/11 passing unit tests and 321k-line reward audit traces across 4 baseline behaviors.<br/>"
        "&bull; Hardware-in-the-loop qualitative demonstration in Gazebo Classic (3/3 multi-target successes, 0 collisions).",
        body
    ))

    # Section 6
    story.append(Paragraph('6. Preliminary Results and Discussion', heading1))
    story.append(Paragraph(
        "Policies P1 and P4 were benchmarked across a frozen held-out test suite of N=50 identical scenarios (scenarios/test/scenarios.json, SHA-256: d9a6a25c...).",
        body
    ))

    # Table 2: Evaluation Results
    results_data = [
        ['Metric', 'P1 (Nominal)', 'P4 (Combined DR)', 'Relative Delta'],
        ['Training Steps Completed', '852,816', '704,512', '-148k (-17.4%)'],
        ['Evaluation Scenarios (N)', '50', '50', 'Identical paired set'],
        ['Success Count & Rate', '47 / 50 (94.0%)', '49 / 50 (98.0%)', '+4.0% (+2 successes)'],
        ['95% Wilson Score CI', '[83.45%, 97.97%]', '[89.51%, 99.65%]', 'Substantial overlap'],
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
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(Spacer(1, 4))
    story.append(t2)
    story.append(Paragraph('TABLE II: Held-Out Evaluation Results Across N=50 Identical Test Scenarios', caption_style))
    story.append(Spacer(1, 4))

    # Figure 2: Results plot
    story.append(Image('docs/midterm_report/paper/figures/evaluation_results.png', width=490, height=170))
    story.append(Paragraph('Fig. 2. Held-out evaluation outcomes (N=50): (a) Outcome distributions; (b) Wilson 95% confidence intervals and exact McNemar test result.', caption_style))
    story.append(Spacer(1, 4))

    # Figure 3: Boxplots
    story.append(Image('docs/midterm_report/paper/figures/trajectory_metrics.png', width=490, height=160))
    story.append(Paragraph('Fig. 3. Navigation efficiency distributions for successful episodes: (a) Execution steps; (b) Trajectory path length in meters.', caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph('6.1 Statistical Rigor and Interpretation', heading2))
    story.append(Paragraph(
        "A critical requirement of scientific reporting is distinguishing descriptive gains from statistical significance. On the paired test set, "
        "46 scenarios succeeded under both policies. Discordant outcomes comprised: Scenario 34 (P1 succeeded, P4 collided), Scenario 7 (P1 collided, P4 succeeded), "
        "and Scenarios 25 & 37 (P1 timed out, P4 succeeded). Evaluating these discordant pairs (b=1, c=3) via the exact two-sided binomial McNemar test yields "
        "<b>p = 0.6250 &gt;&gt; 0.05</b>. Therefore, the 4.0% difference cannot be asserted as statistically significant at N=50. Achieving adequate statistical power "
        "(1 - beta &ge; 0.80) requires scaling the benchmark to N &ge; 250 episodes.",
        body
    ))

    story.append(Paragraph('6.2 Methodological Limitations and Confounders', heading2))
    story.append(Paragraph(
        "&bull; <b>Unequal Training Budgets:</b> P1 received 852k steps while P4 received 704k steps (-17.4%) due to execution deadline constraints. That P4 achieves parity "
        "despite fewer gradient steps indicates sample efficiency, but the budget asymmetry remains an uncontrolled variable.<br/>"
        "&bull; <b>Single Seed:</b> Both models represent a single seed (s=0); training variance across runs cannot currently be estimated.<br/>"
        "&bull; <b>Simulation-Only Scope:</b> Results reflect Gazebo Classic physics. Hardware deployment remains future work.",
        body
    ))

    # Section 7
    story.append(Paragraph('7. Work Remaining and Project Plan', heading1))
    story.append(Paragraph(
        "The remaining post-midterm roadmap includes:<br/>"
        "1. <b>Intermediate Ablations (P2 and P3):</b> Train P2 (sensor noise only) and P3 (dynamics only) with equalized budgets of 850k steps to isolate parameter impacts.<br/>"
        "2. <b>Scaled Evaluation (N=250):</b> Benchmark all 4 policies on 250 diverse test scenarios to achieve statistical power.<br/>"
        "3. <b>Physical Robot Deployment:</b> Validate policy P4 on a physical TurtleBot3 Burger in the laboratory, measuring hardware success rates and odometry drift tolerance.",
        body
    ))

    # Section 8
    story.append(Paragraph('8. Individual Contributions', heading1))
    story.append(Paragraph(
        "<i>Both authors contributed equally to the project; their responsibilities were complementary.</i><br/>"
        "&bull; <b>Saigirish S (B24ME1066):</b> RL navigation pipeline architecture, 43-D observation preprocessor, dead-reckoning module, domain randomizers, "
        "multi-worker Gazebo orchestration, P1 and P4 training, ROS 2 standalone inference node.<br/>"
        "&bull; <b>Siddhi Vinayak (B24BB1036):</b> Reward audit framework, per-step reward trace logging, baseline policy evaluation, unit test suites, "
        "statistical hypothesis testing (McNemar test), and physical hardware deployment preparation.",
        body
    ))

    # Section 9
    story.append(Paragraph('9. Conclusion', heading1))
    story.append(Paragraph(
        "This mid-term report presents an audited, reproducible foundation for mapless autonomous navigation using PPO and domain randomization. "
        "By coupling sector min-pooled LiDAR readings with dead-reckoning goal tracking, the policy achieves high navigation reliability in simulation "
        "(94%--98% success). Methodological rigor demonstrates that observed preliminary differences are not yet statistically significant at N=50, establishing "
        "a clear requirement for expanded benchmarking. The completed architecture provides a solid trajectory toward completing the planned ablation matrix and physical robot deployment.",
        body
    ))

    # References
    story.append(Spacer(1, 4))
    story.append(Paragraph('References', heading1))
    refs = [
        "[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
        "[2] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, \"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World,\" in Proc. IEEE/RSJ IROS, 2017, pp. 23--30.",
        "[3] L. Tai, G. Paolo, and M. Liu, \"Virtual-to-Real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation,\" in Proc. IEEE/RSJ IROS, 2017, pp. 31--36.",
        "[4] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, \"Stable-Baselines3: Reliable Reinforcement Learning Implementations in Python,\" JMLR, vol. 22, no. 268, pp. 1--8, 2021.",
        "[5] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, \"Robot Operating System 2: Design, Architecture, and Uses in the Wild,\" Science Robotics, vol. 7, no. 66, p. eabm6074, 2022.",
        "[6] N. Koenig and A. Howard, \"Design and Use Paradigms for Gazebo, an Open-Source Multi-Robot Simulator,\" in Proc. IEEE/RSJ IROS, vol. 3, 2004, pp. 2149--2154.",
        "[7] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, \"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization,\" in Proc. IEEE ICRA, 2018, pp. 3803--3810.",
        "[8] M. Towers et al., \"Gymnasium: A Standard Interface for Reinforcement Learning Environments,\" arXiv:2407.17032, 2024."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', parent=styles['Normal'], fontName='Times-Roman', fontSize=7.5, leading=10, spaceAfter=2)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {filename}")

if __name__ == '__main__':
    build_pdf()
