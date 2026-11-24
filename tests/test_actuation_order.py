#!/usr/bin/env python3
"""Unit tests verifying Case 8 Actuation Ordering and Numerical Bounds.

Actuation pipeline order:
1. Control delay d in [0, 0.20] s (weighted previous action hold over control step dt = 0.20 s)
2. Command gain (k_v in [0.85, 1.15], k_omega in [0.85, 1.15]; Case 8 stress: k_v=0.80, k_omega=1.20)
3. Velocity saturation / motor boundary clipping ([0.0, 0.15] m/s, [-0.60, 0.60] rad/s)
4. Acceleration limiting (|delta_v| <= a_lin * dt, |delta_omega| <= a_ang * dt)
5. First-order actuator lag (alpha = dt / (tau + dt))
6. Final motor boundary clamping
"""

import math
import numpy as np
import pytest

from rl.env.randomizers import DynamicsRandomizer, DynamicsParams


def test_actuation_pipeline_order_exact_step():
    """Verify each stage of the actuation pipeline against exact step-by-step arithmetic."""
    dt = 0.20
    dyn = DynamicsRandomizer(seed=42)
    dyn.params = DynamicsParams(
        control_delay_s=0.0,       # No delay for isolating stages 2-5
        v_command_gain=1.20,       # Overshoot gain: 0.15 * 1.20 = 0.18
        omega_command_gain=1.20,   # Overshoot gain: 0.60 * 1.20 = 0.72
        linear_acc_limit=0.20,     # max dv = 0.20 * 0.20 = 0.04 m/s
        angular_acc_limit=1.00,    # max dw = 1.00 * 0.20 = 0.20 rad/s
        actuator_lag_tau=0.30,     # alpha = 0.20 / (0.30 + 0.20) = 0.40
    )
    dyn.reset()

    # Step 1: Commanded max velocity from rest (v=0.15, w=0.60)
    # Stage 1: target_v = 0.15, target_w = 0.60
    # Stage 2 (Gain): target_v = 0.18, target_w = 0.72
    # Stage 3 (Saturation): target_v = 0.15, target_w = 0.60
    # Stage 4 (Accel Limit): dv = min(0.15 - 0.0, 0.04) = 0.04 -> v_lim = 0.04
    #                        dw = min(0.60 - 0.0, 0.20) = 0.20 -> w_lim = 0.20
    # Stage 5 (Actuator Lag): v_eff = 0.60 * 0.0 + 0.40 * 0.04 = 0.016 m/s
    #                         w_eff = 0.60 * 0.0 + 0.40 * 0.20 = 0.080 rad/s
    v1, w1 = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.0, prev_cmd_omega=0.0, dt=dt)
    assert math.isclose(v1, 0.016, abs_tol=1e-5), f"Expected v1=0.016, got {v1}"
    assert math.isclose(w1, 0.080, abs_tol=1e-5), f"Expected w1=0.080, got {w1}"

    # Step 2: Commanded max velocity again
    # Stage 1: target_v = 0.15, target_w = 0.60
    # Stage 2: target_v = 0.18, target_w = 0.72
    # Stage 3: target_v = 0.15, target_w = 0.60
    # Stage 4: dv = min(0.15 - 0.016, 0.04) = 0.04 -> v_lim = 0.016 + 0.04 = 0.056
    #          dw = min(0.60 - 0.080, 0.20) = 0.20 -> w_lim = 0.080 + 0.20 = 0.280
    # Stage 5: v_eff = 0.60 * 0.016 + 0.40 * 0.056 = 0.0096 + 0.0224 = 0.032 m/s
    #          w_eff = 0.60 * 0.080 + 0.40 * 0.280 = 0.048 + 0.112 = 0.160 rad/s
    v2, w2 = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.15, prev_cmd_omega=0.60, dt=dt)
    assert math.isclose(v2, 0.032, abs_tol=1e-5), f"Expected v2=0.032, got {v2}"
    assert math.isclose(w2, 0.160, abs_tol=1e-5), f"Expected w2=0.160, got {w2}"


def test_control_delay_stage():
    """Verify that control delay d = 0.20s holds the previous action completely."""
    dt = 0.20
    dyn = DynamicsRandomizer(seed=42)
    dyn.params = DynamicsParams(
        control_delay_s=0.20,      # Full 1-step delay
        v_command_gain=1.00,
        omega_command_gain=1.00,
        linear_acc_limit=10.0,     # Unconstrained accel
        angular_acc_limit=10.0,
        actuator_lag_tau=0.0,      # No lag
    )
    dyn.reset()

    # If previous command was 0.0 and current command is 0.15:
    # With full delay d = dt = 0.20, frac_prev = 1.0 -> target = 0.0
    v, w = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.0, prev_cmd_omega=0.0, dt=dt)
    assert v == 0.0, f"Expected 0.0 under full delay, got {v}"
    assert w == 0.0, f"Expected 0.0 under full delay, got {w}"

    # Now with previous command 0.10: target should be 0.10
    v, w = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.10, prev_cmd_omega=-0.20, dt=dt)
    assert math.isclose(v, 0.10, abs_tol=1e-5)
    assert math.isclose(w, -0.20, abs_tol=1e-5)


def test_velocity_saturation_bounds():
    """Verify motor velocity clipping to [0.0, 0.15] m/s and [-0.60, 0.60] rad/s."""
    dt = 0.20
    dyn = DynamicsRandomizer(seed=42)
    # Huge gains that would blow past physical limits if not saturated
    dyn.params = DynamicsParams(
        v_command_gain=5.00,
        omega_command_gain=5.00,
        control_delay_s=0.0,
        linear_acc_limit=10.0,
        angular_acc_limit=10.0,
        actuator_lag_tau=0.0,
    )
    dyn.reset()

    v, w = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.0, prev_cmd_omega=0.0, dt=dt)
    assert v <= 0.15, f"Linear velocity violated saturation limit: {v}"
    assert w <= 0.60, f"Angular velocity violated saturation limit: {w}"

    # Negative linear velocity command clipped to 0.0 (non-holonomic forward drive)
    v_neg, w_neg = dyn.perturb_action(cmd_v=-0.10, cmd_omega=-1.0, prev_cmd_v=0.0, prev_cmd_omega=0.0, dt=dt)
    assert v_neg == 0.0, f"Linear velocity below zero: {v_neg}"
    assert w_neg >= -0.60, f"Angular velocity below -0.60: {w_neg}"


def test_case8_actuation_parameters():
    """Verify exact numerical bounds and parameters for Case 8 OOD Actuation Lag."""
    dyn = DynamicsRandomizer(seed=100)
    dyn.params = DynamicsParams(
        control_delay_s=0.20,
        actuator_lag_tau=0.30,
        linear_acc_limit=0.20,
        angular_acc_limit=1.00,
        v_command_gain=0.80,
        omega_command_gain=1.20,
    )
    dyn.reset()

    assert dyn.params.control_delay_s == 0.20
    assert dyn.params.actuator_lag_tau == 0.30
    assert dyn.params.linear_acc_limit == 0.20
    assert dyn.params.angular_acc_limit == 1.00
    assert dyn.params.v_command_gain == 0.80
    assert dyn.params.omega_command_gain == 1.20

    # Step over 10 steps to verify stability and convergence toward (0.80 * 0.15 = 0.12, 0.60)
    for _ in range(25):
        v, w = dyn.perturb_action(cmd_v=0.15, cmd_omega=0.60, prev_cmd_v=0.15, prev_cmd_omega=0.60, dt=0.20)

    # Commanded 0.15 * 0.80 = 0.12 m/s
    assert math.isclose(v, 0.12, abs_tol=1e-3)
    # Commanded 0.60 * 1.20 = 0.72 -> saturated at 0.60 rad/s
    assert math.isclose(w, 0.60, abs_tol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", str(__file__)])
