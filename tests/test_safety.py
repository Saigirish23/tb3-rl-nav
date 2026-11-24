"""Unit Tests for tb3_rl_navigation.safety Module.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 10)
Verifies:
1. Velocity command clamping to [0, 0.15] m/s and [-0.60, 0.60] rad/s.
2. Stale-scan watchdog intervention when elapsed time exceeds 0.40 s.
3. Emergency stop intervention when LiDAR minimum range falls below 0.16 m.
4. Deadman switch override forcing immediate zero twist.
5. Teleoperation override passing supervisor commands.
6. Cumulative intervention logging and count isolation.
"""

import pytest
import numpy as np
from tb3_rl_navigation.safety import SafetyFilter, SafetyIntervention


def test_velocity_clamping():
    """Verify velocity commands are strictly clamped to [0, 0.15] m/s and [-0.60, 0.60] rad/s."""
    safety = SafetyFilter(v_min=0.0, v_max=0.15, omega_max=0.60)
    safety.notify_scan_received(scan_timestamp=100.0)

    # 1. Normal command within limits
    v, omega, intervention, _ = safety.filter_command(0.10, 0.30, current_time=100.1)
    assert intervention == SafetyIntervention.NONE
    assert np.isclose(v, 0.10)
    assert np.isclose(omega, 0.30)

    # 2. Over-speed forward velocity
    v, omega, intervention, _ = safety.filter_command(0.25, 0.10, current_time=100.1)
    assert intervention == SafetyIntervention.VELOCITY_CLAMP
    assert np.isclose(v, 0.15)
    assert np.isclose(omega, 0.10)

    # 3. Negative forward velocity (backward drive forbidden by forward-only bias)
    v, omega, intervention, _ = safety.filter_command(-0.05, 0.0, current_time=100.1)
    assert intervention == SafetyIntervention.VELOCITY_CLAMP
    assert np.isclose(v, 0.0)

    # 4. Over-speed angular velocity
    v, omega, intervention, _ = safety.filter_command(0.08, 1.20, current_time=100.1)
    assert intervention == SafetyIntervention.VELOCITY_CLAMP
    assert np.isclose(omega, 0.60)

    v, omega, intervention, _ = safety.filter_command(0.08, -0.95, current_time=100.1)
    assert intervention == SafetyIntervention.VELOCITY_CLAMP
    assert np.isclose(omega, -0.60)


def test_stale_scan_watchdog():
    """Verify watchdog halts robot if no scan has arrived within 0.40s."""
    safety = SafetyFilter(stale_scan_timeout_sec=0.40)
    safety.notify_scan_received(scan_timestamp=10.00)

    # Within timeout (elapsed = 0.20s <= 0.40s)
    v, omega, intervention, _ = safety.filter_command(0.12, 0.20, current_time=10.20)
    assert intervention == SafetyIntervention.NONE
    assert np.isclose(v, 0.12)

    # Past timeout (elapsed = 0.45s > 0.40s)
    v, omega, intervention, info = safety.filter_command(0.12, 0.20, current_time=10.45)
    assert intervention == SafetyIntervention.STALE_SCAN
    assert np.isclose(v, 0.0)
    assert np.isclose(omega, 0.0)
    assert info["status"] == "STALE_SCAN"


def test_emergency_collision_stop():
    """Verify safety layer halts robot if any valid beam is < 0.16 m."""
    safety = SafetyFilter(emergency_stop_dist=0.16)
    safety.notify_scan_received(scan_timestamp=5.0)

    # Clean scan (all ranges >= 0.50m)
    clean_ranges = np.full(360, 1.5, dtype=np.float32)
    v, omega, intervention, _ = safety.filter_command(0.10, 0.0, raw_ranges=clean_ranges, current_time=5.1)
    assert intervention == SafetyIntervention.NONE
    assert np.isclose(v, 0.10)

    # Danger scan: one beam at 0.14m < 0.16m
    danger_ranges = clean_ranges.copy()
    danger_ranges[45] = 0.14
    v, omega, intervention, info = safety.filter_command(0.10, 0.0, raw_ranges=danger_ranges, current_time=5.1)
    assert intervention == SafetyIntervention.COLLISION_STOP
    assert np.isclose(v, 0.0)
    assert np.isclose(omega, 0.0)
    assert np.isclose(info["min_range"], 0.14)

    # Invalid values (0.0, inf, NaN) ignored in collision proximity
    invalid_ranges = np.array([0.0, np.nan, np.inf, 2.0], dtype=np.float32)
    v, omega, intervention, _ = safety.filter_command(0.10, 0.0, raw_ranges=invalid_ranges, current_time=5.1)
    assert intervention == SafetyIntervention.NONE


def test_deadman_switch():
    """Verify deadman switch immediately halts robot regardless of commands."""
    safety = SafetyFilter()
    safety.notify_scan_received(scan_timestamp=1.0)
    safety.set_deadman(True)

    v, omega, intervention, _ = safety.filter_command(0.15, 0.50, current_time=1.1)
    assert intervention == SafetyIntervention.DEADMAN_STOP
    assert np.isclose(v, 0.0)
    assert np.isclose(omega, 0.0)

    # Deactivate deadman
    safety.set_deadman(False)
    v, omega, intervention, _ = safety.filter_command(0.10, 0.20, current_time=1.15)
    assert intervention == SafetyIntervention.NONE
    assert np.isclose(v, 0.10)


def test_teleoperation_override():
    """Verify teleoperation override passes human supervisor commands."""
    safety = SafetyFilter()
    safety.notify_scan_received(scan_timestamp=1.0)
    safety.set_teleop_override(True, v=-0.05, omega=0.40)

    v, omega, intervention, _ = safety.filter_command(0.15, -0.60, current_time=1.1)
    assert intervention == SafetyIntervention.TELEOP_OVERRIDE
    assert np.isclose(v, -0.05)
    assert np.isclose(omega, 0.40)


def test_intervention_counting_and_reset():
    """Verify intervention telemetry accumulation and reset."""
    safety = SafetyFilter(v_max=0.15)
    safety.notify_scan_received(scan_timestamp=1.0)

    safety.filter_command(0.30, 0.0, current_time=1.1)  # Clamp
    safety.filter_command(0.10, 0.0, current_time=1.8)  # Stale scan
    assert safety.total_interventions == 2
    assert safety.intervention_counts[SafetyIntervention.VELOCITY_CLAMP] == 1
    assert safety.intervention_counts[SafetyIntervention.STALE_SCAN] == 1

    safety.reset()
    assert safety.total_interventions == 0
    assert len(safety.intervention_records) == 0

