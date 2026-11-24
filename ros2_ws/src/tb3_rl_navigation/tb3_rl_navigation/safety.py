"""Independent Safety Layer for TurtleBot3 Deployment and SITL.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 10)
Design invariants:
- Independent of the policy; active in SITL and hardware only, NEVER in training.
- Velocity clamps: v in [0.0, 0.15] m/s, omega in [-0.60, 0.60] rad/s.
- Stale-scan watchdog: timeout after ~2 scan periods (default 0.40 s) -> zero twist.
- Emergency stop: raw LiDAR range < emergency threshold (default 0.16 m) -> zero twist.
- Deadman / teleoperation override: allows human supervisor takeover.
- Interventions: logged with timestamp, reason, and counted as evaluation failures.
- Thresholds are fixed before trials and NEVER tuned after seeing results.
"""

from __future__ import annotations
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


class SafetyIntervention(str, Enum):
    """Reason for safety intervention."""
    NONE = "NONE"
    VELOCITY_CLAMP = "VELOCITY_CLAMP"
    STALE_SCAN = "STALE_SCAN"
    COLLISION_STOP = "COLLISION_STOP"
    DEADMAN_STOP = "DEADMAN_STOP"
    TELEOP_OVERRIDE = "TELEOP_OVERRIDE"


@dataclass
class InterventionRecord:
    """Telemetry record for an individual safety intervention."""
    timestamp: float
    step: int
    intervention_type: SafetyIntervention
    raw_v: float
    raw_omega: float
    safe_v: float
    safe_omega: float
    details: str = ""


class SafetyFilter:
    """Safety supervisor filtering policy velocity commands before execution."""

    def __init__(
        self,
        v_min: float = 0.0,
        v_max: float = 0.15,
        omega_max: float = 0.60,
        stale_scan_timeout_sec: float = 0.40,  # ~2 scan periods at 5 Hz
        emergency_stop_dist: float = 0.16,     # meters (footprint 0.105m + margin)
    ):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.stale_scan_timeout_sec = float(stale_scan_timeout_sec)
        self.emergency_stop_dist = float(emergency_stop_dist)

        self.last_scan_time: Optional[float] = None
        self.deadman_active: bool = False
        self.teleop_active: bool = False
        self.teleop_v: float = 0.0
        self.teleop_omega: float = 0.0

        self.step_count: int = 0
        self.intervention_records: List[InterventionRecord] = []
        self.intervention_counts: Dict[SafetyIntervention, int] = {
            k: 0 for k in SafetyIntervention
        }

    def reset(self) -> None:
        """Reset safety layer state at episode/trial boundary."""
        self.last_scan_time = None
        self.deadman_active = False
        self.teleop_active = False
        self.teleop_v = 0.0
        self.teleop_omega = 0.0
        self.step_count = 0
        self.intervention_records.clear()
        self.intervention_counts = {k: 0 for k in SafetyIntervention}

    def notify_scan_received(self, scan_timestamp: Optional[float] = None) -> None:
        """Update last scan arrival time."""
        if scan_timestamp is not None:
            self.last_scan_time = float(scan_timestamp)
        else:
            self.last_scan_time = time.monotonic()

    def set_deadman(self, active: bool) -> None:
        """Set deadman switch state (True = kill switch active -> robot stops)."""
        self.deadman_active = bool(active)

    def set_teleop_override(self, active: bool, v: float = 0.0, omega: float = 0.0) -> None:
        """Set teleoperation override command."""
        self.teleop_active = bool(active)
        self.teleop_v = float(v)
        self.teleop_omega = float(omega)

    def filter_command(
        self,
        cmd_v: float,
        cmd_omega: float,
        raw_ranges: Optional[np.ndarray] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[float, float, SafetyIntervention, Dict[str, Any]]:
        """Filter policy velocity commands against safety criteria.

        Args:
            cmd_v: Commanded linear velocity (m/s).
            cmd_omega: Commanded angular velocity (rad/s).
            raw_ranges: Array of raw LiDAR ranges (optional, for collision check).
            current_time: Current monotonic or simulation time (s).

        Returns:
            (safe_v, safe_omega, intervention_type, info_dict)
        """
        self.step_count += 1
        now = float(current_time) if current_time is not None else time.monotonic()

        # 1. Deadman switch check (Highest priority hardware kill switch)
        if self.deadman_active:
            self._record(now, SafetyIntervention.DEADMAN_STOP, cmd_v, cmd_omega, 0.0, 0.0, "Deadman active")
            return 0.0, 0.0, SafetyIntervention.DEADMAN_STOP, {"status": "DEADMAN_TRIGGERED"}

        # 2. Teleoperation manual takeover
        if self.teleop_active:
            self._record(now, SafetyIntervention.TELEOP_OVERRIDE, cmd_v, cmd_omega, self.teleop_v, self.teleop_omega, "Teleop override active")
            return self.teleop_v, self.teleop_omega, SafetyIntervention.TELEOP_OVERRIDE, {"status": "TELEOP_OVERRIDE"}

        # 3. Stale-scan watchdog check
        if self.last_scan_time is not None:
            time_since_scan = now - self.last_scan_time
            if time_since_scan > self.stale_scan_timeout_sec:
                details = f"Stale scan ({time_since_scan:.3f}s > {self.stale_scan_timeout_sec:.3f}s)"
                self._record(now, SafetyIntervention.STALE_SCAN, cmd_v, cmd_omega, 0.0, 0.0, details)
                return 0.0, 0.0, SafetyIntervention.STALE_SCAN, {"status": "STALE_SCAN", "dt": time_since_scan}

        # 4. Emergency collision proximity stop
        if raw_ranges is not None and len(raw_ranges) > 0:
            arr = np.asarray(raw_ranges, dtype=np.float32)
            valid = arr[(~np.isnan(arr)) & (~np.isinf(arr)) & (arr > 0.0)]
            min_r = float(np.min(valid)) if len(valid) > 0 else float("inf")
            if min_r < self.emergency_stop_dist:
                details = f"Collision stop (min_range={min_r:.3f}m < {self.emergency_stop_dist:.3f}m)"
                self._record(now, SafetyIntervention.COLLISION_STOP, cmd_v, cmd_omega, 0.0, 0.0, details)
                return 0.0, 0.0, SafetyIntervention.COLLISION_STOP, {"status": "COLLISION_STOP", "min_range": min_r}

        # 5. Velocity limit clamping
        clamped_v = float(np.clip(cmd_v, self.v_min, self.v_max))
        clamped_omega = float(np.clip(cmd_omega, -self.omega_max, self.omega_max))

        if abs(clamped_v - cmd_v) > 1e-4 or abs(clamped_omega - cmd_omega) > 1e-4:
            details = f"Clamped v:[{cmd_v:.3f}->{clamped_v:.3f}] w:[{cmd_omega:.3f}->{clamped_omega:.3f}]"
            self._record(now, SafetyIntervention.VELOCITY_CLAMP, cmd_v, cmd_omega, clamped_v, clamped_omega, details)
            return clamped_v, clamped_omega, SafetyIntervention.VELOCITY_CLAMP, {"status": "CLAMPED"}

        return clamped_v, clamped_omega, SafetyIntervention.NONE, {"status": "NOMINAL"}

    def _record(
        self,
        timestamp: float,
        intervention: SafetyIntervention,
        raw_v: float,
        raw_omega: float,
        safe_v: float,
        safe_omega: float,
        details: str = "",
    ) -> None:
        """Internal recorder for telemetry."""
        self.intervention_counts[intervention] += 1
        rec = InterventionRecord(
            timestamp=timestamp,
            step=self.step_count,
            intervention_type=intervention,
            raw_v=raw_v,
            raw_omega=raw_omega,
            safe_v=safe_v,
            safe_omega=safe_omega,
            details=details,
        )
        self.intervention_records.append(rec)

    @property
    def total_interventions(self) -> int:
        """Total number of safety interventions triggered."""
        return sum(
            count for k, count in self.intervention_counts.items()
            if k != SafetyIntervention.NONE
        )

