"""Domain Randomization Modules for TurtleBot3 Navigation.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 6)
- P1: Nominal (no randomizers)
- P2: Sensor / observation randomization
- P3: Dynamics randomization
- P4: Combined (marginal P2 + P3, rejection constraint: latency + delay <= 0.40s)

Design invariants:
- Independent RNG stream so randomizer draws never alter task layout seeds.
- Zero ground truth leakage into policy observations.
- All drawn parameters logged per episode.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

from rl.env.observation import RANGE_MAX, wrap_to_pi

RANGE_MIN = 0.12  # TurtleBot3 LDS-01 minimum physical range (m)


@dataclass
class SensorParams:
    """Per-episode sampled sensor randomization parameters."""
    range_noise_a: float = 0.0          # noise intercept (m)
    range_noise_b: float = 0.0          # noise slope
    beam_dropout_p: float = 0.0         # random beam dropout prob
    block_dropout_prob: float = 0.0     # block dropout activation prob
    num_blocks: int = 0                 # 1 or 2 blocks
    block_length: int = 0               # 3 to 10 beams
    sensor_delivery_latency_scans: int = 0  # 0 or 1 extra scan delay (0.0 or 0.20s)
    mounting_yaw_offset: float = 0.0    # radians
    mounting_x_offset: float = 0.0      # meters
    mounting_y_offset: float = 0.0      # meters
    odometry_v_scale: float = 1.0       # scale factor on measured v
    odometry_yaw_rate_bias: float = 0.0 # rad/s bias on measured omega
    vel_noise_sigma_v: float = 0.0      # m/s
    vel_noise_sigma_omega: float = 0.0  # rad/s


@dataclass
class DynamicsParams:
    """Per-episode sampled dynamics randomization parameters."""
    v_command_gain: float = 1.0         # multiplier on linear velocity cmd
    omega_command_gain: float = 1.0     # multiplier on angular velocity cmd
    actuator_lag_tau: float = 0.0       # first-order lag time constant (s)
    control_delay_s: float = 0.0        # control delay (s), holds previous action
    linear_acc_limit: float = 0.60      # m/s^2
    angular_acc_limit: float = 3.00     # rad/s^2
    friction_multiplier: float = 1.00   # multiplier on surface friction
    mass_multiplier: float = 1.00       # multiplier on robot mass


class SensorRandomizer:
    """Applies sensor and observation perturbations (P2/P4)."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.params = SensorParams()
        self.scan_delay_buffer: List[np.ndarray] = []
        self._is_pinned_corner: bool = False

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, corner: bool = False, corner_sign: int = 1) -> SensorParams:
        """Sample per-episode sensor parameters.

        Args:
            corner: If True, set parameters to extreme boundary corner.
            corner_sign: +1 for upper corner, -1 for lower corner.
        """
        self._is_pinned_corner = corner
        if corner:
            sgn = 1.0 if corner_sign >= 0 else 0.0
            self.params = SensorParams(
                range_noise_a=0.015 if sgn else 0.003,
                range_noise_b=0.010 if sgn else 0.000,
                beam_dropout_p=0.05 if sgn else 0.00,
                block_dropout_prob=0.30 if sgn else 0.00,
                num_blocks=2 if sgn else 0,
                block_length=10 if sgn else 0,
                sensor_delivery_latency_scans=1 if sgn else 0,
                mounting_yaw_offset=math.radians(3.0 * (1.0 if sgn else -1.0)),
                mounting_x_offset=0.02 * (1.0 if sgn else -1.0),
                mounting_y_offset=0.02 * (1.0 if sgn else -1.0),
                odometry_v_scale=1.05 if sgn else 0.95,
                odometry_yaw_rate_bias=0.005 * (1.0 if sgn else -1.0),
                vel_noise_sigma_v=0.005 if sgn else 0.000,
                vel_noise_sigma_omega=0.020 if sgn else 0.000,
            )
        else:
            self.params = SensorParams(
                range_noise_a=float(self.rng.uniform(0.003, 0.015)),
                range_noise_b=float(self.rng.uniform(0.000, 0.010)),
                beam_dropout_p=float(self.rng.uniform(0.00, 0.05)),
                block_dropout_prob=float(self.rng.uniform(0.00, 0.30)),
                num_blocks=int(self.rng.integers(1, 3)),
                block_length=int(self.rng.integers(3, 11)),
                sensor_delivery_latency_scans=int(self.rng.choice([0, 1])),
                mounting_yaw_offset=float(math.radians(self.rng.uniform(-3.0, 3.0))),
                mounting_x_offset=float(self.rng.uniform(-0.02, 0.02)),
                mounting_y_offset=float(self.rng.uniform(-0.02, 0.02)),
                odometry_v_scale=float(self.rng.uniform(0.95, 1.05)),
                odometry_yaw_rate_bias=float(self.rng.uniform(-0.005, 0.005)),
                vel_noise_sigma_v=float(self.rng.uniform(0.000, 0.005)),
                vel_noise_sigma_omega=float(self.rng.uniform(0.000, 0.020)),
            )

        self.scan_delay_buffer.clear()
        return self.params

    def perturb_scan(self, raw_ranges: np.ndarray) -> np.ndarray:
        """Apply noise, beam dropouts, and block dropouts to 360-beam LiDAR ranges."""
        perturbed = np.array(raw_ranges, dtype=np.float32, copy=True)
        valid = np.isfinite(perturbed) & (perturbed > 0.0)

        # 1. LiDAR Range noise: sigma = a + b * r
        if np.any(valid):
            r_valid = perturbed[valid]
            sigma = self.params.range_noise_a + self.params.range_noise_b * r_valid
            noise = self.rng.normal(0.0, sigma).astype(np.float32)
            perturbed[valid] = np.clip(r_valid + noise, RANGE_MIN, RANGE_MAX)

        # 2. Random individual beam dropout
        if self.params.beam_dropout_p > 0.0:
            dropout_mask = self.rng.uniform(0.0, 1.0, size=360) < self.params.beam_dropout_p
            perturbed[dropout_mask] = float('inf')

        # 3. Block dropout (invalid regions)
        if self.rng.uniform(0.0, 1.0) < self.params.block_dropout_prob:
            for _ in range(self.params.num_blocks):
                start_beam = int(self.rng.integers(0, 360))
                for k in range(self.params.block_length):
                    idx = (start_beam + k) % 360
                    perturbed[idx] = float('inf')

        # 4. Sensor delivery latency buffer (0 or 1 extra scan delay)
        if self.params.sensor_delivery_latency_scans > 0:
            self.scan_delay_buffer.append(perturbed)
            if len(self.scan_delay_buffer) > self.params.sensor_delivery_latency_scans:
                return self.scan_delay_buffer.pop(0)
            else:
                return self.scan_delay_buffer[0]

        return perturbed

    def perturb_odometry(self, v: float, omega: float) -> Tuple[float, float]:
        """Apply odometry scale error, yaw-rate bias, and measurement noise."""
        v_meas = v * self.params.odometry_v_scale
        if self.params.vel_noise_sigma_v > 0.0:
            v_meas += float(self.rng.normal(0.0, self.params.vel_noise_sigma_v))

        omega_meas = omega + self.params.odometry_yaw_rate_bias
        if self.params.vel_noise_sigma_omega > 0.0:
            omega_meas += float(self.rng.normal(0.0, self.params.vel_noise_sigma_omega))

        return float(v_meas), float(omega_meas)


class DynamicsRandomizer:
    """Applies actuator lag, control delay, command gains, and acceleration limits (P3/P4)."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.params = DynamicsParams()
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.actual_v = 0.0
        self.actual_omega = 0.0
        self._is_pinned_corner: bool = False

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, corner: bool = False, corner_sign: int = 1) -> DynamicsParams:
        """Sample per-episode dynamics parameters."""
        self._is_pinned_corner = corner
        if corner:
            sgn = 1.0 if corner_sign >= 0 else 0.0
            self.params = DynamicsParams(
                v_command_gain=1.15 if sgn else 0.85,
                omega_command_gain=1.15 if sgn else 0.85,
                actuator_lag_tau=0.30 if sgn else 0.00,
                control_delay_s=0.20 if sgn else 0.00,
                linear_acc_limit=0.20 if not sgn else 0.60,
                angular_acc_limit=1.00 if not sgn else 3.00,
                friction_multiplier=1.30 if sgn else 0.70,
                mass_multiplier=1.20 if sgn else 0.90,
            )
        else:
            self.params = DynamicsParams(
                v_command_gain=float(self.rng.uniform(0.85, 1.15)),
                omega_command_gain=float(self.rng.uniform(0.85, 1.15)),
                actuator_lag_tau=float(self.rng.uniform(0.00, 0.30)),
                control_delay_s=float(self.rng.uniform(0.00, 0.20)),
                linear_acc_limit=float(self.rng.uniform(0.20, 0.60)),
                angular_acc_limit=float(self.rng.uniform(1.00, 3.00)),
                friction_multiplier=float(self.rng.uniform(0.70, 1.30)),
                mass_multiplier=float(self.rng.uniform(0.90, 1.20)),
            )

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.actual_v = 0.0
        self.actual_omega = 0.0
        return self.params

    def perturb_action(
        self,
        cmd_v: float,
        cmd_omega: float,
        prev_cmd_v: float,
        prev_cmd_omega: float,
        dt: float = 0.20,
    ) -> Tuple[float, float]:
        """Apply control delay, command gains, actuator lag, and acceleration limits.

        Args:
            cmd_v: Target linear velocity command in [0, 0.15] m/s.
            cmd_omega: Target angular velocity command in [-0.6, 0.6] rad/s.
            prev_cmd_v: Previous linear velocity command.
            prev_cmd_omega: Previous angular velocity command.
            dt: Control period (0.20s).

        Returns:
            (v_effective, omega_effective) applied to physics.
        """
        # 1. Control delay d in [0, 0.2]s: previous action held for first d seconds of step
        d = min(self.params.control_delay_s, dt)
        if dt > 0.0:
            frac_prev = d / dt
            frac_curr = 1.0 - frac_prev
            target_v = frac_prev * prev_cmd_v + frac_curr * cmd_v
            target_omega = frac_prev * prev_cmd_omega + frac_curr * cmd_omega
        else:
            target_v = cmd_v
            target_omega = cmd_omega

        # 2. Command gain
        target_v *= self.params.v_command_gain
        target_omega *= self.params.omega_command_gain

        # 3. Acceleration limits
        max_delta_v = self.params.linear_acc_limit * dt
        max_delta_omega = self.params.angular_acc_limit * dt
        dv = np.clip(target_v - self.actual_v, -max_delta_v, max_delta_v)
        dw = np.clip(target_omega - self.actual_omega, -max_delta_omega, max_delta_omega)
        v_limited = self.actual_v + dv
        omega_limited = self.actual_omega + dw

        # 4. First-order actuator lag: dv/dt = (target - v) / tau
        tau = self.params.actuator_lag_tau
        if tau > 1e-4:
            alpha = dt / (tau + dt)
            self.actual_v = (1.0 - alpha) * self.actual_v + alpha * v_limited
            self.actual_omega = (1.0 - alpha) * self.actual_omega + alpha * omega_limited
        else:
            self.actual_v = v_limited
            self.actual_omega = omega_limited

        return float(self.actual_v), float(self.actual_omega)


class CombinedRandomizer:
    """Combined sensor and dynamics randomizer with P4 constraint enforcement."""

    def __init__(self, seed: Optional[int] = None):
        self.sensor_rand = SensorRandomizer(seed=seed)
        self.dynamics_rand = DynamicsRandomizer(seed=seed + 1 if seed is not None else None)
        self._is_pinned_corner: bool = False

    @property
    def sensor(self) -> SensorRandomizer:
        return self.sensor_rand

    @property
    def dynamics(self) -> DynamicsRandomizer:
        return self.dynamics_rand

    def seed(self, seed: int) -> None:
        self.sensor_rand.seed(seed)
        self.dynamics_rand.seed(seed + 1)

    def sample_episode(self, corner: bool = False, corner_sign: int = 1) -> Dict[str, Any]:
        """Sample joint parameters enforcing: sensor latency + control delay <= 0.40s."""
        self._is_pinned_corner = corner
        for _ in range(100):
            s_params = self.sensor_rand.sample_episode(corner=corner, corner_sign=corner_sign)
            d_params = self.dynamics_rand.sample_episode(corner=corner, corner_sign=corner_sign)

            sensor_latency = s_params.sensor_delivery_latency_scans * 0.20
            control_delay = d_params.control_delay_s

            if (sensor_latency + control_delay) <= 0.40:
                break
        else:
            # Fallback clamp control delay
            self.dynamics_rand.params.control_delay_s = min(
                self.dynamics_rand.params.control_delay_s,
                0.40 - sensor_latency
            )

        return {
            'sensor': asdict(self.sensor_rand.params),
            'dynamics': asdict(self.dynamics_rand.params),
        }
