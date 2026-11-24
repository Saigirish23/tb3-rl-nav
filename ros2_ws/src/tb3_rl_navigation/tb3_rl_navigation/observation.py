"""Observation Preprocessing and Estimated-Pose Module.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 4, 5)
Pure NumPy implementation with ZERO ROS dependencies.
Used identically in simulation training (Gymnasium) and robot deployment node.

Total Observation Dimension: 43
- 36 LiDAR bins: 10° min-pooled, bin 0 centered on forward, normalized [0, 1]
- 1 Goal distance: normalized by arena diagonal [0, 1]
- 2 Goal bearing: [sin(bearing), cos(bearing)]
- 1 Linear velocity: v / 0.15, clipped [-1, 1]
- 1 Angular velocity: omega / 0.60, clipped [-1, 1]
- 2 Previous action: last normalized action u in [-1, 1]^2 (zeros at reset)
"""

from typing import Tuple, Optional
import numpy as np


# Constants frozen in specification
NUM_LIDAR_BINS = 36
BIN_WIDTH_DEG = 10.0
RANGE_MAX = 3.50  # LDS-01 / simulated LiDAR maximum range in meters
ARENA_DIAGONAL = 5.656854249492381  # sqrt(4.0^2 + 4.0^2) for 4m x 4m arena
V_MAX = 0.15  # m/s
OMEGA_MAX = 0.60  # rad/s
OBS_DIM = 43


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def process_lidar_ranges(
    ranges: np.ndarray,
    angle_min: float = 0.0,
    angle_increment: float = 0.01749303564429283,  # ~1 degree
    range_max: float = RANGE_MAX,
    num_bins: int = NUM_LIDAR_BINS,
) -> np.ndarray:
    """Min-pool LiDAR ranges into 10° bins and normalize to [0, 1].

    - Bin 0 is centered on forward (0°), spanning [-5°, +5°).
    - Bin i covers [i*10° - 5°, i*10° + 5°).
    - Invalid readings (<= 0, inf, NaN) are mapped to range_max before min-pooling.
    - Each bin is normalized by range_max and clipped to [0, 1].

    Args:
        ranges: Array of raw LiDAR ranges (e.g. 360 beams).
        angle_min: Start angle of scan in radians (typically 0.0).
        angle_increment: Angular step per beam in radians (~0.01745).
        range_max: Maximum sensor range in meters (3.50 m).
        num_bins: Number of output bins (36).

    Returns:
        (num_bins,) np.ndarray of np.float32 in [0.0, 1.0].
    """
    ranges = np.asarray(ranges, dtype=np.float32).copy()
    n_beams = len(ranges)

    # 1. Map invalid readings (<= 0, NaN, inf) to range_max
    invalid_mask = np.isnan(ranges) | np.isinf(ranges) | (ranges <= 0.0) | (ranges > range_max)
    ranges[invalid_mask] = range_max

    # 2. Compute beam angles in degrees [0, 360)
    beam_indices = np.arange(n_beams, dtype=np.float32)
    angles_rad = angle_min + beam_indices * angle_increment
    angles_deg = np.rad2deg(angles_rad) % 360.0

    # 3. Bin assignment: Bin 0 is centered at 0° (span [-5°, +5°) == [355°, 360°) U [0°, 5°))
    # Shifting angles by +5° modulo 360 and floor dividing by 10 maps [-5, 5) -> 0, [5, 15) -> 1, etc.
    bin_indices = np.floor(((angles_deg + (BIN_WIDTH_DEG / 2.0)) % 360.0) / BIN_WIDTH_DEG).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # 4. Min-pooling per bin
    binned_ranges = np.full(num_bins, range_max, dtype=np.float32)
    for b in range(num_bins):
        mask = (bin_indices == b)
        if np.any(mask):
            binned_ranges[b] = np.min(ranges[mask])

    # 5. Normalization: range / range_max, clipped to [0, 1]
    binned_norm = np.clip(binned_ranges / range_max, 0.0, 1.0).astype(np.float32)
    return binned_norm


def apply_motion_distortion(
    ranges: np.ndarray,
    v: float,
    omega: float,
    scan_time: float = 0.2,
    angle_min: float = 0.0,
    angle_increment: float = 0.01749303564429283,
    range_max: float = RANGE_MAX,
) -> np.ndarray:
    """Apply nominal per-beam motion distortion caused by robot displacement during scan sweep.

    LDS-01 sweeps counterclockwise over scan_time (0.2s). Beam k is acquired at t_k = k * (scan_time / N).
    While moving with (v, omega), the sensor frame translates and rotates during the sweep.

    Args:
        ranges: Raw 1D array of ranges (e.g. 360 beams).
        v: Forward linear velocity (m/s).
        omega: Angular velocity (rad/s).
        scan_time: Total scan rotation duration (0.2s at 5 Hz).
        angle_min: Minimum scan angle (rad).
        angle_increment: Angle increment per beam (rad).
        range_max: Max sensor range (m).

    Returns:
        Motion-distorted ranges array.
    """
    ranges = np.asarray(ranges, dtype=np.float32).copy()
    if abs(v) < 1e-5 and abs(omega) < 1e-5:
        return ranges

    n_beams = len(ranges)
    beam_idx = np.arange(n_beams, dtype=np.float32)
    t_k = beam_idx * (scan_time / max(1, n_beams))

    # Robot displacement from t=0 to t_k
    delta_theta_k = omega * t_k
    theta_mid = 0.5 * delta_theta_k
    dx_k = v * t_k * np.cos(theta_mid)
    dy_k = v * t_k * np.sin(theta_mid)

    # Valid range mask
    valid_mask = ~np.isnan(ranges) & ~np.isinf(ranges) & (ranges > 0.0) & (ranges <= range_max)
    output = ranges.copy()
    if not np.any(valid_mask):
        return output

    # Sensor frame angle at t=0
    theta_sensor_0 = angle_min + beam_idx[valid_mask] * angle_increment
    valid_ranges = ranges[valid_mask]
    px = valid_ranges * np.cos(theta_sensor_0)
    py = valid_ranges * np.sin(theta_sensor_0)

    # Distorted relative position from moving sensor to detected obstacle point
    # Moving sensor link is displaced by (dx_k, dy_k)
    rx = px - dx_k[valid_mask]
    ry = py - dy_k[valid_mask]
    distorted_r = np.sqrt(rx * rx + ry * ry)

    output[valid_mask] = np.clip(distorted_r, 0.0, range_max)
    return output.astype(np.float32)


class EstimatedPose:
    """Estimated-Pose dead-reckoning model integrated from reset odometry frame.

    Per Section 4 of specification:
    - Integrates measured v and omega from the reset pose.
    - Goal lives in the reset odometry frame.
    - Goal distance and bearing are computed relative to this estimated pose.
    - Eliminates per-step independent noise; preserves physically accumulating drift.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x: float = float(x)
        self.y: float = float(y)
        self.theta: float = float(theta)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        """Reset estimated pose to given initial coordinate."""
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_to_pi(float(theta))

    def update(self, v_meas: float, omega_meas: float, dt: float = 0.20) -> Tuple[float, float, float]:
        """Integrate dead-reckoning motion over dt using midpoint circular arc.

        Args:
            v_meas: Measured linear velocity (m/s).
            omega_meas: Measured angular velocity (rad/s).
            dt: Time step duration (seconds, default 0.2s for 5 Hz).

        Returns:
            Updated (x, y, theta).
        """
        delta_theta = float(omega_meas) * float(dt)
        theta_mid = self.theta + 0.5 * delta_theta

        self.x += float(v_meas) * float(dt) * np.cos(theta_mid)
        self.y += float(v_meas) * float(dt) * np.sin(theta_mid)
        self.theta = wrap_to_pi(self.theta + delta_theta)

        return self.x, self.y, self.theta

    def compute_goal_features(
        self,
        goal_x: float,
        goal_y: float,
        arena_diagonal: float = ARENA_DIAGONAL,
    ) -> Tuple[float, float, float, float]:
        """Compute goal distance (normalized) and bearing sin/cos from estimated pose.

        Args:
            goal_x: Goal x coordinate in the reset odometry frame.
            goal_y: Goal y coordinate in the reset odometry frame.
            arena_diagonal: Normalization constant for distance (5.657 m).

        Returns:
            Tuple of (goal_dist_norm, sin_bearing, cos_bearing, raw_goal_dist).
        """
        dx = float(goal_x) - self.x
        dy = float(goal_y) - self.y
        raw_dist = float(np.hypot(dx, dy))

        # Normalized goal distance in [0, 1]
        dist_norm = float(np.clip(raw_dist / arena_diagonal, 0.0, 1.0))

        # Goal bearing in robot frame
        global_angle = np.arctan2(dy, dx)
        bearing = wrap_to_pi(global_angle - self.theta)

        sin_bearing = float(np.sin(bearing))
        cos_bearing = float(np.cos(bearing))

        return dist_norm, sin_bearing, cos_bearing, raw_dist


class ObservationPreprocessor:
    """Observation Preprocessor constructing the standard 43-dim state vector.

    Compatible with both simulation and physical robot deployment.
    """

    def __init__(
        self,
        range_max: float = RANGE_MAX,
        arena_diagonal: float = ARENA_DIAGONAL,
        v_max: float = V_MAX,
        omega_max: float = OMEGA_MAX,
        num_bins: int = NUM_LIDAR_BINS,
    ):
        self.range_max = range_max
        self.arena_diagonal = arena_diagonal
        self.v_max = v_max
        self.omega_max = omega_max
        self.num_bins = num_bins

        self.estimated_pose = EstimatedPose()
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_theta: float = 0.0,
    ) -> None:
        """Reset internal estimated pose and action buffer."""
        self.estimated_pose.reset(init_x, init_y, init_theta)
        self.prev_action = np.zeros(2, dtype=np.float32)

    def step(
        self,
        ranges: np.ndarray,
        v_meas: float,
        omega_meas: float,
        goal_x: float,
        goal_y: float,
        last_action: Optional[np.ndarray] = None,
        dt: float = 0.20,
        angle_min: float = 0.0,
        angle_increment: float = 0.01749303564429283,
    ) -> np.ndarray:
        """Process sensor readings and return standard 43-dim observation vector.

        Args:
            ranges: Raw LiDAR ranges.
            v_meas: Measured forward velocity.
            omega_meas: Measured angular velocity.
            goal_x: Goal x coordinate in reset frame.
            goal_y: Goal y coordinate in reset frame.
            last_action: Normalized previous action u in [-1, 1]^2.
            dt: Time since last scan (0.2s).
            angle_min: Start angle of LiDAR.
            angle_increment: Angular increment of LiDAR.

        Returns:
            (43,) np.ndarray of np.float32 within expected bounds.
        """
        # Update estimated pose
        self.estimated_pose.update(v_meas, omega_meas, dt=dt)

        # Update previous action buffer
        if last_action is not None:
            self.prev_action = np.clip(np.asarray(last_action, dtype=np.float32), -1.0, 1.0)

        # 1. LiDAR Bins (36)
        binned_scan = process_lidar_ranges(
            ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_max=self.range_max,
            num_bins=self.num_bins,
        )

        # 2 & 3. Goal Distance & Bearing (1 + 2 = 3)
        goal_dist_norm, sin_b, cos_b, _ = self.estimated_pose.compute_goal_features(
            goal_x, goal_y, arena_diagonal=self.arena_diagonal
        )

        # 4 & 5. Measured Velocities (1 + 1 = 2)
        v_norm = float(np.clip(v_meas / self.v_max, -1.0, 1.0))
        omega_norm = float(np.clip(omega_meas / self.omega_max, -1.0, 1.0))

        # 6. Previous Action (2)
        prev_act = self.prev_action.copy()

        # Construct full 43-dim vector
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:36] = binned_scan
        obs[36] = goal_dist_norm
        obs[37] = sin_b
        obs[38] = cos_b
        obs[39] = v_norm
        obs[40] = omega_norm
        obs[41:43] = prev_act

        return obs
