"""TurtleBot3 RL Navigation Package (Deploy-Side).

Contains:
- observation: pure NumPy 43-dim state preprocessor and estimated pose tracker.
- safety: independent safety supervisor (clamps, stale-scan watchdog, collision stop).
- policy_model: lightweight pure NumPy MLP policy inference engine.
- deploy_node: production ROS 2 deployment node.
"""

from tb3_rl_navigation.observation import (
    ObservationPreprocessor,
    EstimatedPose,
    process_lidar_ranges,
    wrap_to_pi,
    apply_motion_distortion,
    OBS_DIM,
    RANGE_MAX,
    ARENA_DIAGONAL,
    NUM_LIDAR_BINS,
    V_MAX,
    OMEGA_MAX,
)
from tb3_rl_navigation.safety import SafetyFilter, SafetyIntervention
from tb3_rl_navigation.policy_model import NumPyPolicy

__all__ = [
    "ObservationPreprocessor",
    "EstimatedPose",
    "process_lidar_ranges",
    "wrap_to_pi",
    "apply_motion_distortion",
    "OBS_DIM",
    "RANGE_MAX",
    "ARENA_DIAGONAL",
    "NUM_LIDAR_BINS",
    "V_MAX",
    "OMEGA_MAX",
    "SafetyFilter",
    "SafetyIntervention",
    "NumPyPolicy",
]

