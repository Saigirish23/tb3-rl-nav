"""Unit Tests for Policy Export, Pure NumPy Inference, and Observation Parity.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 4, 10, 11)
Verifies:
1. Numerical parity (< 1e-6) between SB3 .predict() and NumPyPolicy across random 43-D observations.
2. Parity between deployed .npz archive and live SB3 model.
3. Batch inference output parity.
4. Single-step inference latency (< 100 microseconds).
5. Parity between tb3_rl_navigation.observation and rl.env.observation.
"""

import time
import pytest
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from tb3_rl_navigation.policy_model import NumPyPolicy
import tb3_rl_navigation.observation as deploy_obs
import rl.env.observation as env_obs


@pytest.mark.parametrize("policy_tag, model_path, npz_path", [
    ("P1", "runs/main_p1_s0/final_model.zip", "deploy/p1_policy.npz"),
    ("P4", "runs/main_p4_s0/final_model.zip", "deploy/p4_policy.npz"),
])
def test_sb3_vs_numpy_policy_parity(policy_tag, model_path, npz_path):
    """Verify exported NumPy policy achieves exact numerical parity with SB3 model."""
    m_path = Path(model_path)
    z_path = Path(npz_path)

    if not m_path.exists() or not z_path.exists():
        pytest.skip(f"Artifact {model_path} or {npz_path} not found")

    sb3_model = PPO.load(str(m_path), device="cpu")
    numpy_policy = NumPyPolicy.load_npz(z_path)

    rng = np.random.default_rng(12345)
    test_inputs = rng.uniform(-1.0, 1.0, size=(100, 43)).astype(np.float32)

    # 1. Single sample prediction loop
    max_error = 0.0
    for i in range(100):
        obs = test_inputs[i]
        sb3_act, _ = sb3_model.predict(obs, deterministic=True)
        np_act, _ = numpy_policy.predict(obs, deterministic=True)

        err = float(np.max(np.abs(sb3_act - np_act)))
        if err > max_error:
            max_error = err
        assert err < 1e-6, f"Mismatch on {policy_tag} at sample {i}: error {err} >= 1e-6"

    assert max_error < 1e-6

    # 2. Batch prediction parity
    batch_np_act, _ = numpy_policy.predict(test_inputs, deterministic=True)
    assert batch_np_act.shape == (100, 2)
    for i in range(100):
        single_act, _ = numpy_policy.predict(test_inputs[i])
        assert np.allclose(batch_np_act[i], single_act, atol=1e-7)


def test_numpy_policy_inference_latency():
    """Verify pure NumPy forward pass latency is < 100 microseconds per step."""
    z_path = Path("deploy/p4_policy.npz")
    if not z_path.exists():
        pytest.skip("deploy/p4_policy.npz not found")

    policy = NumPyPolicy.load_npz(z_path)
    sample_obs = np.zeros(43, dtype=np.float32)

    # Warmup
    for _ in range(50):
        policy.predict(sample_obs)

    # Benchmark 2,000 steps
    n_iters = 2000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        policy.predict(sample_obs)
    t1 = time.perf_counter()

    avg_latency_us = ((t1 - t0) / n_iters) * 1e6
    assert avg_latency_us < 100.0, f"Average latency too high: {avg_latency_us:.1f} us (target < 100 us)"


def test_observation_module_parity():
    """Verify observation module in tb3_rl_navigation matches rl.env.observation identically."""
    # Constants
    assert deploy_obs.OBS_DIM == env_obs.OBS_DIM == 43
    assert deploy_obs.NUM_LIDAR_BINS == env_obs.NUM_LIDAR_BINS == 36
    assert deploy_obs.RANGE_MAX == env_obs.RANGE_MAX == 3.50
    assert deploy_obs.V_MAX == env_obs.V_MAX == 0.15
    assert deploy_obs.OMEGA_MAX == env_obs.OMEGA_MAX == 0.60

    # Functional parity on random scans
    rng = np.random.default_rng(999)
    raw_scan = rng.uniform(0.1, 4.0, size=360).astype(np.float32)

    binned_deploy = deploy_obs.process_lidar_ranges(raw_scan)
    binned_env = env_obs.process_lidar_ranges(raw_scan)
    assert np.allclose(binned_deploy, binned_env, atol=1e-7)

    # Preprocessor parity
    prep_deploy = deploy_obs.ObservationPreprocessor()
    prep_env = env_obs.ObservationPreprocessor()

    obs_deploy = prep_deploy.step(
        ranges=raw_scan,
        v_meas=0.10,
        omega_meas=-0.25,
        goal_x=1.5,
        goal_y=0.5,
        last_action=np.array([0.5, -0.2], dtype=np.float32),
        dt=0.20,
    )
    obs_env = prep_env.step(
        ranges=raw_scan,
        v_meas=0.10,
        omega_meas=-0.25,
        goal_x=1.5,
        goal_y=0.5,
        last_action=np.array([0.5, -0.2], dtype=np.float32),
        dt=0.20,
    )
    assert np.allclose(obs_deploy, obs_env, atol=1e-7)

