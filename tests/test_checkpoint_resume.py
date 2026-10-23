"""Unit test for PPO checkpoint save/reload and state restoration parity (non-training).

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 9 Check 9 / Check 10)
Validates:
1. Model policy weights and biases are saved and restored identically.
2. VecNormalize running reward statistics (ret_rms.mean, ret_rms.var, ret_rms.count) are restored identically.
3. Action predictions for identical observation batches match with atol=1e-6.
4. Clean environment and resource teardown.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.env.observation import OBS_DIM


def test_checkpoint_save_reload_parity():
    tmp_dir = tempfile.mkdtemp(prefix="tb3_ckpt_test_")
    try:
        def make_env():
            return TurtleBot3NavigationEnv(use_gazebo=False, seed=42)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

        # Set specific running return normalization statistics
        vec_env.ret_rms.mean = np.array([2.345], dtype=np.float64)
        vec_env.ret_rms.var = np.array([5.678], dtype=np.float64)
        vec_env.ret_rms.count = 250.0

        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=64,
            batch_size=32,
            gamma=0.99,
            seed=42,
            device="cpu",
        )

        ckpt_path = os.path.join(tmp_dir, "checkpoint_test.zip")
        vecnorm_path = os.path.join(tmp_dir, "checkpoint_test_vecnormalize.pkl")

        # 1. Save
        model.save(ckpt_path)
        vec_env.save(vecnorm_path)
        assert os.path.exists(ckpt_path), "Model checkpoint file was not saved"
        assert os.path.exists(vecnorm_path), "VecNormalize file was not saved"

        # 2. Reload into separate environment
        reloaded_base_env = DummyVecEnv([make_env])
        reloaded_vec_env = VecNormalize.load(vecnorm_path, reloaded_base_env)
        reloaded_vec_env.training = False

        reloaded_model = PPO.load(ckpt_path, env=reloaded_vec_env, device="cpu")

        # 3. Assert VecNormalize statistics restoration parity
        np.testing.assert_allclose(
            reloaded_vec_env.ret_rms.mean,
            vec_env.ret_rms.mean,
            atol=1e-7,
            err_msg="VecNormalize mean was not restored identically",
        )
        np.testing.assert_allclose(
            reloaded_vec_env.ret_rms.var,
            vec_env.ret_rms.var,
            atol=1e-7,
            err_msg="VecNormalize variance was not restored identically",
        )
        assert reloaded_vec_env.ret_rms.count == vec_env.ret_rms.count

        # 4. Assert policy parameters bitwise parity
        orig_sd = model.policy.state_dict()
        reloaded_sd = reloaded_model.policy.state_dict()
        for k in orig_sd.keys():
            assert torch.equal(orig_sd[k], reloaded_sd[k]), f"Mismatch in parameter {k}"

        # 5. Assert deterministic action prediction parity across batch
        rng = np.random.default_rng(123)
        obs_batch = rng.uniform(-1.0, 1.0, size=(10, OBS_DIM)).astype(np.float32)

        orig_actions, _ = model.predict(obs_batch, deterministic=True)
        reloaded_actions, _ = reloaded_model.predict(obs_batch, deterministic=True)

        np.testing.assert_allclose(
            orig_actions,
            reloaded_actions,
            atol=1e-6,
            err_msg="Reloaded policy predictions differ from original model",
        )

        # 6. Clean teardown
        vec_env.close()
        reloaded_vec_env.close()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_checkpoint_save_reload_parity()
    print("Checkpoint save/reload parity successfully verified!")
