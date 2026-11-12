#!/usr/bin/env python3
"""Export Stable-Baselines3 PPO Actor MLP to Portable Pure NumPy Archive.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 10, 11)
Exports actor weights to .npz for lightweight embedded inference on Raspberry Pi.
Validates bit-for-bit numerical parity (< 1e-6) against SB3 model on random inputs.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "tb3_rl_navigation"))

from stable_baselines3 import PPO
from tb3_rl_navigation.policy_model import NumPyPolicy


def export_policy(
    model_path: str,
    output_npz_path: str,
    n_verify_samples: int = 100,
    tolerance: float = 1e-6,
) -> bool:
    """Export SB3 model to .npz and verify numerical parity."""
    model_file = Path(model_path)
    out_file = Path(output_npz_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading SB3 model from: {model_file}")
    model = PPO.load(str(model_file), device="cpu")

    # Extract weights from PyTorch state dict
    sd = model.policy.state_dict()
    w0 = sd["mlp_extractor.policy_net.0.weight"].numpy().astype(np.float32)
    b0 = sd["mlp_extractor.policy_net.0.bias"].numpy().astype(np.float32)
    w1 = sd["mlp_extractor.policy_net.2.weight"].numpy().astype(np.float32)
    b1 = sd["mlp_extractor.policy_net.2.bias"].numpy().astype(np.float32)
    w2 = sd["action_net.weight"].numpy().astype(np.float32)
    b2 = sd["action_net.bias"].numpy().astype(np.float32)

    meta = {
        "source_model": str(model_file.resolve()),
        "export_time": datetime.utcnow().isoformat(),
        "input_dim": 43,
        "hidden_dims": [128, 128],
        "output_dim": 2,
        "activation": "tanh",
        "action_range": [-1.0, 1.0],
    }

    # Create NumPy policy instance and export
    np_policy = NumPyPolicy(w0, b0, w1, b1, w2, b2, metadata=meta)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np_policy.export_npz(out_file)
    print(f"Exported .npz archive saved to: {out_file} ({out_file.stat().st_size} bytes)")

    # Parity verification against SB3
    print(f"Running numerical parity verification on {n_verify_samples} random observations...")
    rng = np.random.default_rng(2026)
    test_obs = rng.uniform(-1.0, 1.0, size=(n_verify_samples, 43)).astype(np.float32)

    max_err = 0.0
    for i in range(n_verify_samples):
        obs = test_obs[i]
        sb3_act, _ = model.predict(obs, deterministic=True)
        np_act, _ = np_policy.predict(obs, deterministic=True)
        err = float(np.max(np.abs(sb3_act - np_act)))
        if err > max_err:
            max_err = err
        assert err < tolerance, f"Parity mismatch at sample {i}: max error {err:.2e} >= {tolerance:.2e}"

    print(f"PARITY VERIFIED: Max absolute error across {n_verify_samples} samples: {max_err:.2e} (< {tolerance:.2e})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to NumPy .npz archive")
    parser.add_argument("--model", type=str, required=True, help="Path to final_model.zip")
    parser.add_argument("--output", type=str, required=True, help="Output path for .npz file")
    parser.add_argument("--samples", type=int, default=100, help="Number of parity test samples")
    args = parser.parse_args()

    export_policy(args.model, args.output, n_verify_samples=args.samples)


if __name__ == "__main__":
    main()

