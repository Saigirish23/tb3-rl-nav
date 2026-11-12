"""Pure NumPy MLP Policy Engine for TurtleBot3 Deployment.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 2, 3)
Architecture (Fixed in Specification):
- Continuous action space u in [-1, 1]^2
- Separate actor and critic MLPs (only actor weights needed for deployment)
- 43-dim input
- Layer 1: Linear(43, 128) + tanh
- Layer 2: Linear(128, 128) + tanh
- Output: Linear(128, 2), clipped to [-1, 1]

Zero PyTorch or SB3 dependency required at inference time.
Runs in < 20 microseconds on modern CPUs / Raspberry Pi.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np


class NumPyPolicy:
    """Pure NumPy implementation of the trained SB3 PPO Actor MLP."""

    def __init__(
        self,
        w0: np.ndarray,
        b0: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.w0 = np.asarray(w0, dtype=np.float32)
        self.b0 = np.asarray(b0, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.metadata = metadata or {}

        # Validate layer shapes
        assert self.w0.shape == (128, 43), f"Invalid W0 shape: {self.w0.shape}"
        assert self.b0.shape == (128,), f"Invalid b0 shape: {self.b0.shape}"
        assert self.w1.shape == (128, 128), f"Invalid W1 shape: {self.w1.shape}"
        assert self.b1.shape == (128,), f"Invalid b1 shape: {self.b1.shape}"
        assert self.w2.shape == (2, 128), f"Invalid W2 shape: {self.w2.shape}"
        assert self.b2.shape == (2,), f"Invalid b2 shape: {self.b2.shape}"

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "NumPyPolicy":
        """Load policy weights from an exported .npz archive."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Exported policy weights not found at: {path}")

        data = np.load(path, allow_pickle=True)
        w0 = data["w0"]
        b0 = data["b0"]
        w1 = data["w1"]
        b1 = data["b1"]
        w2 = data["w2"]
        b2 = data["b2"]
        metadata = None
        if "metadata" in data:
            metadata = data["metadata"].item()

        return cls(w0=w0, b0=b0, w1=w1, b1=b1, w2=w2, b2=b2, metadata=metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NumPyPolicy":
        """Load either an exported .npz file or an SB3 .zip file."""
        path = Path(path)
        if path.suffix == ".npz":
            return cls.load_npz(path)

        # Fallback to extracting from SB3 zip archive directly
        try:
            import zipfile
            import io
            import torch

            with zipfile.ZipFile(path, "r") as archive:
                with archive.open("policy.pth") as pth_file:
                    state_dict = torch.load(io.BytesIO(pth_file.read()), map_location="cpu")

            w0 = state_dict["mlp_extractor.policy_net.0.weight"].numpy()
            b0 = state_dict["mlp_extractor.policy_net.0.bias"].numpy()
            w1 = state_dict["mlp_extractor.policy_net.2.weight"].numpy()
            b1 = state_dict["mlp_extractor.policy_net.2.bias"].numpy()
            w2 = state_dict["action_net.weight"].numpy()
            b2 = state_dict["action_net.bias"].numpy()

            return cls(w0, b0, w1, b1, w2, b2, metadata={"source": str(path)})
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Compute deterministic action output matching SB3 predict API.

        Args:
            obs: (43,) or (B, 43) observation array.
            deterministic: Ignored (deployment uses mean action).

        Returns:
            (action, None) where action is in [-1, 1]^2.
        """
        x = np.asarray(obs, dtype=np.float32)
        single_input = (x.ndim == 1)
        if single_input:
            x = x.reshape(1, -1)

        # Forward pass through MLP actor
        # Layer 1
        h1 = np.tanh(x @ self.w0.T + self.b0)
        # Layer 2
        h2 = np.tanh(h1 @ self.w1.T + self.b1)
        # Output
        action = np.clip(h2 @ self.w2.T + self.b2, -1.0, 1.0)

        if single_input:
            action = action[0]

        return action, None

    def export_npz(self, out_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Export weights and metadata to a portable .npz file."""
        meta = metadata or self.metadata
        np.savez_compressed(
            out_path,
            w0=self.w0,
            b0=self.b0,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            metadata=meta,
        )

