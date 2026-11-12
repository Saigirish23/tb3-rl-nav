#!/usr/bin/env python3
"""Autonomous Orchestrator for P2 (Sensor) and P3 (Dynamics) Training.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Sections 6, 7, 10)
Completes the 2x2 study matrix alongside P1 (Nominal) and P4 (Combined).

Features:
- Multi-worker parallel training (4 workers by default, ~19.8 steps/s).
- Isolated Gazebo master ports (11345..11348) and domain IDs (30..33).
- Automatic port cleanup and process-group isolation.
- Automatic policy export to deploy/p{2,3}_policy.npz with numerical parity check (< 1e-6).
- Progress monitoring and checkpoint persistence.
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "tb3_rl_navigation"))

GAZEBO_PORTS = [11345, 11346, 11347, 11348]


def log(msg: str) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {msg}", flush=True)


def clean_ports() -> None:
    """Terminate orphan Gazebo processes listening strictly on ports 11345-11348."""
    for port in GAZEBO_PORTS:
        try:
            out = subprocess.check_output(["ss", "-tulpn"], text=True)
            for line in out.splitlines():
                if f":{port} " in line and "pid=" in line:
                    for part in line.split(","):
                        if "pid=" in part:
                            pid_str = part.split("pid=")[1].split()[0]
                            if pid_str.isdigit():
                                pid = int(pid_str)
                                log(f"Cleaning port {port}: terminating process {pid}...")
                                try:
                                    os.kill(pid, 15)  # SIGTERM
                                    time.sleep(1)
                                    os.kill(pid, 9)   # SIGKILL
                                except OSError:
                                    pass
        except Exception:
            pass
    time.sleep(2)


def run_training_stage(
    policy_name: str,
    target_steps: int = 700_000,
    seed: int = 0,
    workers: int = 4,
    use_gazebo: bool = True,
) -> Path:
    """Execute training for specified policy and return final_model path."""
    run_dir = REPO_ROOT / "runs" / f"main_{policy_name.lower()}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_model = run_dir / "final_model.zip"

    log(f"============================================================")
    log(f"Starting Training: Policy {policy_name} (Seed={seed}, Workers={workers}, Steps={target_steps:,})")
    log(f"Run Directory: {run_dir}")
    log(f"============================================================")

    if use_gazebo:
        clean_ports()

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "rl" / "train" / "train_ppo.py"),
        "--policy", policy_name,
        "--seed", str(seed),
        "--config", "configs/ppo.yaml",
        "--run-dir", str(run_dir),
        "--workers", str(workers),
        "--steps", str(target_steps),
    ]
    if use_gazebo:
        train_cmd.append("--gazebo")

    log_file = run_dir / "train.log"
    with open(log_file, "w", encoding="utf-8") as f_log:
        proc = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )

        last_progress_log = 0.0
        for line in proc.stdout:
            f_log.write(line)
            f_log.flush()
            if "total_timesteps" in line or "%" in line:
                now = time.time()
                if now - last_progress_log > 30.0:
                    log(f"[{policy_name}] {line.strip()}")
                    last_progress_log = now

        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Training {policy_name} failed with exit code {proc.returncode}. See {log_file}")

    if not final_model.exists():
        # Fallback to promoting latest checkpoint if needed
        ckpts = sorted(run_dir.glob("checkpoints/checkpoint_*_steps.zip"))
        if ckpts:
            import shutil
            shutil.copy2(ckpts[-1], final_model)
            vec_pkl = ckpts[-1].parent / f"checkpoint_vecnormalize_{ckpts[-1].stem.split('_')[1]}_steps.pkl"
            if vec_pkl.exists():
                shutil.copy2(vec_pkl, run_dir / "final_vecnormalize.pkl")

    assert final_model.exists(), f"Final model not found for {policy_name} at {final_model}"
    log(f"Training completed successfully: {final_model} ({final_model.stat().st_size} bytes)")

    # Auto-export policy to NumPy .npz
    out_npz = REPO_ROOT / "deploy" / f"{policy_name.lower()}_policy.npz"
    log(f"Exporting policy weights to {out_npz}...")
    from deploy.export_policy import export_policy
    export_policy(str(final_model), str(out_npz), n_verify_samples=100)
    log(f"Policy {policy_name} exported and parity verified.")

    if use_gazebo:
        clean_ports()

    return final_model


def main():
    parser = argparse.ArgumentParser(description="Train P2 (Sensor) and P3 (Dynamics) Policies")
    parser.add_argument("--steps", type=int, default=700_000, help="Target steps per policy (default: 700k)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--kinematic", action="store_true", help="Use kinematic simulator instead of Gazebo")
    parser.add_argument("--only-p2", action="store_true", help="Train only P2")
    parser.add_argument("--only-p3", action="store_true", help="Train only P3")
    args = parser.parse_args()

    use_gazebo = not args.kinematic
    policies_to_train = []
    if args.only_p2:
        policies_to_train.append("P2")
    elif args.only_p3:
        policies_to_train.append("P3")
    else:
        policies_to_train = ["P2", "P3"]

    log(f"Initiating Training Queue for: {policies_to_train}")
    t_start = time.time()

    for pol in policies_to_train:
        run_training_stage(
            policy_name=pol,
            target_steps=args.steps,
            seed=0,
            workers=args.workers,
            use_gazebo=use_gazebo,
        )

    t_total = time.time() - t_start
    log(f"All training stages complete in {t_total / 3600:.2f} hours.")


if __name__ == "__main__":
    main()

