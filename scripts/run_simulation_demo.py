#!/usr/bin/env python3
"""Polished & Reproducible Gazebo Classic PPO Simulation Demo Runner.

Features:
- Seamless execution of multi-goal waypoint navigation tours in Gazebo Classic.
- Supports both P4 (Combined domain randomization) and P1 (Nominal) policies.
- Automatically handles simulator lifecycle via GazeboWatchdog or connects to existing.
- Real-time formatted console output with progress indicators.
- Saves machine-readable JSON metrics under runs/demo_run_<timestamp>/ without overwriting.
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.train.watchdog import GazeboWatchdog

DEFAULT_PORT = 11345
DEFAULT_DOMAIN_ID = 30
GOAL_TOLERANCE = 0.20

STANDARD_SCENARIOS = [
    {"goal_x": 1.2, "goal_y": 0.0, "label": "Straight Forward (East)"},
    {"goal_x": 1.0, "goal_y": 1.0, "label": "Diagonal Positive (North-East)"},
    {"goal_x": -1.0, "goal_y": 1.0, "label": "Left Quadrant (North-West)"},
]


def print_banner(policy_name: str, model_path: str, port: int, domain_id: int):
    print("=" * 78)
    print("  TURTLEBOT3 PPO MAPLESS LIDAR NAVIGATION - GAZEBO SIMULATION DEMO")
    print("=" * 78)
    print(f"  Selected Policy:   {policy_name}")
    print(f"  Model Checkpoint:  {model_path}")
    print(f"  Simulator Port:    {port}")
    print(f"  ROS 2 Domain ID:   {domain_id}")
    print(f"  Goal Tolerance:    {GOAL_TOLERANCE:.2f} m")
    print(f"  Scenarios:         3 multi-waypoint challenges in 4m x 4m arena")
    print("=" * 78)
    print("")


def run_demo(
    policy: str = "P4",
    port: int = DEFAULT_PORT,
    domain_id: int = DEFAULT_DOMAIN_ID,
    connect_existing: bool = False,
    output_dir: str = "",
) -> Dict[str, Any]:
    # Determine model and vecnorm paths
    if policy.upper() == "P4":
        model_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p4_s0" / "final_vecnormalize.pkl"
    elif policy.upper() == "P1":
        model_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_model.zip"
        vecnorm_path = REPO_ROOT / "runs" / "main_p1_s0" / "final_vecnormalize.pkl"
    else:
        raise ValueError(f"Unknown policy: {policy}. Must be P1 or P4.")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Set up timestamped output directory
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    if not output_dir:
        run_output_dir = REPO_ROOT / "runs" / f"demo_run_{timestamp_str}"
    else:
        run_output_dir = Path(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print_banner(policy, str(model_path.relative_to(REPO_ROOT)), port, domain_id)

    watchdog = None
    if not connect_existing:
        print(f"[DemoRunner] Spawning headless Gazebo Classic on port {port} (Domain {domain_id})...")
        watchdog = GazeboWatchdog(port=port, domain_id=domain_id)
        watchdog.start_simulator()
        print("[DemoRunner] Waiting 10s for Gazebo simulator and robot spawner to initialize...")
        time.sleep(10)
    else:
        print(f"[DemoRunner] Connecting to existing Gazebo instance on port {port} (Domain {domain_id})...")

    demo_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": policy,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path.exists() else None,
        "port": port,
        "ros_domain_id": domain_id,
        "goal_tolerance": GOAL_TOLERANCE,
        "trials": [],
    }

    env = os.environ.copy()
    env["GAZEBO_MASTER_URI"] = f"http://localhost:{port}"
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["TURTLEBOT3_MODEL"] = "burger"

    all_passed = True
    try:
        for idx, sc in enumerate(STANDARD_SCENARIOS, 1):
            gx, gy, label = sc["goal_x"], sc["goal_y"], sc["label"]
            print("-" * 78)
            print(f">>> TRIAL {idx}/3: Target Goal=({gx:.2f}, {gy:.2f}) [{label}]")
            print("-" * 78)

            cmd = [
                sys.executable,
                str(REPO_ROOT / "deploy" / "ppo_inference_node.py"),
                "--model-path", str(model_path),
                "--goal-x", str(gx),
                "--goal-y", str(gy),
                "--goal-tolerance", str(GOAL_TOLERANCE),
                "--output-dir", str(run_output_dir),
            ]
            if vecnorm_path.exists():
                cmd.extend(["--vecnorm-path", str(vecnorm_path)])

            t0 = time.monotonic()
            proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), text=True)
            wall_time = round(time.monotonic() - t0, 2)

            # Find latest written trial log
            trial_logs = sorted(run_output_dir.glob("inference_trial_*.json"))
            trial_data = {}
            if trial_logs:
                latest_log = trial_logs[-1]
                with open(latest_log, "r", encoding="utf-8") as f:
                    trial_data = json.load(f)

            is_success = trial_data.get("success", False)
            outcome = trial_data.get("outcome", "FAILED")
            steps = trial_data.get("steps", 0)
            final_dist = trial_data.get("final_goal_distance_meters", float("nan"))
            disp = trial_data.get("displacement_meters", float("nan"))
            min_clear = trial_data.get("min_obstacle_clearance_meters", float("nan"))

            if not is_success:
                all_passed = False

            demo_summary["trials"].append({
                "trial": idx,
                "label": label,
                "goal": [gx, gy],
                "outcome": outcome,
                "success": is_success,
                "steps": steps,
                "duration_sec": wall_time,
                "final_distance_m": final_dist,
                "displacement_m": disp,
                "min_clearance_m": min_clear,
                "trial_log_file": str(latest_log.relative_to(REPO_ROOT)) if trial_logs else None,
            })
            print("")

    finally:
        if watchdog is not None:
            print("[DemoRunner] Shutting down Gazebo simulator cleanly...")
            watchdog.terminate_simulator()

    # Save summary JSON
    summary_path = run_output_dir / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(demo_summary, f, indent=2)

    # Print Formatted Results Table
    print("=" * 78)
    print("  SIMULATION DEMO ACCEPTANCE TEST RESULTS")
    print("=" * 78)
    header = f"{'Trial':<6} | {'Target Goal':<14} | {'Outcome':<10} | {'Steps':<6} | {'Time':<6} | {'Final Dist':<10} | {'Min Clear':<10}"
    print(header)
    print("-" * 78)
    for t in demo_summary["trials"]:
        g_str = f"({t['goal'][0]:.2f}, {t['goal'][1]:.2f})"
        row = f"{t['trial']:<6} | {g_str:<14} | {t['outcome']:<10} | {t['steps']:<6} | {t['duration_sec']:<5.1f}s | {t['final_distance_m']:<9.3f}m | {t['min_clearance_m']:<9.3f}m"
        print(row)
    print("=" * 78)
    print(f"Summary JSON saved: {summary_path.relative_to(REPO_ROOT)}")
    print(f"Overall Status:     {'ALL TRIALS PASSED (100% SUCCESS)' if all_passed else 'SOME TRIALS FAILED'}")
    print("=" * 78)

    return demo_summary


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 PPO Simulation Demo Runner")
    parser.add_argument("--policy", type=str, default="P4", choices=["P4", "P1"], help="Policy to run (P4 or P1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Gazebo port (default 11345)")
    parser.add_argument("--domain-id", type=int, default=DEFAULT_DOMAIN_ID, help="ROS Domain ID (default 30)")
    parser.add_argument("--connect-existing", action="store_true", help="Connect to existing Gazebo instance instead of launching new one")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory")
    args = parser.parse_args()

    res = run_demo(
        policy=args.policy,
        port=args.port,
        domain_id=args.domain_id,
        connect_existing=args.connect_existing,
        output_dir=args.output_dir,
    )
    all_success = all(t.get("success", False) for t in res.get("trials", []))
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
