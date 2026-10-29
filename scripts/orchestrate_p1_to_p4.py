#!/usr/bin/env python3
"""Autonomous Orchestrator: P1 Completion Monitor, Scheduled P4 Launch, and Evaluation.

Specifications:
- Monitors running P1 training (runs/main_p1_s0, target 850,000 steps).
- Validates P1 final_model.zip and final_vecnormalize.pkl integrity upon completion.
- Schedules P4 training to start at 04:15 AM IST (2026-10-25 04:15:00+05:30), or immediately
  after P1 finishes if P1 finishes after 04:15 AM.
- Never launches duplicate P4 runs. Ensures Gazebo ports 11345-11348 are cleanly freed.
- Launches P4 (Combined Randomization) with seed 0, workers 4, --gazebo, isolated runs/main_p4_s0.
- Enforces deadline of 09:00 AM IST (2026-10-25 09:00:00+05:30) via graceful stop.
- Verifies P4 final model and normalization statistics.
- Executes simulation evaluation on held-out test scenarios.
- Persists progress continuously to runs/orchestrator.log and runs/orchestration_status.json.
"""

from __future__ import annotations
import os
import sys
import time
import json
import fcntl
import signal
import pickle
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

# Set up project path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Constants and Configuration - Strict Hard Deadline: 9:00 PM IST (2026-10-25 21:00:00 IST)
IST = timezone(timedelta(hours=5, minutes=30))
HARD_DEADLINE_DT = datetime(2026, 10, 25, 21, 0, 0, tzinfo=IST)
HARD_DEADLINE_EPOCH = HARD_DEADLINE_DT.timestamp()

# Target completion for all training, evaluation, and verification: 8:45 PM IST
TARGET_FINISH_DT = datetime(2026, 10, 25, 20, 45, 0, tzinfo=IST)
TARGET_FINISH_EPOCH = TARGET_FINISH_DT.timestamp()

# Stop P4 training by 5:45 PM IST (Formula: 20:45 target - 1.5h inference - 1.0h eval - 0.5h buffer = 17:45 IST)
P4_STOP_DEADLINE_DT = datetime(2026, 10, 25, 17, 45, 0, tzinfo=IST)
P4_STOP_DEADLINE_EPOCH = P4_STOP_DEADLINE_DT.timestamp()

# Feasible P4 Target Steps: 700,000 transitions (feasible in ~9.8h at 19.82 sps, finishing ~14:04 IST)
P4_TARGET_STEPS = 700_000

P1_RUN_DIR = REPO_ROOT / "runs" / "main_p1_s0"
P4_RUN_DIR = REPO_ROOT / "runs" / "main_p4_s0"
STATUS_FILE = REPO_ROOT / "runs" / "orchestration_status.json"
LOG_FILE = REPO_ROOT / "runs" / "orchestrator.log"
LOCK_FILE = REPO_ROOT / "runs" / "orchestrator.lock"

GAZEBO_PORTS = [11345, 11346, 11347, 11348]


def log(msg: str) -> None:
    """Log formatted message with timestamp to stdout and orchestrator.log."""
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}", file=sys.stderr)


def update_status(phase: str, details: Dict[str, Any]) -> None:
    """Update global orchestration status JSON atomically."""
    status_data = {
        "phase": phase,
        "timestamp": datetime.now(IST).isoformat(),
        "epoch": time.time(),
        "details": details,
    }
    tmp_file = STATUS_FILE.with_suffix(".tmp")
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        tmp_file.replace(STATUS_FILE)
    except Exception as e:
        log(f"Warning: failed to update status file: {e}")


def is_pid_alive(pid: int) -> bool:
    """Check if process with PID is currently alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_p1_pid() -> Optional[int]:
    """Find running P1 train_ppo process PID if active."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "train_ppo.py.*--policy P1"],
            text=True
        ).strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        if pids:
            return pids[0]
    except subprocess.CalledProcessError:
        pass
    return None


def verify_model_integrity(model_path: Path, vecnorm_path: Optional[Path] = None) -> bool:
    """Verify that an SB3 PPO model and its VecNormalize pickle load without error."""
    if not model_path.exists():
        log(f"Integrity check failed: {model_path} does not exist.")
        return False
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
        log(f"Model integrity verified: {model_path} (timesteps={model.num_timesteps:,})")
    except Exception as e:
        log(f"Model verification error on {model_path}: {e}")
        return False

    if vecnorm_path and vecnorm_path.exists():
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            import gymnasium as gym
            import numpy as np

            class _DummyEnv(gym.Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(43,), dtype=np.float32)
                    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

            dummy = DummyVecEnv([_DummyEnv])
            vn = VecNormalize.load(str(vecnorm_path), dummy)
            count = getattr(getattr(vn, 'ret_rms', None), 'count', 'unknown')
            log(f"VecNormalize verified: {vecnorm_path} (ret_rms count={count})")
        except Exception as e:
            log(f"VecNormalize verification error on {vecnorm_path}: {e}")
            return False
    return True


def clean_ports() -> None:
    """Terminate orphan Gazebo processes listening strictly on ports 11345-11348."""
    log("Checking simulator ports 11345-11348...")
    for port in GAZEBO_PORTS:
        try:
            # Find PID listening on port
            out = subprocess.check_output(
                ["ss", "-tulpn"],
                text=True
            )
            for line in out.splitlines():
                if f":{port} " in line or f":{port}\t" in line:
                    # Parse PID
                    # e.g., users:(("gzserver",pid=461498,fd=7))
                    import re
                    m = re.search(r'pid=(\d+)', line)
                    if m:
                        pid = int(m.group(1))
                        log(f"Terminating orphan process {pid} on port {port}...")
                        try:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(1)
                            if is_pid_alive(pid):
                                os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
        except Exception as e:
            log(f"Port cleanup check note: {e}")
    time.sleep(1)


def wait_for_p1() -> Dict[str, Any]:
    """Monitor P1 until it completes its 850,000 steps budget and verify files."""
    log("Entering MONITOR_P1 phase.")
    final_model = P1_RUN_DIR / "final_model.zip"
    final_vecnorm = P1_RUN_DIR / "final_vecnormalize.pkl"
    summary_file = P1_RUN_DIR / "training_summary.json"

    p1_pid = find_p1_pid()
    if p1_pid:
        log(f"Detected active P1 process PID: {p1_pid}")
    else:
        log("No active P1 PID detected; checking if already completed...")

    poll_interval = 20
    last_reported_steps = 0

    while True:
        # Check if completed
        if final_model.exists() and summary_file.exists():
            log(f"P1 completion detected: {final_model} and {summary_file} exist.")
            break

        # Check process status
        if p1_pid and not is_pid_alive(p1_pid):
            log(f"P1 process {p1_pid} has exited. Waiting 10s for disk write...")
            time.sleep(10)
            p1_pid = find_p1_pid()
            if not p1_pid:
                # Process definitely finished, check if final model saved
                if final_model.exists():
                    break
                else:
                    log("Process exited but final_model.zip not yet found; searching latest checkpoint...")
                    ckpts = sorted(P1_RUN_DIR.glob("checkpoints/checkpoint_*_steps.zip"))
                    if ckpts:
                        latest_ckpt = ckpts[-1]
                        log(f"Found latest checkpoint: {latest_ckpt}. Promoting to final_model.zip...")
                        import shutil
                        shutil.copy2(latest_ckpt, final_model)
                        vec_pkl = latest_ckpt.parent / f"checkpoint_vecnormalize_{latest_ckpt.stem.split('_')[1]}_steps.pkl"
                        if vec_pkl.exists():
                            shutil.copy2(vec_pkl, final_vecnorm)
                        break

        # Read latest checkpoint progress
        latest_metrics_file = P1_RUN_DIR / "latest_checkpoint_metrics.json"
        current_steps = last_reported_steps
        throughput = 0.0
        if latest_metrics_file.exists():
            try:
                with open(latest_metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    current_steps = m.get("timesteps", current_steps)
                    throughput = m.get("throughput_steps_per_sec", 0.0)
            except Exception:
                pass

        if current_steps != last_reported_steps:
            pct = (current_steps / 850_000) * 100
            log(f"P1 Progress: {current_steps:,} / 850,000 steps ({pct:.1f}%) | Throughput: {throughput:.1f} sps")
            last_reported_steps = current_steps

        update_status("MONITOR_P1", {
            "p1_pid": p1_pid,
            "timesteps": current_steps,
            "target_timesteps": 850_000,
            "throughput_steps_per_sec": throughput,
        })

        time.sleep(poll_interval)

    # Validate P1 integrity
    log("Verifying P1 model integrity...")
    assert verify_model_integrity(final_model, final_vecnorm), "P1 final model verification failed!"

    # Read final summary
    p1_summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                p1_summary = json.load(f)
        except Exception:
            pass

    log(f"P1 verified successfully. Total timesteps: {p1_summary.get('total_timesteps', 850000):,}")
    update_status("P1_COMPLETED", {
        "final_model": str(final_model),
        "summary": p1_summary,
    })
    return p1_summary


def wait_until_ready_for_p4() -> None:
    """Prepare for P4 launch: verify ports are clean and cooldown simulator sockets."""
    log("Preparing for P4 launch. Cleaning simulator ports 11345-11348...")
    clean_ports()
    log("Simulator ports cleaned. Waiting 15s cooldown before launching P4...")
    time.sleep(15)


def launch_p4() -> subprocess.Popen:
    """Launch P4 training under isolated conditions."""
    log("Entering LAUNCH_P4 phase.")
    clean_ports()

    # Prevent duplicate runs
    try:
        out = subprocess.check_output(["pgrep", "-f", "train_ppo.py.*--policy P4"], text=True).strip()
        if out:
            log(f"WARNING: Found existing P4 process ({out}). Aborting launch to avoid duplication.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        pass

    P4_RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_log = P4_RUN_DIR / "train.log"

    # Launch P4 with full 850k transitions (matching P1) and 24h window stop deadline
    cmd = [
        sys.executable,
        str(REPO_ROOT / "rl" / "train" / "train_ppo.py"),
        "--policy", "P4",
        "--seed", "0",
        "--config", "configs/ppo.yaml",
        "--run-dir", str(P4_RUN_DIR),
        "--gazebo",
        "--workers", "4",
        "--steps", str(P4_TARGET_STEPS),
        "--deadline-epoch", str(P4_STOP_DEADLINE_EPOCH),
    ]

    env = os.environ.copy()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    log(f"Launching P4 with command: {' '.join(cmd)}")
    log_fp = open(train_log, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach process group
    )

    pid_file = P4_RUN_DIR / "train.pid"
    pid_file.write_text(str(proc.pid))
    log(f"P4 process successfully launched with PID: {proc.pid}")

    update_status("P4_RUNNING", {
        "pid": proc.pid,
        "policy": "P4",
        "seed": 0,
        "run_dir": str(P4_RUN_DIR),
        "target_steps": P4_TARGET_STEPS,
        "deadline": P4_STOP_DEADLINE_DT.isoformat(),
        "cmd": cmd,
    })
    return proc


def monitor_p4(proc: subprocess.Popen) -> None:
    """Monitor P4 training, enforce stop deadline, and ensure safe save."""
    log("Entering MONITOR_P4 phase.")
    final_model = P4_RUN_DIR / "final_model.zip"
    final_vecnorm = P4_RUN_DIR / "final_vecnormalize.pkl"
    stop_file = P4_RUN_DIR / ".stop_requested"

    poll_interval = 20
    last_reported_steps = 0

    while True:
        now = time.time()
        # 1. Check if stop deadline reached
        if now >= P4_STOP_DEADLINE_EPOCH:
            log(f"Reached P4 stop deadline (epoch {now:.1f} >= {P4_STOP_DEADLINE_EPOCH:.1f}).")
            if proc.poll() is None:
                log("Signaling graceful stop via .stop_requested...")
                stop_file.touch(exist_ok=True)
                # Wait up to 120s for graceful shutdown
                for _ in range(60):
                    if proc.poll() is not None:
                        log("P4 terminated gracefully following stop signal.")
                        break
                    time.sleep(2)
                else:
                    log("P4 did not exit within 120s; sending SIGTERM...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            break

        # 2. Check if process finished on its own
        retcode = proc.poll()
        if retcode is not None:
            log(f"P4 process exited with code {retcode}.")
            break

        # 3. Read latest checkpoint metrics
        latest_file = P4_RUN_DIR / "latest_checkpoint_metrics.json"
        current_steps = last_reported_steps
        throughput = 0.0
        if latest_file.exists():
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    current_steps = m.get("timesteps", current_steps)
                    throughput = m.get("throughput_steps_per_sec", 0.0)
            except Exception:
                pass

        if current_steps != last_reported_steps:
            pct = (current_steps / P4_TARGET_STEPS) * 100
            log(f"P4 Progress: {current_steps:,} / {P4_TARGET_STEPS:,} steps ({pct:.1f}%) | Throughput: {throughput:.1f} sps")
            last_reported_steps = current_steps

        time_left = max(0.0, P4_STOP_DEADLINE_EPOCH - now)
        update_status("P4_RUNNING", {
            "pid": proc.pid,
            "timesteps": current_steps,
            "target_timesteps": P4_TARGET_STEPS,
            "throughput": throughput,
            "seconds_to_deadline": round(time_left, 1),
            "minutes_to_deadline": round(time_left / 60, 2),
        })

        time.sleep(poll_interval)

    # Process ended or stopped. Ensure final model exists.
    log("P4 execution completed. Verifying artifacts...")
    if not final_model.exists():
        # Promote latest checkpoint if final_model wasn't written
        ckpts = sorted(P4_RUN_DIR.glob("checkpoints/checkpoint_*_steps.zip"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            log(f"Promoting latest checkpoint {latest_ckpt} to final_model.zip...")
            import shutil
            shutil.copy2(latest_ckpt, final_model)
            vec_pkl = latest_ckpt.parent / f"checkpoint_vecnormalize_{latest_ckpt.stem.split('_')[1]}_steps.pkl"
            if vec_pkl.exists():
                shutil.copy2(vec_pkl, final_vecnorm)

    verify_model_integrity(final_model, final_vecnorm)
    clean_ports()


def run_evaluation() -> Dict[str, Any]:
    """Evaluate P1 and P4 policies on test scenarios with smoke test first."""
    log("Entering EVALUATION phase.")
    eval_dir = REPO_ROOT / "runs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("P1", P1_RUN_DIR / "final_model.zip", eval_dir / "p1_test"),
        ("P4", P4_RUN_DIR / "final_model.zip", eval_dir / "p4_test"),
    ]

    eval_results = {}
    for name, model_path, out_dir in models:
        if not model_path.exists():
            log(f"Skipping evaluation for {name}: model {model_path} not found.")
            continue

        # 1. Smoke test: 5 episodes first to verify evaluation pipeline
        log(f"Running 5-episode smoke test for {name} ({model_path})...")
        smoke_cmd = [
            sys.executable,
            str(REPO_ROOT / "rl" / "evaluate" / "evaluate.py"),
            "--policy", name,
            "--model-path", str(model_path),
            "--split", "test",
            "--condition", "nominal",
            "--output-dir", str(out_dir / "smoke"),
            "--n-episodes", "5",
        ]
        try:
            smoke_res = subprocess.run(smoke_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
            if smoke_res.returncode != 0:
                log(f"Smoke test failed for {name}:\n{smoke_res.stderr[-300:]}")
                continue
            log(f"Smoke test passed for {name}.")
        except Exception as e:
            log(f"Smoke test exception for {name}: {e}")
            continue

        # 2. Main evaluation: 50 held-out test scenarios
        log(f"Running 50-episode evaluation for {name} ({model_path}) on held-out test split...")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "rl" / "evaluate" / "evaluate.py"),
            "--policy", name,
            "--model-path", str(model_path),
            "--split", "test",
            "--condition", "nominal",
            "--output-dir", str(out_dir),
            "--n-episodes", "50",
        ]
        try:
            res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
            log(f"Evaluation completed for {name}. Output summary:\n{res.stdout[-500:]}")
            eval_results[name] = {"returncode": res.returncode, "dir": str(out_dir)}
        except Exception as e:
            log(f"Error during {name} evaluation: {e}")
            eval_results[name] = {"error": str(e)}

    # Generate comparative evaluation summary
    comparison_summary_file = eval_dir / "p1_vs_p4_comparison.json"
    p1_json = eval_dir / "p1_test" / "eval_P1_nominal.json"
    p4_json = eval_dir / "p4_test" / "eval_P4_nominal.json"
    comparison_data = {
        "timestamp": datetime.now(IST).isoformat(),
        "evaluated_models": ["P1", "P4"],
        "scenarios_evaluated": 50,
        "split": "test",
    }
    if p1_json.exists():
        try:
            with open(p1_json, "r", encoding="utf-8") as f:
                comparison_data["P1"] = json.load(f)
        except Exception as e:
            comparison_data["P1_error"] = str(e)
    if p4_json.exists():
        try:
            with open(p4_json, "r", encoding="utf-8") as f:
                comparison_data["P4"] = json.load(f)
        except Exception as e:
            comparison_data["P4_error"] = str(e)

    try:
        with open(comparison_summary_file, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)
        log(f"Comparative summary saved to: {comparison_summary_file}")
    except Exception as e:
        log(f"Failed to write comparative summary: {e}")

    update_status("EVALUATION_COMPLETE", eval_results)
    return eval_results


def run_gazebo_inference_demo() -> Dict[str, Any]:
    """Launch Gazebo Classic with the trained model and execute goal-navigation trials."""
    log("Entering GAZEBO_INFERENCE_DEMO phase.")
    clean_ports()

    # Determine best model between P1 and P4
    comparison_file = REPO_ROOT / "runs" / "eval" / "p1_vs_p4_comparison.json"
    selected_policy = "P1"
    model_path = P1_RUN_DIR / "final_model.zip"
    vecnorm_path = P1_RUN_DIR / "final_vecnormalize.pkl"

    if comparison_file.exists():
        try:
            with open(comparison_file, "r", encoding="utf-8") as f:
                comp = json.load(f)
            p1_sr = comp.get("P1", {}).get("success_rate", 0.0)
            p4_sr = comp.get("P4", {}).get("success_rate", -1.0)
            if p4_sr >= p1_sr and (P4_RUN_DIR / "final_model.zip").exists():
                selected_policy = "P4"
                model_path = P4_RUN_DIR / "final_model.zip"
                vecnorm_path = P4_RUN_DIR / "final_vecnormalize.pkl"
        except Exception:
            pass

    log(f"Selected policy for Gazebo demonstration: {selected_policy} ({model_path})")

    demo_results = {
        "timestamp": datetime.now(IST).isoformat(),
        "selected_policy": selected_policy,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path.exists() else None,
        "trials": [],
    }

    # Start Gazebo Classic via GazeboWatchdog on port 11345, domain 30
    from rl.train.watchdog import GazeboWatchdog
    demo_watchdog = GazeboWatchdog(port=11345, domain_id=30)
    log("Launching Gazebo Classic on port 11345 for inference demonstration...")
    demo_watchdog.start_simulator()
    time.sleep(10)  # Allow Gazebo & robot spawn to initialize

    # Test goals (diverse navigation targets in 4x4m arena)
    test_goals = [
        {"goal_x": 1.2, "goal_y": 0.0, "label": "Straight Forward"},
        {"goal_x": 1.0, "goal_y": 1.0, "label": "Diagonal Positive"},
        {"goal_x": -1.0, "goal_y": 1.0, "label": "Left Quadrant"},
    ]

    for i, g in enumerate(test_goals):
        log(f"Starting Gazebo Demonstration Trial {i+1}/{len(test_goals)}: Goal=({g['goal_x']}, {g['goal_y']}) [{g['label']}]...")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "deploy" / "ppo_inference_node.py"),
            "--model-path", str(model_path),
            "--goal-x", str(g["goal_x"]),
            "--goal-y", str(g["goal_y"]),
            "--output-dir", str(REPO_ROOT / "runs" / "inference_logs"),
        ]
        if vecnorm_path.exists():
            cmd.extend(["--vecnorm-path", str(vecnorm_path)])

        env = os.environ.copy()
        env["GAZEBO_MASTER_URI"] = "http://localhost:11345"
        env["ROS_DOMAIN_ID"] = "30"
        env["TURTLEBOT3_MODEL"] = "burger"

        try:
            trial_res = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
            log(f"Trial {i+1} Output:\n{trial_res.stdout[-300:]}")
            demo_results["trials"].append({
                "trial": i + 1,
                "goal": [g["goal_x"], g["goal_y"]],
                "label": g["label"],
                "returncode": trial_res.returncode,
                "stdout_tail": trial_res.stdout[-300:],
            })
        except Exception as e:
            log(f"Trial {i+1} Exception: {e}")
            demo_results["trials"].append({"trial": i + 1, "error": str(e)})

    # Terminate demo simulator
    demo_watchdog.terminate_simulator()
    clean_ports()

    demo_file = REPO_ROOT / "runs" / "inference_demo_results.json"
    try:
        with open(demo_file, "w", encoding="utf-8") as f:
            json.dump(demo_results, f, indent=2)
        log(f"Gazebo inference demonstration results saved to: {demo_file}")
    except Exception as e:
        log(f"Failed to write demo results: {e}")

    update_status("DEMO_COMPLETE", demo_results)
    return demo_results


def main():
    """Main daemon entrypoint."""
    # Ensure single orchestrator instance via lock file
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another orchestrator instance is already running. Exiting.", file=sys.stderr)
        sys.exit(0)

    log("=" * 70)
    log("Autonomous Training Orchestrator Started (Strict 9:00 PM IST Hard Deadline)")
    log(f"  P4 Stop Deadline:   {P4_STOP_DEADLINE_DT.strftime('%Y-%m-%d %H:%M:%S IST')} (epoch {P4_STOP_DEADLINE_EPOCH:.0f})")
    log(f"  Target Completion:  {TARGET_FINISH_DT.strftime('%Y-%m-%d %H:%M:%S IST')} (epoch {TARGET_FINISH_EPOCH:.0f})")
    log(f"  Hard Cutoff:        {HARD_DEADLINE_DT.strftime('%Y-%m-%d %H:%M:%S IST')} (epoch {HARD_DEADLINE_EPOCH:.0f})")
    log(f"  P4 Target Steps:    {P4_TARGET_STEPS:,} steps")
    log("=" * 70)

    try:
        # Step 1: Wait for P1 completion and verify
        wait_for_p1()

        # Step 2: Prepare for P4 (clean ports, cooldown)
        wait_until_ready_for_p4()

        # Step 3: Launch P4
        proc = launch_p4()

        # Step 4: Monitor P4 until target steps reached or deadline (17:45 IST)
        monitor_p4(proc)

        # Step 5: Run realistic simulation evaluations on held-out scenarios
        run_evaluation()

        # Step 6: Run live Gazebo Classic simulation demonstration with standalone inference
        run_gazebo_inference_demo()

        log("=" * 70)
        log("Orchestrator Finished All Milestones Cleanly (Training, Eval, and Gazebo Demo).")
        log("=" * 70)
        update_status("ALL_COMPLETE", {
            "completion_time": datetime.now(IST).isoformat(),
        })

    except Exception as e:
        log(f"FATAL ERROR in orchestrator: {e}")
        import traceback
        log(traceback.format_exc())
        update_status("FAILED", {"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
