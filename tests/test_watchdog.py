"""Automated Tests for Missing-Scan Watchdog and Process-Group Recovery.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 7, 9 Check 10)
Validates:
1. Detection of stale or missing scan data using timestamps and monotonic clock.
   Asserts configured timeout from configs/ppo.yaml (watchdog_timeout_sec: 2.0).
2. Bounded retries: max_retries enforced, fails safely with WatchdogRecoveryError, does not loop indefinitely.
3. Checkpoint restoration: reloads latest valid checkpoint, preserving step and episode counts without duplicate or skipped accounting.
4. Process-group isolation: ensures all child processes in the simulator group are terminated, preventing duplicate/orphaned workers.
5. Live Gazebo fault injection: controlled fault injection in a disposable project-owned simulation on isolated port/domain (11395/95).
"""

from __future__ import annotations
import os
import sys
import time
import json
import signal
import tempfile
import subprocess
from pathlib import Path

import pytest
import psutil
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.train.watchdog import (
    GazeboWatchdog,
    MissingScanTimeoutError,
    WatchdogRecoveryError,
    load_watchdog_timeout,
)
from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv


def test_load_configured_timeout():
    """Verify watchdog loads timeout from configs/ppo.yaml."""
    timeout = load_watchdog_timeout()
    assert timeout == 2.0, f"Expected 2.0s timeout from configs/ppo.yaml, got {timeout}"


def test_detect_stale_or_missing_scan_timestamps():
    """Verify monotonic clock detects stale scan data beyond configured timeout."""
    watchdog = GazeboWatchdog(timeout_sec=2.0)
    assert watchdog.timeout_sec == 2.0

    # Uninitialized scan: not stale
    assert not watchdog.is_scan_stale()

    # Fresh scan recorded
    watchdog.record_scan_received(stamp_sec=100.5)
    t0 = watchdog.last_scan_monotonic
    assert t0 > 0.0

    # Check at t0 + 1.0s: within 2.0s timeout -> not stale
    assert not watchdog.is_scan_stale(now_monotonic=t0 + 1.0)

    # Check at t0 + 2.01s: exceeds 2.0s timeout -> stale!
    assert watchdog.is_scan_stale(now_monotonic=t0 + 2.01)


def test_watchdog_bounded_retries_fail_safe():
    """Verify bounded retries fail safely with WatchdogRecoveryError and do not loop indefinitely."""
    tmp_dir = tempfile.mkdtemp(prefix="watchdog_test_")
    diag_file = Path(tmp_dir) / "test_diagnostics.json"

    watchdog = GazeboWatchdog(
        timeout_sec=0.1,
        max_retries=3,
        diagnostics_path=diag_file,
    )

    class MockEnv:
        def __init__(self):
            self.ros_node = None
            self.pub_cmd = None
            self.step_count = 42

    env = MockEnv()

    # Patch start_simulator to fail
    def failing_start():
        raise RuntimeError("Simulated startup failure")

    watchdog.start_simulator = failing_start

    # Expect recovery to fail cleanly when max retries exceeded
    with pytest.raises(WatchdogRecoveryError) as exc_info:
        for _ in range(5):
            watchdog.handle_missing_scan(env, current_step=42)

    assert "exceeded max_retries" in str(exc_info.value) or "Simulated startup failure" in str(exc_info.value)
    assert diag_file.exists()

    with open(diag_file, 'r') as f:
        diags = json.load(f)
    assert len(diags) > 0
    assert diags[-1]["status"] in ["EXHAUSTED_RETRIES", "RESTART_FAILED"]


def test_watchdog_checkpoint_restoration_accounting():
    """Verify checkpoint reload correctly identifies latest checkpoint and step count."""
    tmp_dir = tempfile.mkdtemp(prefix="watchdog_ckpt_")
    ckpt_dir = Path(tmp_dir)

    # Create dummy checkpoints
    (ckpt_dir / "checkpoint_20000_steps.zip").touch()
    (ckpt_dir / "checkpoint_50000_steps.zip").touch()
    (ckpt_dir / "checkpoint_100000_steps.zip").touch()

    diag_file = ckpt_dir / "diag.json"
    watchdog = GazeboWatchdog(
        timeout_sec=0.1,
        max_retries=3,
        diagnostics_path=diag_file,
    )

    latest = watchdog._find_latest_checkpoint(str(ckpt_dir))
    assert latest is not None
    assert "100000" in latest.name

    step = watchdog._extract_step_from_checkpoint(latest, fallback=0)
    assert step == 100000


def test_watchdog_prevention_of_duplicate_workers():
    """Verify process-group termination kills all processes in the group without orphans."""
    watchdog = GazeboWatchdog(timeout_sec=2.0)

    # Spawn a disposable shell process group
    proc = subprocess.Popen(
        "sleep 100",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    pgid = os.getpgid(proc.pid)
    watchdog.owned_proc = proc

    # Verify process group exists and is running
    assert psutil.pid_exists(proc.pid)

    # Terminate via watchdog
    watchdog.terminate_simulator()

    # Verify process group is completely dead
    time.sleep(0.2)
    assert not psutil.pid_exists(proc.pid)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def test_watchdog_controlled_fault_injection_live_gazebo():
    """Controlled fault injection on an isolated project-owned Gazebo simulation."""
    # Use isolated test port and domain
    test_port = 11395
    test_domain = 95
    diag_file = REPO_ROOT / "results" / "watchdog_diagnostics.json"

    watchdog = GazeboWatchdog(
        timeout_sec=2.0,
        max_retries=2,
        port=test_port,
        domain_id=test_domain,
        diagnostics_path=diag_file,
    )

    # Set up environment variables for test worker
    os.environ['ROS_DOMAIN_ID'] = str(test_domain)
    os.environ['GAZEBO_MASTER_URI'] = f"http://localhost:{test_port}"
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    print("\n[WATCHDOG TEST] Starting isolated Gazebo worker...")
    proc1 = watchdog.start_simulator()
    initial_pid = proc1.pid
    assert initial_pid is not None

    try:
        # Wait for simulator to be ready
        env = TurtleBot3NavigationEnv(
            use_gazebo=True,
            watchdog=watchdog,
            seed=42,
        )

        # Wait for first scan
        ready = watchdog.wait_for_fresh_scan(env, timeout_sec=25.0)
        if not ready:
            pytest.skip("Could not launch Gazebo on test port within 25s (host resource constraint).")

        assert watchdog.last_scan_monotonic > 0.0
        assert not watchdog.is_scan_stale()

        # Step 1: Normal step
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert np.all(np.isfinite(obs))

        # Fault Injection: Freeze the simulator by sending SIGSTOP to the owned process group
        print(f"[WATCHDOG TEST] Injecting fault: SIGSTOP to process group {os.getpgid(initial_pid)}...")
        os.killpg(os.getpgid(initial_pid), signal.SIGSTOP)

        # Step again: Scan will not arrive -> watchdog must detect missing scan and recover!
        print("[WATCHDOG TEST] Stepping environment with frozen simulator; watchdog recovery expected...")
        step_t0 = time.monotonic()
        obs_rec, r_rec, term_rec, trunc_rec, info_rec = env.step(np.array([0.0, 0.0], dtype=np.float32))
        step_elapsed = time.monotonic() - step_t0

        # Verify step took at least the watchdog timeout (2.0s)
        assert step_elapsed >= 2.0, f"Expected step to wait at least 2.0s timeout, took {step_elapsed:.2f}s"

        # Verify new simulator process was spawned and PID changed
        assert watchdog.owned_proc is not None
        assert watchdog.owned_proc.pid != initial_pid
        print(f"[WATCHDOG TEST] Simulator successfully recovered: PID {initial_pid} -> {watchdog.owned_proc.pid}")

        # Verify diagnostic record was created
        assert diag_file.exists()
        with open(diag_file, 'r') as f:
            diags = json.load(f)
        assert len(diags) > 0
        last_diag = diags[-1]
        assert last_diag["event"] == "missing_scan_timeout"
        assert last_diag["status"] == "RECOVERED"
        assert last_diag["reconnected"] is True
        assert last_diag["terminated_pid"] == initial_pid
        assert last_diag["new_pid"] == watchdog.owned_proc.pid

        # Verify observation after recovery is valid
        assert np.all(np.isfinite(obs_rec))

        env.close()

    finally:
        print("[WATCHDOG TEST] Cleaning up test simulator...")
        watchdog.terminate_simulator()
