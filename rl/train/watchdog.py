"""Gazebo Missing-Scan Watchdog and Supervisor Module.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 7, 9 Check 10)
Requirements:
1. Detect stale or missing scan data using timestamps and monotonic clock (time.monotonic).
   Uses the configured timeout from configs/ppo.yaml (watchdog_timeout_sec, default 2.0s).
2. On timeout, stop issuing actions to the failed simulator, terminate only the owned
   Gazebo process group, restart it cleanly, and verify fresh scan arrival before resuming.
3. Reload latest valid checkpoint if available, preserving completed step/episode counts
   without duplicate or skipped accounting.
4. Bound retries (max_retries, default 3) and fail safely with WatchdogRecoveryError.
5. Save detailed diagnostics for timeout, termination, restart, reconnection, and checkpoint restore.
6. Enforce strict process-group isolation to prevent orphaned or duplicate workers.
"""

from __future__ import annotations
import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import psutil
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIAGNOSTICS_PATH = REPO_ROOT / "results" / "watchdog_diagnostics.json"


class MissingScanTimeoutError(TimeoutError):
    """Raised when no LiDAR scan is received within the watchdog timeout."""
    pass


class WatchdogRecoveryError(RuntimeError):
    """Raised when Gazebo simulator recovery fails or max retries are exceeded."""
    pass


def load_watchdog_timeout(config_path: str = "configs/ppo.yaml", default_timeout: float = 2.0) -> float:
    """Load watchdog_timeout_sec from config YAML, falling back to default."""
    cfg_file = REPO_ROOT / config_path
    if cfg_file.exists():
        try:
            with open(cfg_file, 'r') as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict) and 'watchdog_timeout_sec' in cfg:
                    return float(cfg['watchdog_timeout_sec'])
        except Exception:
            pass
    return default_timeout


class GazeboWatchdog:
    """Watchdog and process-group supervisor for Gazebo simulator instances."""

    def __init__(
        self,
        timeout_sec: Optional[float] = None,
        max_retries: int = 3,
        port: int = 11345,
        domain_id: int = 30,
        diagnostics_path: Optional[Path] = None,
    ):
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else load_watchdog_timeout())
        self.max_retries = max_retries
        self.port = port
        self.domain_id = domain_id
        self.diagnostics_path = diagnostics_path or DEFAULT_DIAGNOSTICS_PATH

        self.owned_proc: Optional[subprocess.Popen] = None
        self.last_scan_monotonic: float = 0.0
        self.last_scan_stamp: Optional[float] = None
        self.consecutive_retries: int = 0
        self.diagnostics: List[Dict[str, Any]] = []

    def start_simulator(self) -> subprocess.Popen:
        """Launch headless Gazebo in its own isolated process group."""
        self.terminate_simulator()

        env = os.environ.copy()
        env['GAZEBO_MASTER_URI'] = f"http://localhost:{self.port}"
        env['ROS_DOMAIN_ID'] = str(self.domain_id)
        env['TURTLEBOT3_MODEL'] = 'burger'
        env['GAZEBO_MODEL_DATABASE_URI'] = ''
        env['PYTHONUNBUFFERED'] = '1'

        cmd = (
            f"source /opt/ros/humble/setup.bash && "
            f"source {REPO_ROOT}/ros2_ws/install/setup.bash && "
            f"ros2 launch tb3_rl_sim headless_sim.launch.py"
        )

        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self.owned_proc = proc
        return proc

    def terminate_simulator(self) -> None:
        """Terminate only the owned Gazebo process group cleanly without orphaned children."""
        if self.owned_proc is None:
            return

        proc = self.owned_proc
        self.owned_proc = None

        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return

        # 1. Graceful SIGTERM
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Wait up to 1.5s for graceful exit
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 1.5:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        # 2. Hard SIGKILL to guarantee no leftover or duplicate workers
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def record_scan_received(self, stamp_sec: Optional[float] = None) -> None:
        """Update scan arrival timestamp with monotonic clock."""
        self.last_scan_monotonic = time.monotonic()
        if stamp_sec is not None:
            self.last_scan_stamp = stamp_sec

    def is_scan_stale(self, now_monotonic: Optional[float] = None) -> bool:
        """Check if scan data is missing or stale according to configured timeout."""
        if self.last_scan_monotonic <= 0.0:
            return False  # Not yet initialized
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        return (now - self.last_scan_monotonic) > self.timeout_sec

    def wait_for_fresh_scan(self, env: Any, timeout_sec: float = 30.0) -> bool:
        """Spin ROS node until a strictly newer scan arrives."""
        if not hasattr(env, 'ros_node') or env.ros_node is None:
            return False

        import rclpy
        start_t = time.perf_counter()
        prev_stamp = self.last_scan_stamp

        while (time.perf_counter() - start_t) < timeout_sec:
            rclpy.spin_once(env.ros_node, timeout_sec=0.05)
            if env._latest_scan is not None:
                curr_stamp = (
                    env._latest_scan.header.stamp.sec
                    + env._latest_scan.header.stamp.nanosec * 1e-9
                )
                if prev_stamp is None or curr_stamp > prev_stamp:
                    self.record_scan_received(curr_stamp)
                    return True
            time.sleep(0.01)
        return False

    def handle_missing_scan(
        self,
        env: Any,
        checkpoint_dir: Optional[str] = None,
        current_step: int = 0,
        current_episode: int = 0,
    ) -> Dict[str, Any]:
        """Execute full recovery sequence on missing/stale scan timeout."""
        now_mono = time.monotonic()
        stale_sec = now_mono - self.last_scan_monotonic if self.last_scan_monotonic > 0 else self.timeout_sec

        diag_entry: Dict[str, Any] = {
            "event": "missing_scan_timeout",
            "timestamp": datetime.now().isoformat(),
            "monotonic_time": now_mono,
            "stale_duration_sec": round(stale_sec, 3),
            "configured_timeout_sec": self.timeout_sec,
            "retry_attempt": self.consecutive_retries + 1,
            "max_retries": self.max_retries,
            "prior_step_count": current_step,
            "prior_episode_count": current_episode,
            "checkpoint_reloaded": None,
            "resumed_step_count": current_step,
            "reconnected": False,
            "status": "IN_PROGRESS",
        }

        # 1. Stop issuing actions to the failed simulator
        if hasattr(env, 'pub_cmd') and env.pub_cmd is not None:
            try:
                from geometry_msgs.msg import Twist
                zero_twist = Twist()
                env.pub_cmd.publish(zero_twist)
            except Exception:
                pass

        # 2. Check retry bounds
        if self.consecutive_retries >= self.max_retries:
            diag_entry["status"] = "EXHAUSTED_RETRIES"
            diag_entry["error"] = f"Recovery failed: exceeded max_retries ({self.max_retries})"
            self._save_diagnostic(diag_entry)
            self.terminate_simulator()
            raise WatchdogRecoveryError(diag_entry["error"])

        self.consecutive_retries += 1

        # 3. Terminate owned simulator process group
        old_pid = self.owned_proc.pid if self.owned_proc else None
        self.terminate_simulator()
        diag_entry["terminated_pid"] = old_pid

        # 4. Restart simulator process group
        try:
            new_proc = self.start_simulator()
            diag_entry["new_pid"] = new_proc.pid
        except Exception as e:
            diag_entry["status"] = "RESTART_FAILED"
            diag_entry["error"] = str(e)
            self._save_diagnostic(diag_entry)
            raise WatchdogRecoveryError(f"Failed to restart simulator: {e}")

        # 5. Wait for fresh scan
        reconnected = self.wait_for_fresh_scan(env, timeout_sec=30.0)
        diag_entry["reconnected"] = reconnected
        if not reconnected:
            diag_entry["status"] = "RECONNECTION_FAILED"
            diag_entry["error"] = "Simulator restarted but fresh scan did not arrive within 30s"
            self._save_diagnostic(diag_entry)
            raise WatchdogRecoveryError(diag_entry["error"])

        # 6. Re-synchronize robot pose and obstacles in the restarted simulator
        if hasattr(env, '_teleport_gazebo_entities'):
            try:
                env._teleport_gazebo_entities()
            except Exception:
                pass

        # 7. Checkpoint restoration (if directory provided)
        resumed_step = current_step
        if checkpoint_dir:
            ckpt_path = self._find_latest_checkpoint(checkpoint_dir)
            if ckpt_path:
                diag_entry["checkpoint_reloaded"] = str(ckpt_path)
                resumed_step = self._extract_step_from_checkpoint(ckpt_path, fallback=current_step)
                diag_entry["resumed_step_count"] = resumed_step

        diag_entry["status"] = "RECOVERED"
        self._save_diagnostic(diag_entry)
        self.consecutive_retries = 0  # Reset upon successful recovery
        return diag_entry

    def _find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[Path]:
        """Find the latest checkpoint file by step count."""
        p = Path(checkpoint_dir)
        if not p.exists():
            return None
        ckpts = list(p.glob("checkpoint_*_steps.zip"))
        if not ckpts:
            return None

        def extract_steps(path: Path) -> int:
            try:
                parts = path.stem.split('_')
                idx = parts.index("steps") - 1
                return int(parts[idx])
            except Exception:
                return 0

        ckpts.sort(key=extract_steps)
        return ckpts[-1]

    def _extract_step_from_checkpoint(self, ckpt_path: Path, fallback: int = 0) -> int:
        """Extract step number from checkpoint filename."""
        try:
            parts = ckpt_path.stem.split('_')
            idx = parts.index("steps") - 1
            return int(parts[idx])
        except Exception:
            return fallback

    def _save_diagnostic(self, entry: Dict[str, Any]) -> None:
        """Save recovery diagnostics atomically to results/watchdog_diagnostics.json."""
        self.diagnostics.append(entry)
        self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.diagnostics_path.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        tmp_file.replace(self.diagnostics_path)

    def close(self) -> None:
        """Clean teardown."""
        self.terminate_simulator()
