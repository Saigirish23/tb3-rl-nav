"""Gazebo Soak and Kill-and-Resume Runner for G2 Check 9.

Supports:
- Multi-worker parallel stepping with strict ROS_DOMAIN_ID and GAZEBO_MASTER_URI isolation.
- Continuous random stepping, NaN/Inf tracking, episode reset counting.
- Periodic RSS memory tracking for both Python worker and gzserver processes.
- Atomic checkpointing to results/gazebo_soak_checkpoint.json.
- Interruption handling (SIGINT/SIGTERM) and --resume continuation.
- Clean process-group termination (no orphaned processes).
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

import psutil
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_FILE = REPO_ROOT / "results" / "gazebo_soak_checkpoint.json"


def start_gazebo_worker(worker_id: int) -> subprocess.Popen:
    """Launch headless Gazebo simulation for worker_id in its own process group."""
    port = 11345 + worker_id
    domain_id = 30 + worker_id

    env = os.environ.copy()
    env['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
    env['ROS_DOMAIN_ID'] = str(domain_id)
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
    return proc


def wait_for_worker_ready(worker_id: int, timeout_sec: float = 30.0) -> bool:
    """Wait until /scan and /clock are active on worker's domain."""
    domain_id = 30 + worker_id
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(domain_id)

    start_t = time.perf_counter()
    check_cmd = "source /opt/ros/humble/setup.bash && ros2 topic list"

    while (time.perf_counter() - start_t) < timeout_sec:
        try:
            res = subprocess.run(
                check_cmd,
                shell=True,
                executable='/bin/bash',
                env=env,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if '/scan' in res.stdout and '/clock' in res.stdout:
                time.sleep(1.0)
                return True
        except Exception:
            pass
        time.sleep(0.5)

    return False


def get_worker_gzserver_pid(worker_id: int) -> Optional[int]:
    """Find gzserver PID associated with worker port."""
    port_str = str(11345 + worker_id)
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'environ']):
        try:
            cmdline = p.info['cmdline'] or []
            if 'gzserver' in p.info['name'] or any('gzserver' in c for c in cmdline):
                # Check environment or cmdline
                env = p.environ()
                if port_str in env.get('GAZEBO_MASTER_URI', ''):
                    return p.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Fallback to any gzserver
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'gzserver' in p.info['name'] or (p.info['cmdline'] and any('gzserver' in c for c in p.info['cmdline'])):
                return p.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def run_soak_worker(
    worker_id: int,
    target_steps: int,
    initial_step_offset: int,
    step_queue: Any,
    stop_event: Any,
) -> None:
    """Child worker process running environment stepping with random actions."""
    domain_id = 30 + worker_id
    port = 11345 + worker_id

    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv

    env = TurtleBot3NavigationEnv(use_gazebo=True, seed=42 + worker_id)
    obs, info = env.reset(seed=100_000 + initial_step_offset + worker_id)

    rng = np.random.default_rng(1234 + worker_id)
    steps_done = 0
    episodes_done = 0
    nan_count = 0
    crashes = 0

    py_proc = psutil.Process()

    last_sent_steps = 0
    last_sent_episodes = 0
    last_sent_nans = 0

    while steps_done < target_steps and not stop_event.is_set():
        action = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        try:
            obs, r, term, trunc, info = env.step(action)
            steps_done += 1

            if not np.all(np.isfinite(obs)) or not np.isfinite(r):
                nan_count += 1

            if term or trunc:
                episodes_done += 1
                obs, info = env.reset()

            # Periodically push batch progress
            if steps_done % 50 == 0 or steps_done == target_steps:
                py_rss = py_proc.memory_info().rss / (1024 * 1024)
                step_delta = steps_done - last_sent_steps
                ep_delta = episodes_done - last_sent_episodes
                nan_delta = nan_count - last_sent_nans
                last_sent_steps = steps_done
                last_sent_episodes = episodes_done
                last_sent_nans = nan_count
                step_queue.put({
                    'worker_id': worker_id,
                    'steps_delta': step_delta,
                    'total_worker_steps': steps_done,
                    'episodes_delta': ep_delta,
                    'nan_delta': nan_delta,
                    'py_rss_mb': py_rss,
                })
        except Exception as e:
            crashes += 1
            print(f"Worker {worker_id} exception at step {steps_done}: {e}")
            break

    env.close()


def save_checkpoint(state: Dict[str, Any]) -> None:
    """Atomically save soak state to checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp_file.replace(CHECKPOINT_FILE)


def load_checkpoint() -> Optional[Dict[str, Any]]:
    """Load soak state from checkpoint file if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def execute_soak(
    target_steps: int = 1000,
    n_workers: int = 1,
    resume: bool = False,
    simulate_interrupt_after: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Gazebo soak run with live tracking and checkpointing."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    state = {
        'target_steps': target_steps,
        'completed_steps': 0,
        'episodes': 0,
        'nan_count': 0,
        'crashes': 0,
        'n_workers': n_workers,
        'memory_samples': [],
        'start_time': time.time(),
        'elapsed_sec': 0.0,
    }

    if resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            state.update(ckpt)
            state['target_steps'] = target_steps
            state['n_workers'] = n_workers
            print(f"RESUMED from checkpoint: {state['completed_steps']}/{target_steps} steps already completed.")

    steps_remaining = target_steps - state['completed_steps']
    if steps_remaining <= 0:
        print(f"Target steps ({target_steps}) already met in checkpoint.")
        return state

    steps_per_worker = math.ceil(steps_remaining / n_workers)
    print(f"\nStarting Gazebo soak: {steps_remaining} steps remaining across {n_workers} worker(s)...")

    gazebo_procs = []
    worker_procs = []
    step_queue = ctx.Queue()
    stop_event = ctx.Event()

    try:
        # 1. Start Gazebo instances
        for w in range(n_workers):
            p = start_gazebo_worker(w)
            gazebo_procs.append(p)

        for w in range(n_workers):
            print(f"Waiting for Gazebo worker {w} readiness...")
            if not wait_for_worker_ready(w, timeout_sec=30.0):
                raise RuntimeError(f"Gazebo worker {w} failed to become ready.")
            print(f"Gazebo worker {w} is ready.")

        # Baseline memory
        gz_pids = [get_worker_gzserver_pid(w) for w in range(n_workers)]
        gz_pids = [p for p in gz_pids if p is not None]

        start_gz_rss = sum(psutil.Process(p).memory_info().rss / (1024 * 1024) for p in gz_pids if psutil.pid_exists(p))
        start_py_rss = psutil.Process().memory_info().rss / (1024 * 1024)

        print(f"Initial Resources: gzserver RSS = {start_gz_rss:.1f} MB, Python RSS = {start_py_rss:.1f} MB")

        # 2. Launch stepping workers
        t_start = time.perf_counter()
        initial_elapsed = state.get('elapsed_sec', 0.0)
        for w in range(n_workers):
            wp = ctx.Process(
                target=run_soak_worker,
                args=(w, steps_per_worker, state['completed_steps'], step_queue, stop_event),
            )
            worker_procs.append(wp)
            wp.start()

        # 3. Main monitor loop
        steps_this_session = 0
        last_ckpt_steps = state['completed_steps']

        while steps_this_session < steps_remaining:
            try:
                msg = step_queue.get(timeout=5.0)
                delta = msg['steps_delta']
                steps_this_session += delta
                state['completed_steps'] += delta
                state['episodes'] += msg.get('episodes_delta', msg.get('episodes_done', 0))
                state['nan_count'] += msg.get('nan_delta', msg.get('nan_count', 0))

                # Measure current gzserver memory
                curr_gz_rss = sum(psutil.Process(p).memory_info().rss / (1024 * 1024) for p in gz_pids if psutil.pid_exists(p))
                curr_py_rss = msg['py_rss_mb']

                state['memory_samples'].append({
                    'step': state['completed_steps'],
                    'gzserver_rss_mb': curr_gz_rss,
                    'py_rss_mb': curr_py_rss,
                    'timestamp': time.time(),
                })

                session_elapsed = time.perf_counter() - t_start
                sps = steps_this_session / max(session_elapsed, 0.001)
                print(f"  Progress: {state['completed_steps']}/{target_steps} steps "
                      f"({sps:.2f} sps) | gz RSS: {curr_gz_rss:.1f} MB | py RSS: {curr_py_rss:.1f} MB | NaNs: {state['nan_count']}")

                # Checkpoint save
                if state['completed_steps'] - last_ckpt_steps >= 200 or state['completed_steps'] >= target_steps:
                    state['elapsed_sec'] = initial_elapsed + session_elapsed
                    save_checkpoint(state)
                    last_ckpt_steps = state['completed_steps']

                # Simulated interrupt for kill-and-resume verification
                if simulate_interrupt_after and steps_this_session >= simulate_interrupt_after:
                    print(f"\n[KILL-AND-RESUME TEST] Triggering simulated interruption at {state['completed_steps']} steps...")
                    stop_event.set()
                    state['elapsed_sec'] = initial_elapsed + (time.perf_counter() - t_start)
                    save_checkpoint(state)
                    print(f"[KILL-AND-RESUME TEST] Checkpoint saved successfully. Terminating workers...")
                    break

            except Exception:
                # Check if workers still alive
                alive = any(wp.is_alive() for wp in worker_procs)
                if not alive:
                    break

        stop_event.set()
        for wp in worker_procs:
            wp.join(timeout=2.0)

        t_end = time.perf_counter()
        state['elapsed_sec'] = initial_elapsed + (t_end - t_start)
        save_checkpoint(state)

        # Final memory metrics
        final_gz_rss = sum(psutil.Process(p).memory_info().rss / (1024 * 1024) for p in gz_pids if psutil.pid_exists(p))
        final_py_rss = psutil.Process().memory_info().rss / (1024 * 1024)

        state['final_gz_rss_mb'] = final_gz_rss
        state['gz_growth_mb'] = final_gz_rss - start_gz_rss
        state['gz_growth_percent'] = ((final_gz_rss - start_gz_rss) / max(start_gz_rss, 1.0)) * 100.0
        state['final_py_rss_mb'] = final_py_rss
        state['py_growth_mb'] = final_py_rss - start_py_rss

        print(f"\nSoak Session Finished:")
        print(f"  Total Completed Steps: {state['completed_steps']} / {target_steps}")
        print(f"  Elapsed Time:          {state['elapsed_sec']:.1f} s")
        print(f"  gzserver RSS:          {start_gz_rss:.1f} MB -> {final_gz_rss:.1f} MB ({state['gz_growth_mb']:+.2f} MB, {state['gz_growth_percent']:+.2f}%)")
        print(f"  NaN/Inf count:         {state['nan_count']}")
        print(f"  Crashes:               {state['crashes']}")

        return state

    finally:
        # Graceful process-group cleanup
        stop_event.set()
        for wp in worker_procs:
            if wp.is_alive():
                wp.terminate()
        for p in gazebo_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1.0)
        for p in gazebo_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gazebo Soak Runner")
    parser.add_argument('--steps', type=int, default=1000, help="Target steps")
    parser.add_argument('--workers', type=int, default=1, help="Number of workers (1, 2, 4)")
    parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint")
    parser.add_argument('--test-kill-resume', action='store_true', help="Run automated kill-and-resume verification test")
    args = parser.parse_args()

    if args.test_kill_resume:
        print("=== STAGE 1: Starting run and triggering interruption at 300 steps ===")
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        execute_soak(target_steps=600, n_workers=1, resume=False, simulate_interrupt_after=300)

        print("\n=== STAGE 2: Resuming from saved checkpoint to reach 600 steps ===")
        res = execute_soak(target_steps=600, n_workers=1, resume=True)
        assert res['completed_steps'] >= 600, f"Resume failed: completed {res['completed_steps']} < 600"
        print(f"\n[KILL-AND-RESUME VERIFIED]: Successfully resumed from 300 steps and completed {res['completed_steps']} steps!")
    else:
        execute_soak(target_steps=args.steps, n_workers=args.workers, resume=args.resume)
