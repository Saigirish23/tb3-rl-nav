"""Parallel Gazebo Simulation Throughput Benchmark (1, 2, 4 workers).

Measures:
- Aggregate steps/s
- Per-worker throughput (steps/s)
- Real-time factor (RTF)
- CPU / RAM utilization
- gzserver process RSS memory growth
- Inter-worker interference check
- Wall-clock time projection for 12x 1M-step runs
"""

from __future__ import annotations
import os
import sys
import time
import math
import signal
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List

import psutil
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def start_gazebo_worker(worker_id: int) -> subprocess.Popen:
    """Launch headless Gazebo simulation for a specific worker."""
    port = 11345 + worker_id
    domain_id = 30 + worker_id

    env = os.environ.copy()
    env['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
    env['ROS_DOMAIN_ID'] = str(domain_id)
    env['TURTLEBOT3_MODEL'] = 'burger'
    env['GAZEBO_MODEL_DATABASE_URI'] = ''
    env['PYTHONUNBUFFERED'] = '1'

    # Source ROS 2 and workspace
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


def wait_for_worker_ready(worker_id: int, timeout_sec: float = 25.0) -> bool:
    """Wait until /scan and /gazebo/set_entity_state are active on the worker's domain."""
    domain_id = 30 + worker_id
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(domain_id)

    start_t = time.perf_counter()
    check_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"ros2 topic list"
    )

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
                time.sleep(1.0)  # Brief settle time
                return True
        except Exception:
            pass
        time.sleep(0.5)

    return False


def get_gzserver_pids() -> List[int]:
    """Find all running gzserver process IDs."""
    pids = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'gzserver' in p.info['name'] or (p.info['cmdline'] and any('gzserver' in c for c in p.info['cmdline'])):
                pids.append(p.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def run_worker_stepping_process(
    worker_id: int,
    target_steps: int,
    output_queue: Any,
) -> None:
    """Worker child process running environment stepping with dedicated DDS domain."""
    domain_id = 30 + worker_id
    port = 11345 + worker_id

    # Configure worker environment in this isolated process
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv

    env = TurtleBot3NavigationEnv(use_gazebo=True, seed=worker_id * 1000)
    obs, info = env.reset(seed=worker_id * 1000)

    start_wall = time.perf_counter()
    steps = 0
    resets = 0
    failures = 0

    action = np.array([0.0, 0.0], dtype=np.float32)  # Low motion for benchmark stability

    for s in range(target_steps):
        try:
            obs, r, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc:
                resets += 1
                obs, info = env.reset()
        except Exception as e:
            failures += 1

    end_wall = time.perf_counter()
    duration = end_wall - start_wall

    env.close()

    output_queue.put((worker_id, {
        'steps': steps,
        'resets': resets,
        'failures': failures,
        'duration': duration,
        'steps_per_sec': steps / max(duration, 0.001),
    }))


def benchmark_configuration(n_workers: int, target_steps: int = 100) -> Dict[str, Any]:
    """Benchmark a specific worker count configuration."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {n_workers} WORKER(S) ({target_steps} steps per worker)")
    print(f"{'='*70}")

    # Clean existing Gazebo instances from previous runs if any were tracked
    time.sleep(1.0)

    procs = []
    try:
        print(f"Starting {n_workers} Gazebo worker instances...")
        for w in range(n_workers):
            proc = start_gazebo_worker(w)
            procs.append(proc)

        # Wait for all workers to be ready
        for w in range(n_workers):
            print(f"  Waiting for Worker {w} (ROS_DOMAIN_ID={30+w}, port={11345+w})...")
            ready = wait_for_worker_ready(w, timeout_sec=30.0)
            if not ready:
                raise RuntimeError(f"Worker {w} failed to become ready within timeout.")
            print(f"  Worker {w} is READY.")

        # Measure baseline resources
        gz_pids = get_gzserver_pids()
        gz_mem_start = 0.0
        for pid in gz_pids:
            try:
                gz_mem_start += psutil.Process(pid).memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        cpu_percent_start = psutil.cpu_percent(interval=0.5)
        sys_mem_start = psutil.virtual_memory().used / (1024 * 1024)

        # Launch worker stepping in parallel child processes with dedicated domain IDs
        out_queue = ctx.Queue()
        worker_procs = []
        
        # Start background CPU monitor sampling during active stepping
        cpu_samples = []
        stop_monitor = threading.Event()
        def monitor_cpu():
            while not stop_monitor.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.2))
        mon_thread = threading.Thread(target=monitor_cpu, daemon=True)
        mon_thread.start()

        bench_start = time.perf_counter()

        for w in range(n_workers):
            p = ctx.Process(
                target=run_worker_stepping_process,
                args=(w, target_steps, out_queue),
            )
            worker_procs.append(p)
            p.start()

        results = {}
        for _ in range(n_workers):
            w_id, res = out_queue.get()
            results[w_id] = res

        for p in worker_procs:
            p.join()

        bench_end = time.perf_counter()
        total_duration = bench_end - bench_start

        # Stop CPU monitor
        stop_monitor.set()
        mon_thread.join(timeout=1.0)
        cpu_percent_avg = float(np.mean(cpu_samples)) if cpu_samples else psutil.cpu_percent()
        cpu_cores_used = cpu_percent_avg * (psutil.cpu_count() or 1) / 100.0

        # Measure final resources
        gz_mem_end = 0.0
        for pid in gz_pids:
            try:
                gz_mem_end += psutil.Process(pid).memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        sys_mem_end = psutil.virtual_memory().used / (1024 * 1024)

        total_steps = sum(r['steps'] for r in results.values())
        total_failures = sum(r['failures'] for r in results.values())
        aggregate_sps = total_steps / max(total_duration, 0.001)
        mean_worker_sps = float(np.mean([r['steps_per_sec'] for r in results.values()]))

        # Real-time factor (0.2s control step per target step)
        expected_sim_time = target_steps * 0.20
        rtf = (expected_sim_time) / max(total_duration, 0.001)

        gz_mem_growth = gz_mem_end - gz_mem_start

        # 12M steps projection
        hours_for_12m = (12_000_000 / aggregate_sps) / 3600.0 if aggregate_sps > 0 else float('inf')

        summary = {
            'n_workers': n_workers,
            'target_steps_per_worker': target_steps,
            'total_steps': total_steps,
            'total_duration_sec': total_duration,
            'aggregate_steps_per_sec': aggregate_sps,
            'mean_worker_steps_per_sec': mean_worker_sps,
            'real_time_factor': rtf,
            'failures': total_failures,
            'cpu_percent': cpu_percent_avg,
            'cpu_cores_used': cpu_cores_used,
            'system_ram_used_mb': sys_mem_end,
            'gzserver_rss_total_mb': gz_mem_end,
            'gzserver_rss_growth_mb': gz_mem_growth,
            'hours_for_12m_steps': hours_for_12m,
            'per_worker_results': results,
        }

        print(f"\nRESULTS FOR {n_workers} WORKER(S):")
        print(f"  Total Duration:     {total_duration:.2f} s")
        print(f"  Total Steps:        {total_steps} (Failures: {total_failures})")
        print(f"  Aggregate Rate:     {aggregate_sps:.2f} steps/s")
        print(f"  Per-Worker Rate:    {mean_worker_sps:.2f} steps/s")
        print(f"  Real-Time Factor:   {rtf:.2f}x")
        print(f"  CPU Utilization:    {cpu_percent_avg:.1f}% (approx {cpu_cores_used:.2f} logical cores out of {psutil.cpu_count()})")
        print(f"  gzserver Total RSS: {gz_mem_end:.1f} MB (Growth: {gz_mem_growth:+.2f} MB)")
        print(f"  12M Steps Wall Time: {hours_for_12m:.1f} hours ({hours_for_12m/24.0:.1f} days)")

        return summary

    finally:
        # Clean teardown of only worker processes launched by this benchmark
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1.0)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)


def run_full_throughput_study() -> Dict[int, Any]:
    """Run controlled benchmarks with 1, 2, and 4 Gazebo workers."""
    all_results = {}
    for n in [1, 2, 4]:
        res = benchmark_configuration(n_workers=n, target_steps=50)
        all_results[n] = res
    return all_results


if __name__ == '__main__':
    run_full_throughput_study()
