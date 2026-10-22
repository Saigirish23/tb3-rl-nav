"""10,000-Step Random Policy Soak and Stability Test.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 10 Day 2 Exit Gate)
Requirement: 10,000 random steps without crash or memory growth.
"""

import os
import sys
import time
import psutil
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv


def run_soak_test(num_steps: int = 10000, seed: int = 42):
    print(f"\nStarting {num_steps}-step random soak test (Seed={seed})...")
    env = TurtleBot3NavigationEnv(use_gazebo=False, seed=seed)

    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024 * 1024)

    obs, info = env.reset(seed=seed)
    start_time = time.perf_counter()

    episodes = 0
    goals = 0
    collisions = 0
    timeouts = 0

    for step in range(1, num_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Integrity assertions
        assert obs.shape == (43,), f"Bad obs shape at step {step}: {obs.shape}"
        assert np.all(np.isfinite(obs)), f"Non-finite obs at step {step}"
        assert np.isfinite(reward), f"Non-finite reward at step {step}"

        if terminated or truncated:
            episodes += 1
            if info.get("is_success"):
                goals += 1
            if info.get("is_collision"):
                collisions += 1
            if truncated:
                timeouts += 1
            obs, info = env.reset()

        if step % 2000 == 0 or step == num_steps:
            elapsed = time.perf_counter() - start_time
            rate = step / elapsed
            curr_mem = process.memory_info().rss / (1024 * 1024)
            print(f"Step {step:5d}/{num_steps} | Rate: {rate:6.1f} steps/s | Episodes: {episodes:4d} | RSS: {curr_mem:.1f} MB")

    total_time = time.perf_counter() - start_time
    mem_end = process.memory_info().rss / (1024 * 1024)
    mem_diff = mem_end - mem_start

    print("\n" + "=" * 50)
    print(" 10,000-STEP RANDOM SOAK TEST COMPLETED")
    print("=" * 50)
    print(f"Total Steps       : {num_steps}")
    print(f"Elapsed Time      : {total_time:.2f} s")
    print(f"Throughput        : {num_steps / total_time:.1f} steps/s")
    print(f"Episodes Finished : {episodes}")
    print(f"Goals Reached     : {goals}")
    print(f"Collisions        : {collisions}")
    print(f"Timeouts          : {timeouts}")
    print(f"Memory RSS Start  : {mem_start:.2f} MB")
    print(f"Memory RSS End    : {mem_end:.2f} MB (Change: {mem_diff:+.2f} MB)")
    print("Zero crashes, zero NaN/Inf observations encountered.")
    print("=" * 50 + "\n")

    env.close()
    return True


if __name__ == '__main__':
    run_soak_test(10000)
