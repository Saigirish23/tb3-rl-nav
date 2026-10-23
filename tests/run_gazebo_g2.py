"""Runner script to launch headless Gazebo and execute Gazebo-specific G2 tests."""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def run():
    time.sleep(1.0)

    env = os.environ.copy()
    env['GAZEBO_MASTER_URI'] = "http://localhost:11345"
    env['ROS_DOMAIN_ID'] = "30"
    env['TURTLEBOT3_MODEL'] = 'burger'
    env['GAZEBO_MODEL_DATABASE_URI'] = ''
    env['PYTHONUNBUFFERED'] = '1'

    launch_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {REPO_ROOT}/ros2_ws/install/setup.bash && "
        f"ros2 launch tb3_rl_sim headless_sim.launch.py"
    )

    print("Starting Gazebo Classic headless simulator...")
    proc = subprocess.Popen(
        launch_cmd,
        shell=True,
        executable='/bin/bash',
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        # Wait for Gazebo topics to be ready
        print("Waiting for Gazebo /scan and /clock...")
        ready = False
        start_wait = time.perf_counter()
        check_cmd = f"source /opt/ros/humble/setup.bash && ros2 topic list"

        while (time.perf_counter() - start_wait) < 30.0:
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
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            print("ERROR: Gazebo failed to start within 30 seconds.")
            return 1

        print("Gazebo is READY. Running tests/test_g2_gazebo.py...")
        time.sleep(2.0)  # Settle time

        test_cmd = (
            f"{sys.executable} -m pytest "
            f"tests/test_g2_gazebo.py -v -s --tb=short"
        )
        test_res = subprocess.run(
            test_cmd,
            shell=True,
            executable='/bin/bash',
            env=env,
        )
        return test_res.returncode

    finally:
        print("\nShutting down Gazebo...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(1.0)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        print("Gazebo shutdown complete.")

if __name__ == '__main__':
    sys.exit(run())
