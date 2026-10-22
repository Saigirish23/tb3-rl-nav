# Session Snapshot: `sim-real-rl` Project Setup

* **Date**: 2026-10-22
* **Current State**: Stage 3 (Python venv ready; APT download in progress)

---

## 1. Project & Workspace Status

* **Repository Root**: `/home/saigirish/sim-real-rl`
* **Git Commit**: `bfc16ad` (*docs: record Stage 3 Python venv and environment specification*)
* **Specification**: Frozen per [`docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`](file:///home/saigirish/sim-real-rl/docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf).
* **Tracked Decisions**: Recorded in [`docs/decisions.md`](file:///home/saigirish/sim-real-rl/docs/decisions.md).
* **Setup History**: Chronologically logged in [`docs/setup_log.md`](file:///home/saigirish/sim-real-rl/docs/setup_log.md).
* **Environment Spec**: Pinned in [`docs/environment.md`](file:///home/saigirish/sim-real-rl/docs/environment.md).

---

## 2. Verified on Your Machine

1. **Host & Resources**:
   * **OS**: Ubuntu 22.04.5 LTS, Kernel `6.8.0-138-generic`.
   * **CPU**: Intel Core i7-14650HX (16 cores, 24 threads).
   * **RAM**: 15 GiB (5.8 GiB available).
   * **Disk Space**: ~16 GiB available on `/dev/nvme0n1p7` (exceeds the 15 GiB safety threshold).
2. **ROS 2 & Middleware**:
   * ROS 2 Humble Hawksbill with `rmw_cyclonedds_cpp`, `ROS_DOMAIN_ID=30`.
   * Nav2 (`ros-humble-navigation2` 1.1.20) and SLAM Toolbox (`ros-humble-slam-toolbox` 2.6.10) verified in `/opt/ros/humble`.
3. **Python Virtual Environment (`.venv`)**:
   * Created at `~/sim-real-rl/.venv` with `--system-site-packages`.
   * `import rclpy` verified from `/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy`.
   * CPU-only PyTorch `2.14.1+cpu` and `torchvision` `0.29.1+cpu` installed.
   * `stable-baselines3` (2.7.1), `gymnasium` (1.2.3), `numpy` (1.26.4, `<2.0`), `pytest` (8.4.2), and `pyyaml` (5.4.1) verified.

---

## 3. Pending Action / Current Stage (Stage 3 -> Stage 4)

* During execution of:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    gazebo \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3-description \
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3-gazebo
  ```
  The download of package 4 (`ros-humble-gazebo-plugins`, 1.6 MB) was interrupted via `SIGINT` (`Ctrl+C`) due to a slow network mirror (~10 kB/s).
* **Only 17.7 MB total** archives need to be fetched, using just **48.7 MB** of additional disk space.
* Once network bandwidth is stable, re-running the command resumes immediately.

---

## 4. Next Step When Resuming

1. Complete the APT command:
   ```bash
   sudo apt-get install -y gazebo ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3-description ros-humble-turtlebot3-msgs ros-humble-turtlebot3-gazebo
   ```
2. Verify:
   * `gazebo --version` (Gazebo 11.10.x)
   * `ros2 pkg prefix gazebo_ros`
3. Advance to **Stage 4**: Create `tb3_rl_sim`, write headless `gzserver` launch with `gazebo_ros_state`, verify topics (`/scan`, `/odom`, `/cmd_vel`, `/clock`), QoS, and run the headless benchmark.
