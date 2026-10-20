# Setup Log: `sim-real-rl` Project

All environment setup, dependency installations, and workspace builds are recorded here chronologically.

---

## Stage 1: System Inspection (Read-Only)
* **Date**: 2026-10-21
* **Objective**: Audit host environment, ROS distribution, hardware resources, existing workspaces, and Gazebo Classic availability.

### Commands Run & Results
1. `lsb_release -a; uname -r; nproc; lscpu | head -20; free -h; df -h ~ /; nvidia-smi`
   * **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish), Kernel `6.8.0-138-generic`
   * **CPU**: Intel Core i7-14650HX (16 physical cores, 24 vCPUs, up to 5.2 GHz)
   * **RAM**: 15 GiB total (2.0 GiB free, 4.5 GiB available for processes), 15 GiB swap
   * **GPU**: NVIDIA GeForce RTX 4060 Laptop (8 GiB VRAM), Driver `580.178.04`, CUDA 13.0
   * **Disk Space**:
     * Filesystem `/dev/nvme0n1p7` mounted on `/` and `~`
     * 260 GiB used / 285 GiB total (97% used)
     * **11 GiB available** -> **Below user threshold of 15 GiB**. Action: Minimize installation sizes, use CPU-only PyTorch wheel.
2. `printenv ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH TURTLEBOT3_MODEL ROS_DOMAIN_ID RMW_IMPLEMENTATION`
   * `ROS_DISTRO`: `humble`
   * `ROS_DOMAIN_ID`: `30`
   * `RMW_IMPLEMENTATION`: `rmw_cyclonedds_cpp`
   * `TURTLEBOT3_MODEL`: unset
   * `AMENT_PREFIX_PATH`: Included overlays from `~/turtlebot3_ws/install` and `~/ros2_ws/install` alongside `/opt/ros/humble`.
3. `which ros2 colcon rosdep vcs git pdftotext; python3 --version; python3 -m venv --help | head -1`
   * All tools present in `/usr/bin` or `/opt/ros/humble/bin`.
   * Python: `3.10.12`, `venv` module operational.
4. `ros2 doctor --report`
   * Verified ROS 2 Humble distribution, active DDS configuration.
5. `gazebo --version`
   * Exited with code 127 (`command not found`). Gazebo Classic binary is not installed on system PATH.
6. `dpkg -l | grep -E '^ii +ros-humble-(turtlebot3|nav2|navigation2|slam-toolbox|gazebo|dynamixel|xacro)'`
   * `ros-humble-navigation2` & `ros-humble-nav2-*`: Installed (`1.1.20-1jammy`).
   * `ros-humble-slam-toolbox`: Installed (`2.6.10-1jammy`).
   * `ros-humble-xacro`: Installed (`2.1.1-1jammy`).
   * `ros-humble-gazebo-msgs`: Installed (`3.9.0-1jammy`).
   * `ros-humble-gazebo-ros` / `gazebo_ros_pkgs` / `gazebo11`: **Missing**.
   * `ros-humble-turtlebot3*`: **Not installed via apt**.
7. `for p in turtlebot3_gazebo turtlebot3_description turtlebot3_msgs gazebo_ros nav2_bringup slam_toolbox; do echo "== $p"; ros2 pkg prefix $p; done`
   * `turtlebot3_gazebo`, `turtlebot3_description`, `turtlebot3_msgs` resolved to `~/turtlebot3_ws/install/`.
   * `gazebo_ros`: `Package not found`.
   * `nav2_bringup`, `slam_toolbox`: `/opt/ros/humble`.
8. `find ~ -maxdepth 4 -type d -name install -not -path '*/.cache/*' 2>/dev/null`
   * Workspaces found: `~/turtlebot3_ws/install`, `~/ros2_ws/install`, `~/rclcpp_dev_ws/install`, `~/tb3_swarm_ws/install`.
9. `grep -n -i 'ros\|source' ~/.bashrc`
   * Auto-sources `turtlebot3_ws`, `/usr/share/gazebo/setup.sh`, `ros2_ws`, and `Warehouse_Simulation`.

---

## Stage 2: Project Creation
* **Date**: 2026-10-21
* **Objective**: Initialize project directory structure, git repository, `.gitignore`, documentation templates, and copy the specification PDF.

### Actions Executed
1. Directory structure created:
   * `ros2_ws/src/`
   * `rl/{env,train,evaluate,analysis}/`
   * `baselines/`
   * `configs/`
   * `scenarios/{train,validation,test,stress}/`
   * `deploy/`
   * `tests/`
   * `runs/`
   * `docs/`
2. `.gitkeep` placeholders added to empty directories.
3. Spec PDF copied from `/home/saigirish/Downloads/Sim2Real_LiDAR_Navigation_Pipeline.pdf` to `docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf`.
4. `.gitignore` created to exclude `.venv/`, ROS build/install/log, runs/, checkpoints, model weights, and upstream source clones.
5. `README.md`, `docs/setup_log.md`, and `docs/decisions.md` created.
6. Git repository initialized with `git init`.
