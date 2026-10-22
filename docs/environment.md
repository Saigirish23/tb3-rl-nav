# Environment Specifications: `sim-real-rl`

This document records the exact system configuration, package versions, and terminal workflow for reproducibility across development, simulation, and deployment.

---

## 1. Host System

| Component | Specification |
| :--- | :--- |
| **OS** | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| **Kernel** | Linux `6.8.0-138-generic` (x86_64) |
| **CPU** | Intel Core i7-14650HX (16 cores, 24 threads) |
| **RAM** | 15 GiB Total (5.8 GiB available) |
| **Disk Space** | 285 GiB total, ~17 GiB free on `/` (mount for `~`) |
| **GPU / CUDA** | NVIDIA GeForce RTX 4060 Laptop (8 GiB VRAM), Driver `580.178.04`, CUDA 13.0 |

---

## 2. ROS 2 & Middleware

| Component | Specification | Status |
| :--- | :--- | :--- |
| **ROS Distribution** | ROS 2 Humble Hawksbill (`/opt/ros/humble`) | Verified |
| **RMW Implementation** | `rmw_cyclonedds_cpp` | Verified |
| **ROS Domain ID** | `30` (default in environment) | Configured |
| **Target Robot Model** | TurtleBot3 Burger (`export TURTLEBOT3_MODEL=burger`) | Pinned |
| **Simulator** | Gazebo Classic 11 (`gazebo`, `ros-humble-gazebo-ros-pkgs`) | Pending APT install |

---

## 3. Python Virtual Environment (`.venv`)

Created with `--system-site-packages` at `~/sim-real-rl/.venv` using system Python `3.10.12`.

### Core RL & Math Stack (Pinned Versions)

| Package | Version | Source / Note |
| :--- | :--- | :--- |
| **Python** | `3.10.12` | System runtime |
| **`torch`** | `2.14.1+cpu` | CPU-only wheel (`https://download.pytorch.org/whl/cpu`) |
| **`torchvision`** | `0.29.1+cpu` | CPU-only wheel |
| **`stable-baselines3`** | `2.7.1` | PPO implementation |
| **`gymnasium`** | `1.2.3` | Standard environment API |
| **`numpy`** | `1.26.4` | Pinned `<2.0` for ROS 2 Humble C-extension compatibility |
| **`rclpy`** | `3.3.22` | Inherited from `/opt/ros/humble/local/lib/python3.10/dist-packages` |
| **`pytest`** | `8.4.2` | Unit testing |
| **`pyyaml`** | `5.4.1` | Configuration management |

---

## 4. Terminal Activation Workflow

In every new terminal session for development or training, run:

```bash
# 1. Source ROS 2 Humble base
source /opt/ros/humble/setup.bash

# 2. Set TurtleBot3 model and domain ID
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30

# 3. Source project workspace (after building ros2_ws)
source ~/sim-real-rl/ros2_ws/install/setup.bash

# 4. Activate the RL virtual environment
source ~/sim-real-rl/.venv/bin/activate
```
