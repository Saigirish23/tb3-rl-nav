#!/usr/bin/env python3
"""Unified Policy Evaluation Launch File for Sim-to-Real TB3 RL Navigation.

Launches Gazebo Classic 11 with the 5x5m warehouse arena and evaluates trained
Gen1 policies (P1, P2, P3, P4) across the 9 Phase 2 benchmark conditions.

Usage:
  ros2 launch tb3_rl_sim evaluate_policies.launch.py
  ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=pilot policy:=P1
  ros2 launch tb3_rl_sim evaluate_policies.launch.py mode:=benchmark policy:=all
  ros2 launch tb3_rl_sim evaluate_policies.launch.py use_gazebo:=false
"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    ExecuteProcess,
    RegisterEventHandler,
    EmitEvent,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    pkg_tb3_rl_sim = get_package_share_directory('tb3_rl_sim')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # Environment setup
    gazebo_base = '/usr/share/gazebo-11'
    gazebo_models = '/usr/share/gazebo-11/models'
    tb3_models = os.path.join(pkg_tb3_gazebo, 'models')

    curr_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    combined_model_path = f"{tb3_models}:{gazebo_models}:{curr_model_path}".strip(':')

    curr_res_path = os.environ.get('GAZEBO_RESOURCE_PATH', '')
    combined_res_path = f"{gazebo_base}:{curr_res_path}".strip(':')

    # Resolve python script location
    runner_script = os.path.join(pkg_tb3_rl_sim, 'scripts', 'evaluate_benchmark.py')
    if not os.path.exists(runner_script):
        # Fallback to repository root
        repo_root = Path(pkg_tb3_rl_sim).resolve().parent.parent.parent
        runner_script = str(repo_root / 'scripts' / 'evaluate_benchmark.py')

    # Launch Arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='pilot',
        choices=['pilot', 'benchmark'],
        description='Evaluation mode: pilot (36 episodes) or benchmark (720 episodes)'
    )
    policy_arg = DeclareLaunchArgument(
        'policy',
        default_value='all',
        description='Policy selection: all, P1, P2, P3, or P4 (or comma-separated e.g. P1,P4)'
    )
    condition_arg = DeclareLaunchArgument(
        'condition',
        default_value='all',
        description='Condition selection: all or specific condition (e.g. case1_corridor, case5_dynamic)'
    )
    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='2000000',
        description='Base seed for evaluation'
    )
    out_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='runs/benchmark_results',
        description='Output directory for CSV and JSON summary'
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run Gazebo headless without gzclient GUI (true/false)'
    )
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Use live Gazebo simulation (true/false)'
    )
    use_final_arg = DeclareLaunchArgument(
        'use_final',
        default_value='false',
        description='Evaluate final model checkpoints instead of best checkpoints (true/false)'
    )

    # Condition objects
    cond_gazebo_headless = IfCondition(
        PythonExpression([
            "'", LaunchConfiguration('use_gazebo'), "'.lower() == 'true' and '",
            LaunchConfiguration('headless'), "'.lower() == 'true'"
        ])
    )
    cond_gazebo_gui = IfCondition(
        PythonExpression([
            "'", LaunchConfiguration('use_gazebo'), "'.lower() == 'true' and '",
            LaunchConfiguration('headless'), "'.lower() == 'false'"
        ])
    )

    # Include headless Gazebo launch
    headless_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_rl_sim, 'launch', 'headless_sim.launch.py')
        ),
        condition=cond_gazebo_headless,
    )

    # Include GUI Gazebo launch
    gui_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_rl_sim, 'launch', 'sim.launch.py')
        ),
        condition=cond_gazebo_gui,
    )

    # Benchmark runner process
    runner_cmd = ExecuteProcess(
        cmd=[
            sys.executable,
            runner_script,
            '--mode', LaunchConfiguration('mode'),
            '--policy', LaunchConfiguration('policy'),
            '--condition', LaunchConfiguration('condition'),
            '--base-seed', LaunchConfiguration('seed'),
            '--out-dir', LaunchConfiguration('output_dir'),
            '--use-gazebo', LaunchConfiguration('use_gazebo'),
        ],
        output='screen'
    )

    # Shutdown Gazebo when evaluation runner process exits
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=runner_cmd,
            on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    ld = LaunchDescription()

    # Environment variables
    ld.add_action(SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'))
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''))
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path))
    ld.add_action(SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', combined_res_path))
    ld.add_action(SetEnvironmentVariable('PYTHONUNBUFFERED', '1'))

    # Declare arguments
    ld.add_action(mode_arg)
    ld.add_action(policy_arg)
    ld.add_action(condition_arg)
    ld.add_action(seed_arg)
    ld.add_action(out_dir_arg)
    ld.add_action(headless_arg)
    ld.add_action(use_gazebo_arg)
    ld.add_action(use_final_arg)

    # Gazebo instances (conditional)
    ld.add_action(headless_gazebo_cmd)
    ld.add_action(gui_gazebo_cmd)

    # Runner and shutdown handler
    ld.add_action(runner_cmd)
    ld.add_action(shutdown_handler)

    return ld

