import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    pkg_tb3_rl_sim = get_package_share_directory('tb3_rl_sim')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # Model and resource paths
    gazebo_base = '/usr/share/gazebo-11'
    gazebo_models = '/usr/share/gazebo-11/models'
    tb3_models = os.path.join(pkg_tb3_gazebo, 'models')

    curr_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    combined_model_path = f"{tb3_models}:{gazebo_models}:{curr_model_path}".strip(':')

    curr_res_path = os.environ.get('GAZEBO_RESOURCE_PATH', '')
    combined_res_path = f"{gazebo_base}:{curr_res_path}".strip(':')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_tb3_rl_sim, 'worlds', 'arena_4x4.world'),
        description='Path to SDF/World file'
    )

    x_pose_arg = DeclareLaunchArgument('x_pose', default_value='0.0', description='Initial robot x pose')
    y_pose_arg = DeclareLaunchArgument('y_pose', default_value='-1.2', description='Initial robot y pose')
    z_pose_arg = DeclareLaunchArgument('z_pose', default_value='0.015', description='Initial robot z pose')

    # Headless gzserver launch (NO gzclient)
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'false',
            'init': 'true',
            'factory': 'true',
        }.items()
    )

    # Robot State Publisher for TurtleBot3 Burger
    urdf_path = os.path.join(pkg_tb3_gazebo, 'urdf', 'turtlebot3_burger.urdf')
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_desc
        }]
    )

    # Spawn entity using Gazebo ROS spawner
    tb3_sdf_path = os.path.join(pkg_tb3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')
    spawn_turtlebot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'burger',
            '-file', tb3_sdf_path,
            '-x', LaunchConfiguration('x_pose'),
            '-y', LaunchConfiguration('y_pose'),
            '-z', LaunchConfiguration('z_pose')
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''))
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', combined_model_path))
    ld.add_action(SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', combined_res_path))
    ld.add_action(world_arg)
    ld.add_action(x_pose_arg)
    ld.add_action(y_pose_arg)
    ld.add_action(z_pose_arg)
    ld.add_action(gzserver_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
