import os
import time

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    socspioneer_p = get_package_share_directory("socspioneer")
    pf_p = get_package_share_directory("pf_localisation")

    world_path = DeclareLaunchArgument(
        name="world_path",
        default_value= os.path.join(pf_p, "data", "sim_data", "meeting.world")
    )

    map_path = DeclareLaunchArgument(
        name="map_path",
        default_value= os.path.join(pf_p, "data", "sim_data", "meeting.yaml")    
    )

    lifecycle_manager = Node(
    package="nav2_lifecycle_manager",
    executable="lifecycle_manager",
    parameters=[{"node_names": ["amcl","map_server"], "autostart": True, "use_sim_time": True}],
    arguments= [
    '--ros-args',
    '--log-level',
    'ERROR', 
    ])

    map_server = Node(
    package="nav2_map_server",
    executable="map_server",
    parameters=[{"yaml_filename": LaunchConfiguration('map_path'), "use_sim_time": True}],
    arguments= [
    '--ros-args',
    '--log-level',
    'ERROR',
    ])

    stage_ros2 = Node(
        package="stage_ros2",
        executable="stage_ros2",
        parameters=[{"world_file": os.path.join(pf_p, "data", "sim_data", "meeting.world"), "use_sim_time": True}],
        arguments= [
        '--ros-args',
        '--log-level',
        'ERROR',
        ])
    
    socspioneer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(socspioneer_p,'launch','stage.launch.py')),
        launch_arguments=[('map_path', LaunchConfiguration('map_path')),
                          ('world_path', LaunchConfiguration('world_path'))])
    
    socspioneer_key = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(socspioneer_p,'launch','keyboard_teleop.launch.py'))
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        parameters=[{"use_sim_time": True,}],
        arguments=["-d", [os.path.join(socspioneer_p, "config", "map_view.rviz")], 
                   '--ros-args',
        '--log-level',
        'ERROR',
        ]
    )

    pf_localisation = Node(
        package="pf_localisation",
        executable="node.py",
        parameters=[{"use_sim_time": True}]
        # arguments= [
        # '--ros-args',
        # '--log-level',
        # 'ERROR',
        # ]
        )


    ld = LaunchDescription()
    ld.add_action(world_path)
    ld.add_action(map_path)
    ld.add_action(lifecycle_manager)

#SIM_DATA
    ld.add_action(socspioneer)
    ld.add_action(socspioneer_key)

#REAL_DATA
    # ld.add_action(foxglove_bridge)
    # ld.add_action(stage_ros2)
    # ld.add_action(map_server)
    # ld.add_action(amcl)
    # ld.add_action(rviz2)

    # time.sleep(3)
    ld.add_action(pf_localisation)

    return ld
