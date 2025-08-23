import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('puppet')
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'robot_v2.urdf.xacro'])
    #rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'my_robot.rviz'])

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation clock'
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': Command(['xacro ', xacro_file])
        }]
    )

    joint_states_node = Node(
        package='puppet',
        executable='joint_states',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    odometry_node = Node(
        package='puppet',
        executable='odometry',
        name='odometry_publisher',
        output='screen',
    )
    

    lidar_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('sllidar_ros2'),
                    'launch',
                    'sllidar_a2m8_launch.py'
                )
            ),
            launch_arguments={'channel_type':'serial',
                     'serial_port': '/dev/ttyUSB1', 
                     'serial_baudrate': '115200', 
                     'frame_id': 'laser_frame',
                     'inverted': 'false', 
                     'angle_compensate': 'true'}.items())

    depth_to_laserscan = Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan_node',
            remappings=[
                ('depth', '/camera/camera/depth/image_rect_raw'),
                ('depth_camera_info', '/camera/camera/depth/camera_info'),
                ('scan', '/depth_scan')
            ],
            parameters=[{
                'output_frame_id': 'camera_link',
                'scan_height': 125,
                'scan_time': 0.033,
                'range_min': 0.3,
                'range_max': 0.7,
                'angle_min': -0.7436,  # -42.6°
                'angle_max': 0.7436,   # +42.6°
                #'angle_increment': 0.001757, # 0.1007°/pixel
                'roi_center_y': 0.7,
                'roi_height': 0.3
            }]
        )
    scan_merger = Node(
            package='puppet',
            executable='scan_merger',
            name='scan_merger',
            output='screen'
        )

    get_fast_pose = Node(
        package='puppet',
        executable='robot_pose',
        name='robot_pose_in_map',
        output='screen'
    )


    slam_toolbox_params = {
        'use_sim_time': False, # Truyền LaunchConfiguration
        'odom_frame': 'odom', # Truyền LaunchConfiguration
        'map_frame': 'map',      # Truyền LaunchConfiguration
        'base_frame': 'base_link',    # Truyền LaunchConfiguration
        'scan_topic': '/merged_scan', # Truyền LaunchConfiguration (PythonExpression)
        
        'mode': 'mapping', # "mapping" hoặc "localization"
        'map_resolution': 0.05,
        'map_update_interval': 5.,
        'minimum_time_interval': 0.5,
        'minimum_travel_distance': 0.5,
        'max_laser_range': 12.0,
        'min_laser_range': 0.10, # Giá trị ví dụ
        'map_width_meters': 30.0, # Kích thước ban đầu
        'map_height_meters': 30.0,
        'map_start_x': 0.0,
        'map_start_y': 0.0,
        'map_start_at_robot_pose': False,
        'tf_buffer_duration': 10.0,

        'use_scan_matching': True,
        'use_scan_barycenter_for_matching': True,
        'scan_buffer_size': 10,
        'scan_buffer_maximum_scan_distance': 10.0,
        
        'ceres_scan_matcher_options.use_nonmonotonic_steps': False,
        'ceres_scan_matcher_options.max_num_iterations': 20,
        'ceres_scan_matcher_options.num_threads': 2,
        'ceres_scan_matcher_options.occupied_space_weight': 20.0,
        'ceres_scan_matcher_options.translation_weight': 10.0,
        'ceres_scan_matcher_options.rotation_weight': 40.0,
        
        'use_loop_closure': True,
        'loop_search_maximum_scan_distance': 12.0,
        'loop_match_minimum_response_coarse': 0.45, # Giá trị ví dụ
        
        'constraint_scan_weight': 1.0,
        'constraint_loop_weight': 1.0e9,
        
        'solver_plugin': "solver_plugins::CeresSolver",
        'interactive_mode': True
    }

    slam_toolbox_node = Node(
        parameters=[slam_toolbox_params], # Chỉ truyền dictionary này
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen'
    )

    controller_node = Node(
        package='puppet',
        executable='controller',
        name='control_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    auto_mapping = Node(
        package='puppet',
        executable='auto_mapping',
        name='auto_mapping',
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        robot_state_publisher_node,
        joint_states_node,
        odometry_node,
        lidar_node,
        depth_to_laserscan,
        scan_merger,
        slam_toolbox_node,
        controller_node,
        get_fast_pose,
        #auto_mapping
    ])
