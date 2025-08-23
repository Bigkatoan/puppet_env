from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan_node',
            remappings=[
                ('depth', '/camera/camera/depth/image_rect_raw'),    # Input depth topic
                ('depth_camera_info', '/camera/camera/depth/camera_info'),  # Camera info
                ('scan', '/depth_scan')                 # Output scan topic
            ],
            parameters=[{
                'output_frame_id': 'camera_depth_frame',   # TF frame của camera
                'scan_height': 125,                      # Số hàng pixel sử dụng
                'scan_time': 0.033,                     # Thời gian giữa các scan (s)
                'range_min': 0.3,                       # Khoảng cách tối thiểu (m)
                'range_max': 0.7,                       # Khoảng cách tối đa (m)
                'angle_min': -0.785,                    # Góc quét tối thiểu (-45°)
                'angle_max': 0.785,                     # Góc quét tối đa (+45°)
                'angle_increment': 0.0087,              # Độ phân giải góc (0.5°)
            }]
        )
    ])