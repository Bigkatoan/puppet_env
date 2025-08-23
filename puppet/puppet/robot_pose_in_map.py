#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class FastPosePublisher(Node):
    def __init__(self):
        super().__init__('fast_pose_publisher')
        
        # Khai báo tham số
        self.declare_parameter('source_frame', 'base_link')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('publish_frequency', 30.0)  # Hz
        
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, '/fast_pose', 10)
        
        # Timer
        self.timer = self.create_timer(1.0 / self.publish_frequency, self.timer_callback)
        
        self.get_logger().info(
            f"Publishing pose of {self.source_frame} in {self.target_frame} at {self.publish_frequency} Hz"
        )

    def timer_callback(self):
        try:
            # SỬA LỖI: Đảo thứ tự frame tại đây
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,   # Frame đích (hệ quy chiếu)
                self.source_frame,   # Frame nguồn (vật cần xác định vị trí)
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            
            # Tạo PoseStamped từ transform
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.target_frame  # Frame đích
            
            # Vị trí và hướng của source_frame TRONG target_frame
            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z
            pose_msg.pose.orientation = transform.transform.rotation
            
            # Publish
            self.pose_pub.publish(pose_msg)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {str(e)}", throttle_duration_sec=1)

def main(args=None):
    rclpy.init(args=args)
    node = FastPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()