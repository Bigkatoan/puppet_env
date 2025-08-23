#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np

class SimpleScanMerger(Node):
    def __init__(self):
        super().__init__('simple_scan_merger')
        
        # Khai báo các tham số
        self.declare_parameters(
            namespace='',
            parameters=[
                ('output_frame', 'base_link'),
                ('merged_scan_topic', '/merged_scan'),
                ('lidar_topic', '/scan'),
                ('camera_topic', '/depth_scan'),
                ('camera_fov', 1.5),  # Góc quét camera (radian), ~85.2 độ
                ('camera_center_angle', 0.0),  # Góc trung tâm camera
            ]
        )
        
        # Lấy giá trị tham số
        self.output_frame = self.get_parameter('output_frame').value
        self.merged_scan_topic = self.get_parameter('merged_scan_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_fov = self.get_parameter('camera_fov').value
        self.camera_center_angle = self.get_parameter('camera_center_angle').value
        
        # Tính toán góc quét camera
        self.camera_min_angle = self.camera_center_angle - self.camera_fov/2
        self.camera_max_angle = self.camera_center_angle + self.camera_fov/2
        
        # Tạo publisher cho scan hợp nhất
        self.merged_pub = self.create_publisher(LaserScan, self.merged_scan_topic, 10)
        
        # Tạo subscribers với message_filters
        lidar_sub = Subscriber(self, LaserScan, self.lidar_topic)
        camera_sub = Subscriber(self, LaserScan, self.camera_topic)
        
        # Thiết lập bộ đồng bộ thời gian
        self.synchronizer = ApproximateTimeSynchronizer(
            [lidar_sub, camera_sub],
            queue_size=10,
            slop=0.1
        )
        self.synchronizer.registerCallback(self.sync_callback)
        
        self.get_logger().info("Simple Scan Merger Node started")
        self.get_logger().info(f"Lidar topic: {self.lidar_topic}")
        self.get_logger().info(f"Camera topic: {self.camera_topic}")
        self.get_logger().info(f"Camera FOV: {np.degrees(self.camera_fov):.1f}° centered at {np.degrees(self.camera_center_angle):.1f}°")
        self.get_logger().info("Camera data has priority in its FOV")
        
    def sync_callback(self, lidar_msg, camera_msg):
        """Xử lý khi nhận được message đồng bộ"""
        # Tạo message scan hợp nhất
        merged_scan = LaserScan()
        
        # Sao chép metadata từ lidar scan
        merged_scan.header = lidar_msg.header
        merged_scan.header.frame_id = self.output_frame
        merged_scan.angle_min = lidar_msg.angle_min
        merged_scan.angle_max = lidar_msg.angle_max
        merged_scan.angle_increment = lidar_msg.angle_increment
        merged_scan.time_increment = lidar_msg.time_increment
        merged_scan.scan_time = lidar_msg.scan_time
        merged_scan.range_min = lidar_msg.range_min
        merged_scan.range_max = lidar_msg.range_max
        
        # Sử dụng dữ liệu từ lidar làm cơ sở
        merged_scan.ranges = list(lidar_msg.ranges)
        
        # Xác định số điểm trong scan
        num_points = len(merged_scan.ranges)
        
        # Hàm chuyển đổi góc thành index
        def angle_to_index(angle):
            # Chuẩn hóa góc về [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            # Tính index
            idx = int(round((normalized_angle - merged_scan.angle_min) / merged_scan.angle_increment))
            # Đảm bảo index nằm trong giới hạn
            if idx < 0: return 0
            if idx >= num_points: return num_points - 1
            return idx
        
        # Hợp nhất dữ liệu từ camera (ưu tiên trong FOV của camera)
        for i, cam_range in enumerate(camera_msg.ranges):
            # Tính góc tương ứng
            angle = camera_msg.angle_min + i * camera_msg.angle_increment
            
            # Kiểm tra xem góc có nằm trong FOV của camera không
            if not (self.camera_min_angle <= angle <= self.camera_max_angle):
                continue
                
            # Tìm index tương ứng trong merged scan
            idx = angle_to_index(angle)
            
            # Chỉ cập nhật nếu dữ liệu camera hợp lệ
            if cam_range >= camera_msg.range_min and cam_range <= camera_msg.range_max:
                if 0 <= idx < num_points:
                    merged_scan.ranges[idx] = cam_range
        
        # Publish scan hợp nhất
        self.merged_pub.publish(merged_scan)
        self.get_logger().info("Published merged scan", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()