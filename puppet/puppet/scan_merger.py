#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np

class ScanMerger(Node):
    def __init__(self):
        super().__init__('scan_merger')
        
        # Khai báo các tham số
        self.declare_parameters(
            namespace='',
            parameters=[
                ('output_frame', 'base_link'),
                ('merged_scan_topic', '/merged_scan'),
                ('scan1_topic', '/scan'),          # Lidar scan
                ('scan2_topic', '/depth_scan'),     # Camera depth scan
                ('angle_min', -3.14159),            # -180 độ
                ('angle_max', 3.14159),             # 180 độ
                ('angle_increment', 0.0087),        # 0.5 độ
                ('range_min', 0.15),
                ('range_max', 12.0),
                ('time_sync_tolerance', 0.1),       # 0.1 giây
                ('queue_size', 10),
                ('camera_priority', True),          # Ưu tiên dữ liệu camera khi trùng
                ('camera_priority_range', 0.8),     # Phạm vi ưu tiên camera (m)
                ('filter_ghosts', True),
                ('ghost_threshold', 0.5),
                ('camera_fov', 1.57),              # Góc quét camera (radian - mặc định 90 độ)
                ('camera_center_angle', 0.0)       # Góc trung tâm camera (radian)
            ]
        )
        
        # Lấy giá trị tham số
        params = self.get_parameters([
            'output_frame', 'merged_scan_topic', 'scan1_topic', 'scan2_topic',
            'angle_min', 'angle_max', 'angle_increment', 'range_min', 'range_max',
            'time_sync_tolerance', 'queue_size', 'camera_priority',
            'camera_priority_range', 'filter_ghosts', 'ghost_threshold',
            'camera_fov', 'camera_center_angle'
        ])
        
        self.params = {p.name: p.value for p in params}
        
        # Tạo publisher cho scan hợp nhất
        self.merged_pub = self.create_publisher(
            LaserScan,
            self.params['merged_scan_topic'],
            10
        )
        
        # Tạo subscribers với message_filters
        scan1_sub = Subscriber(self, LaserScan, self.params['scan1_topic'])
        scan2_sub = Subscriber(self, LaserScan, self.params['scan2_topic'])
        
        # Thiết lập bộ đồng bộ thời gian gần đúng
        self.synchronizer = ApproximateTimeSynchronizer(
            [scan1_sub, scan2_sub],
            queue_size=self.params['queue_size'],
            slop=self.params['time_sync_tolerance']
        )
        self.synchronizer.registerCallback(self.sync_callback)
        
        self.get_logger().info("Scan Merger Node started. Waiting for scans...")
        self.get_logger().info(f"Scan1 (Lidar): {self.params['scan1_topic']}")
        self.get_logger().info(f"Scan2 (Camera): {self.params['scan2_topic']}")
        self.get_logger().info(f"Camera priority: {self.params['camera_priority']} "
                              f"(up to {self.params['camera_priority_range']}m)")
        self.get_logger().info(f"Camera FOV: {np.degrees(self.params['camera_fov']):.1f}°")
        self.get_logger().info(f"Camera center angle: {np.degrees(self.params['camera_center_angle']):.1f}°")
        
        # Biến để theo dõi hiệu suất
        self.scan_count = 0
        self.last_time = self.get_clock().now()
        
    def sync_callback(self, scan1_msg, scan2_msg):
        """Xử lý khi nhận được message đồng bộ"""
        current_time = self.get_clock().now()
        
        # Kiểm tra chênh lệch thời gian
        time_diff = abs(scan1_msg.header.stamp.sec - scan2_msg.header.stamp.sec) + \
                   abs(scan1_msg.header.stamp.nanosec - scan2_msg.header.stamp.nanosec) * 1e-9
                   
        if time_diff > self.params['time_sync_tolerance']:
            self.get_logger().warn(
                f"Time diff exceeded tolerance: {time_diff:.4f}s > {self.params['time_sync_tolerance']}s",
                throttle_duration_sec=2.0
            )
        
        # Tạo message scan hợp nhất
        merged_scan = LaserScan()
        merged_scan.header = scan1_msg.header
        merged_scan.header.frame_id = self.params['output_frame']
        merged_scan.angle_min = self.params['angle_min']
        merged_scan.angle_max = self.params['angle_max']
        merged_scan.angle_increment = self.params['angle_increment']
        merged_scan.time_increment = 0.0  # Không xác định
        merged_scan.scan_time = max(scan1_msg.scan_time, scan2_msg.scan_time)
        merged_scan.range_min = self.params['range_min']
        merged_scan.range_max = self.params['range_max']
        
        # Tính số điểm trong scan hợp nhất
        num_readings = int((merged_scan.angle_max - merged_scan.angle_min) / 
                          merged_scan.angle_increment)
        merged_scan.ranges = [float('inf')] * num_readings
        merged_scan.intensities = [0.0] * num_readings  # Sử dụng để đánh dấu nguồn dữ liệu
        
        # Hàm chuyển đổi góc thành index trong mảng
        def angle_to_index(angle):
            # Chuẩn hóa góc về [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            # Tính index
            index = int(round((normalized_angle - merged_scan.angle_min) / merged_scan.angle_increment))
            # Đảm bảo index nằm trong giới hạn
            if index < 0:
                return 0
            if index >= num_readings:
                return num_readings - 1
            return index
        
        # Tính góc quét camera (horizontal FOV)
        camera_center_angle = self.params['camera_center_angle']
        camera_fov = self.params['camera_fov']
        camera_min_angle = camera_center_angle - camera_fov/2
        camera_max_angle = camera_center_angle + camera_fov/2
        
        # 1. Xử lý dữ liệu từ camera - CHỈ trong phạm vi góc quét và khoảng cách hiệu quả
        for i, r in enumerate(scan2_msg.ranges):
            # Kiểm tra khoảng cách hợp lệ
            if r < scan2_msg.range_min or r > scan2_msg.range_max:
                continue
                
            # Tính góc của điểm dữ liệu
            angle = scan2_msg.angle_min + i * scan2_msg.angle_increment
            
            # Kiểm tra xem góc có nằm trong FOV của camera không
            if not (camera_min_angle <= angle <= camera_max_angle):
                continue
                
            # Chỉ xử lý trong khoảng ưu tiên của camera
            if r > self.params['camera_priority_range']:
                continue
                
            # Chuyển góc thành index
            idx = angle_to_index(angle)
            
            if 0 <= idx < num_readings:
                # Ghi nhận dữ liệu từ camera
                merged_scan.ranges[idx] = r
                merged_scan.intensities[idx] = 1.0  # Đánh dấu dữ liệu từ camera
        
        # 2. Xử lý dữ liệu từ lidar - CHO TOÀN BỘ VÙNG QUÉT
        lidar_points_added = 0
        camera_points_added = 0
        
        for i, r in enumerate(scan1_msg.ranges):
            # Kiểm tra khoảng cách hợp lệ
            if r < scan1_msg.range_min or r > scan1_msg.range_max:
                continue
                
            # Tính góc của điểm dữ liệu
            angle = scan1_msg.angle_min + i * scan1_msg.angle_increment
            idx = angle_to_index(angle)
            
            if 0 <= idx < num_readings:
                current_value = merged_scan.ranges[idx]
                current_source = merged_scan.intensities[idx]
                
                # Xác định xem góc này có nằm trong FOV camera không
                in_camera_fov = (camera_min_angle <= angle <= camera_max_angle)
                
                # QUY TẮC HỢP NHẤT:
                if not in_camera_fov:
                    # Trường hợp 1: Ngoài FOV camera -> luôn sử dụng lidar
                    merged_scan.ranges[idx] = r
                    merged_scan.intensities[idx] = 0.0
                    lidar_points_added += 1
                elif in_camera_fov and current_value == float('inf'):
                    # Trường hợp 2: Trong FOV camera nhưng camera không có dữ liệu -> sử dụng lidar
                    merged_scan.ranges[idx] = r
                    merged_scan.intensities[idx] = 0.0
                    lidar_points_added += 1
                elif in_camera_fov and current_source > 0.5:
                    # Trường hợp 3: Camera đã cung cấp dữ liệu cho vùng này
                    camera_points_added += 1
                    
                    # Nếu lidar phát hiện vật thể gần hơn, ưu tiên lidar
                    if r < current_value:
                        merged_scan.ranges[idx] = r
                        merged_scan.intensities[idx] = 0.0
                        lidar_points_added += 1
        
        # Lọc "bóng ma" (ghost objects)
        if self.params['filter_ghosts']:
            self.filter_ghosts(merged_scan.ranges, self.params['ghost_threshold'])
        
        # Publish scan hợp nhất
        self.merged_pub.publish(merged_scan)
        
        # Log hiệu suất
        self.scan_count += 1
        if self.scan_count % 10 == 0:
            delta = current_time - self.last_time
            freq = 10 / delta.nanoseconds * 1e9
            self.get_logger().info(
                f"Publishing merged scan at {freq:.1f} Hz | "
                f"Camera points: {camera_points_added} | "
                f"Lidar points: {lidar_points_added} | "
                f"Camera FOV: {np.degrees(camera_min_angle):.1f}° to {np.degrees(camera_max_angle):.1f}°"
            )
            self.last_time = current_time
    
    def filter_ghosts(self, ranges, threshold):
        """Lọc các giá trị bất thường (ghost objects)"""
        for i in range(1, len(ranges)-1):
            if ranges[i] == float('inf'):
                continue
                
            # Tính sự khác biệt với các điểm lân cận
            diff_prev = abs(ranges[i] - ranges[i-1])
            diff_next = abs(ranges[i] - ranges[i+1])
            
            # Nếu điểm hiện tại khác biệt lớn với cả hai lân cận
            if diff_prev > threshold and diff_next > threshold:
                # Thay thế bằng trung bình của lân cận
                ranges[i] = (ranges[i-1] + ranges[i+1]) / 2

def main(args=None):
    rclpy.init(args=args)
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()