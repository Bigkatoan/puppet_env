#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
import random
import math
from action_msgs.msg import GoalStatusArray
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose, do_transform_pose_stamped
from rclpy.duration import Duration

class SmartGoalGenerator(Node):
    def __init__(self):
        super().__init__('smart_goal_generator')
        
        # Parameters
        self.min_distance = 1.0
        self.goal_tolerance = 0.5
        self.wait_time = 3.0
        self.max_attempts = 50
        self.max_goal_radius = 10.0
        self.max_success_count = 50
        self.local_costmap_threshold = 50
        
        # Variables
        self.current_position = Point()
        self.global_costmap = None
        self.local_costmap = None
        self.goal_sent = False
        self.goal_reached = False
        self.last_goal_time = self.get_clock().now()
        self.success_count = 0
        self.current_goal = None
        self.last_goal_position = None
        self.nav2_status = "IDLE"
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Nav2 Action Client
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info("Waiting for Nav2 action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("Nav2 action server connected!")
        
        # Subscribers
        self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_costmap_callback,
            10)
            
        self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.local_costmap_callback,
            10)
            
        self.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self.goal_status_callback,
            10)
        
        # Timer for goal generation
        self.create_timer(0.5, self.timer_callback)
        
        self.get_logger().info("Smart Goal Generator initialized")
        self.get_logger().info(f"Will stop after {self.max_success_count} successful goals")

    def global_costmap_callback(self, msg):
        self.global_costmap = msg
        self.get_logger().debug("Received global costmap")

    def local_costmap_callback(self, msg):
        self.local_costmap = msg
        self.get_logger().debug("Received local costmap")

    def goal_status_callback(self, msg):
        """Xử lý trạng thái goal từ Nav2"""
        if not self.goal_sent:
            return
            
        for status in msg.status_list:
            # Cập nhật trạng thái Nav2
            if status.status == 1:
                self.nav2_status = "ACTIVE"
            elif status.status == 4:  # GoalStatus.STATUS_SUCCEEDED
                self.nav2_status = "SUCCEEDED"
                if not self.goal_reached:
                    self.get_logger().info("="*80)
                    self.get_logger().info("Nav2 reported goal succeeded!")
                    
                    # Kiểm tra xem goal có trùng với goal đã gửi không
                    if self.last_goal_position:
                        distance = math.sqrt(
                            (self.last_goal_position.x - self.current_goal.pose.position.x)**2 +
                            (self.last_goal_position.y - self.current_goal.pose.position.y)**2
                        )
                        self.get_logger().info(f"Distance between sent goal and Nav2 goal: {distance:.4f} m")
                    
                    self.get_logger().info(f"Goal position: "
                                          f"({self.current_goal.pose.position.x:.2f}, "
                                          f"{self.current_goal.pose.position.y:.2f})")
                    
                    self.goal_reached = True
                    self.last_goal_time = self.get_clock().now()
                    self.success_count += 1
                    self.last_goal_position = self.current_goal.pose.position
                    
                    if self.success_count >= self.max_success_count:
                        self.get_logger().info(f"Reached {self.max_success_count} successful goals. Shutting down.")
                        rclpy.shutdown()
                        return
                    
                    self.get_logger().info(f"Waiting for {self.wait_time} seconds before next goal...")
                    self.get_logger().info("="*80)
            elif status.status == 2:
                self.nav2_status = "CANCELED"
            elif status.status == 3:
                self.nav2_status = "ABORTED"

    def timer_callback(self):
        if not self.global_costmap or not self.local_costmap:
            self.get_logger().warn("Waiting for costmaps...", throttle_duration_sec=5)
            return
            
        if self.goal_reached:
            elapsed = (self.get_clock().now() - self.last_goal_time).nanoseconds / 1e9
            if elapsed >= self.wait_time:
                self.get_logger().info("="*80)
                self.get_logger().info("Wait time over. Generating new goal.")
                self.goal_reached = False
                self.goal_sent = False
                self.generate_new_goal()
                self.get_logger().info("="*80)
            else:
                remaining = self.wait_time - elapsed
                self.get_logger().info(f"Time remaining: {remaining:.1f} seconds", throttle_duration_sec=1)
            return
        
        if not self.goal_sent:
            self.generate_new_goal()

    def is_position_safe(self, position):
        """Kiểm tra vị trí có an toàn"""
        if not self.global_costmap:
            return False
        
        # Luôn kiểm tra global costmap
        if not self.check_global_costmap_position(position):
            return False
        
        # Chỉ kiểm tra local costmap nếu có dữ liệu và điểm nằm trong phạm vi local costmap
        if self.local_costmap:
            # Tính khoảng cách từ robot đến điểm
            distance = math.sqrt(
                (self.current_position.x - position.x)**2 + 
                (self.current_position.y - position.y)**2
            )
            
            # Chỉ kiểm tra nếu điểm nằm trong phạm vi local costmap
            if distance <= self.local_costmap.info.width * self.local_costmap.info.resolution / 2:
                if not self.check_local_costmap_position(position):
                    return False
        
        return True

    def check_global_costmap_position(self, position):
        """Kiểm tra vị trí trong global costmap"""
        return self.check_costmap_position(position, self.global_costmap, 'map')

    def check_local_costmap_position(self, position):
        """Kiểm tra vị trí trong local costmap"""
        return self.check_costmap_position(position, self.local_costmap, 'odom')

    def check_costmap_position(self, position, costmap, target_frame):
        """Kiểm tra vị trí trong costmap - Sử dụng PoseStamped thay vì PointStamped"""
        try:
            # Lấy transform với timeout ngắn
            transform = self.tf_buffer.lookup_transform(
                costmap.header.frame_id,
                target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            
            # Tạo PoseStamped thay vì PointStamped
            pose_in_target_frame = PoseStamped()
            pose_in_target_frame.header.frame_id = target_frame
            pose_in_target_frame.header.stamp = self.get_clock().now().to_msg()
            pose_in_target_frame.pose.position = position
            pose_in_target_frame.pose.orientation.w = 1.0  # Hướng mặc định

            # Chuyển đổi điểm sang frame của costmap
            pose_in_costmap_frame = do_transform_pose_stamped(pose_in_target_frame, transform)
            
            # Sử dụng điểm đã transform
            position_to_check = pose_in_costmap_frame.pose.position
            
            # Chuyển đổi tọa độ thế giới sang tọa độ bản đồ
            origin = costmap.info.origin.position
            resolution = costmap.info.resolution
            width = costmap.info.width
            height = costmap.info.height
            
            map_x = int((position_to_check.x - origin.x) / resolution)
            map_y = int((position_to_check.y - origin.y) / resolution)
            
            # Kiểm tra xem có nằm trong giới hạn bản đồ không
            if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
                return False
            
            # Kiểm tra giá trị cost
            index = map_y * width + map_x
            if index < 0 or index >= len(costmap.data):
                return False
            
            cost = costmap.data[index]
            return cost <= self.local_costmap_threshold
            
        except Exception as e:
            self.get_logger().warn(f"Transform error: {str(e)}")
            return False

    def generate_new_goal(self):
        """Tạo điểm đích mới trên toàn bộ global costmap"""
        if not self.global_costmap:
            self.get_logger().warn("Global costmap not available, cannot generate goal")
            return
            
        goal_found = False
        attempts = 0
        
        # Lấy kích thước global costmap
        origin = self.global_costmap.info.origin.position
        resolution = self.global_costmap.info.resolution
        width = self.global_costmap.info.width
        height = self.global_costmap.info.height
        
        while not goal_found and attempts < self.max_attempts:
            attempts += 1
            
            # Tạo điểm ngẫu nhiên trên toàn bộ global costmap
            map_x = random.randint(0, width - 1)
            map_y = random.randint(0, height - 1)
            
            # Chuyển đổi thành tọa độ thế giới
            x = origin.x + (map_x + 0.5) * resolution
            y = origin.y + (map_y + 0.5) * resolution
            position = Point(x=x, y=y, z=0.0)
            
            if self.is_position_safe(position):
                # Kiểm tra khoảng cách tối thiểu
                if hasattr(self, 'current_position'):
                    distance_to_robot = math.sqrt(
                        (self.current_position.x - x)**2 + 
                        (self.current_position.y - y)**2
                    )
                    
                    if distance_to_robot >= self.min_distance:
                        goal_found = True
                else:
                    goal_found = True
                    
                if goal_found:
                    # Tạo PoseStamped
                    goal_pose = PoseStamped()
                    goal_pose.header.frame_id = 'map'
                    goal_pose.header.stamp = self.get_clock().now().to_msg()
                    goal_pose.pose.position = position
                    goal_pose.pose.orientation.w = 1.0
                    
                    self.current_goal = goal_pose
                    
                    # Gửi goal tới Nav2
                    goal_msg = NavigateToPose.Goal()
                    goal_msg.pose = goal_pose
                    
                    self.goal_sent = True
                    self.goal_reached = False
                    self.send_goal_future = self.action_client.send_goal_async(goal_msg)
                    self.send_goal_future.add_done_callback(self.goal_response_callback)
                    
                    self.get_logger().info("="*80)
                    self.get_logger().info(f"Sending new goal #{self.success_count+1}: ({x:.2f}, {y:.2f})")
                    self.get_logger().info(f"Nav2 status: {self.nav2_status}")
                    self.get_logger().info("="*80)
        
        if not goal_found:
            self.get_logger().error(f"Failed to find valid goal after {self.max_attempts} attempts!")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by Nav2!")
            self.goal_sent = False
            self.nav2_status = "REJECTED"
        else:
            self.get_logger().info("Goal accepted by Nav2!")
            self.nav2_status = "ACCEPTED"

def main(args=None):
    rclpy.init(args=args)
    node = SmartGoalGenerator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()