#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Pose, TransformStamped
import math
import numpy as np
import tf2_ros
# import tf_transformations # Cần cài đặt: pip install tf-transformations

def euler_to_quaternion(roll, pitch, yaw):
    """
    Chuyển đổi góc Euler (roll, pitch, yaw) sang Quaternion.
    Hàm này dành cho ROS 2 và trả về một đối tượng geometry_msgs.msg.Quaternion.
    
    Các góc đầu vào phải ở đơn vị radian.
    """
    try:
        # Chuyển đổi đầu vào sang float để tính toán
        roll = float(roll)
        pitch = float(pitch)
        yaw = float(yaw)

        # Tính toán các giá trị cos và sin của một nửa góc
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        # Tạo đối tượng Quaternion từ geometry_msgs
        q = Quaternion()

        # Tính toán các thành phần của quaternion theo công thức
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    except Exception as e:
        print(f"Lỗi trong hàm euler_to_quaternion: {e}")
        # Trả về quaternion đơn vị (không xoay) nếu có lỗi
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

class PureEncoderOdometryNode(Node):
    """
    Node ROS 2 tính toán Odometry chỉ dựa trên dữ liệu từ encoder bánh xe,
    publish lên /odom_encoder. TF transform có thể được bật/tắt.
    Bao gồm tham số hiệu chỉnh vận tốc bánh xe.
    """
    def __init__(self):
        super().__init__('pure_encoder_odometry_node')
        self.get_logger().info("Khoi tao Node Pure Encoder Odometry Calculator (with calibration)...")

        # --- Tham số ---
        self.declare_parameter('base_link_frame', 'base_link')
        self.declare_parameter('odom_encoder_frame', 'odom')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('odom_encoder_topic', '/odom')
        self.declare_parameter('wheel_radius', 0.034) # mét
        self.declare_parameter('track_width', 0.309716) # mét
        self.declare_parameter('joint_order', [
            'rear_left_wheel_joint', # Tương ứng với RPM[0] từ CAN?
            'front_left_wheel_joint',  # Tương ứng với RPM[1] từ CAN?
            'front_right_wheel_joint',   # Tương ứng với RPM[2] từ CAN?
            'rear_right_wheel_joint' 
        ])
        self.declare_parameter('publish_frequency', 20.0) # Hz
        self.declare_parameter('publish_tf', True)
        # *** THÊM THAM SỐ HIỆU CHỈNH VẬN TỐC ***
        self.declare_parameter('left_wheel_velocity_scale', 1.0)
        self.declare_parameter('right_wheel_velocity_scale', 1.0)


        # Lấy tham số
        self.base_link_frame_id = self.get_parameter('base_link_frame').get_parameter_value().string_value
        self.odom_encoder_frame_id = self.get_parameter('odom_encoder_frame').get_parameter_value().string_value
        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.odom_encoder_topic_out = self.get_parameter('odom_encoder_topic').get_parameter_value().string_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.track_width = self.get_parameter('track_width').get_parameter_value().double_value
        self.expected_joint_order = self.get_parameter('joint_order').get_parameter_value().string_array_value
        publish_frequency = self.get_parameter('publish_frequency').get_parameter_value().double_value
        self.publish_period = 1.0 / publish_frequency if publish_frequency > 0 else 0.05
        self.should_publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        # *** LẤY THAM SỐ HIỆU CHỈNH ***
        self.left_wheel_scale = self.get_parameter('left_wheel_velocity_scale').get_parameter_value().double_value
        self.right_wheel_scale = self.get_parameter('right_wheel_velocity_scale').get_parameter_value().double_value


        self.get_logger().info(f"Subscribing to JointStates: {self.joint_state_topic}")
        self.get_logger().info(f"Publishing Odometry to: {self.odom_encoder_topic_out} with frame_id='{self.odom_encoder_frame_id}'")
        if self.should_publish_tf:
            self.get_logger().info(f"Publishing TF: {self.odom_encoder_frame_id} -> {self.base_link_frame_id}")
        else:
            self.get_logger().info(f"TF publishing from this node is DISABLED ({self.odom_encoder_frame_id} -> {self.base_link_frame_id})")
        self.get_logger().info(f"Wheel Radius: {self.wheel_radius}, Track Width: {self.track_width}")
        self.get_logger().info(f"Velocity Scales: Left={self.left_wheel_scale:.4f}, Right={self.right_wheel_scale:.4f}")


        if abs(self.track_width) < 1e-6:
            self.get_logger().fatal("Track width is zero or too small! Odometry calculation will be incorrect.")
            self.create_timer(0.1, lambda: self.context.try_shutdown()); return

        # --- Biến trạng thái ---
        self.current_wheel_velocities = {}
        self.last_joint_state_time = None
        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.last_update_time = self.get_clock().now()

        # --- Subscribers ---
        qos_profile_reliable = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_state_topic, self.joint_state_callback, qos_profile_reliable)

        # --- Publisher và Broadcaster ---
        self.odom_pub = self.create_publisher(Odometry, self.odom_encoder_topic_out, qos_profile_reliable)
        if self.should_publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None

        # --- Timer ---
        if rclpy.ok():
            self.odom_timer = self.create_timer(self.publish_period, self.calculate_and_publish_odometry)
            self.get_logger().info("Node Pure Encoder Odometry Calculator da san sang.")
        else:
            self.get_logger().error("Khoi tao node that bai truoc khi tao timer.")

    def joint_state_callback(self, msg: JointState):
        self.last_joint_state_time = self.get_clock().now()
        if len(msg.name) == len(msg.velocity):
            self.current_wheel_velocities = dict(zip(msg.name, msg.velocity))
        else:
            self.get_logger().warn("JointState message không hợp lệ (name/velocity length mismatch).")

    def calculate_and_publish_odometry(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        dt_safe = max(dt, 1e-6)

        if dt <= 0: return
        if self.last_joint_state_time is None or not self.current_wheel_velocities:
            self.get_logger().warn("Chua nhan duoc du lieu JointState hop le de tinh toan odometry.", throttle_duration_sec=5.0)
            return

        vx_robot = 0.0; omega_z_robot = 0.0
        if len(self.current_wheel_velocities) >= 4:
            try:
                omega_lr = self.current_wheel_velocities[self.expected_joint_order[0]]
                omega_lf = self.current_wheel_velocities[self.expected_joint_order[1]]
                omega_rf = self.current_wheel_velocities[self.expected_joint_order[2]]
                omega_rr = self.current_wheel_velocities[self.expected_joint_order[3]]
                
                # Vận tốc dài của từng bên (chưa hiệu chỉnh)
                v_right_raw = (omega_rf + omega_rr) / 2.0 * self.wheel_radius
                v_left_raw = (omega_lf + omega_lr) / 2.0 * self.wheel_radius

                #v_left_raw, v_right_raw = v_right_raw, v_left_raw
                
                # *** ÁP DỤNG HỆ SỐ HIỆU CHỈNH ***
                v_right_scaled = v_right_raw * self.right_wheel_scale
                v_left_scaled = v_left_raw * self.left_wheel_scale
                
                # Tính toán vận tốc robot dựa trên vận tốc đã hiệu chỉnh
                vx_robot = (v_right_scaled + v_left_scaled) / 2.0
                if abs(self.track_width) > 1e-6:
                    # Quy ước: omega_z dương là quay trái (CCW) -> v_right > v_left
                    omega_z_robot = (v_right_scaled - v_left_scaled) / self.track_width
                else: omega_z_robot = 0.0

            except KeyError as e:
                self.get_logger().warn(f"Khong tim thay khop noi '{e}' trong du lieu JointState khi tinh odometry."); return
            except Exception as e_calc:
                 self.get_logger().error(f"Loi khi tinh van toc encoder cho odometry: {e_calc}"); return
        else:
            self.get_logger().warn("Khong du du lieu van toc banh xe de tinh odometry.", throttle_duration_sec=5.0); return

        delta_theta = omega_z_robot * dt_safe
        delta_x = vx_robot * math.cos(self.theta + delta_theta / 2.0) * dt_safe
        delta_y = vx_robot * math.sin(self.theta + delta_theta / 2.0) * dt_safe
        self.x += delta_x; self.y += delta_y; self.theta += delta_theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = self.odom_encoder_frame_id
        odom_msg.child_frame_id = self.base_link_frame_id
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, self.theta)
        odom_msg.twist.twist.linear.x = vx_robot
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = omega_z_robot
        odom_msg.pose.covariance = np.diag([0.1**2, 0.1**2, 1e-9, 1e-9, 1e-9, (np.deg2rad(5.0))**2]).flatten().tolist()
        odom_msg.twist.covariance = np.diag([0.05**2, 1e-9, 1e-9, 1e-9, 1e-9, (np.deg2rad(2.0))**2]).flatten().tolist()
        self.odom_pub.publish(odom_msg)

        if self.should_publish_tf and self.tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = current_time.to_msg()
            t.header.frame_id = self.odom_encoder_frame_id
            t.child_frame_id = self.base_link_frame_id
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

        self.last_update_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = PureEncoderOdometryNode()
    try:
        if rclpy.ok(): rclpy.spin(node)
    except KeyboardInterrupt:
        print("Nhan Ctrl+C de thoat.")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        print("Chuong trinh Encoder Odometry ket thuc.")

if __name__ == '__main__':
    main()