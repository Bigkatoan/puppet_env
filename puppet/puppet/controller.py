#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import can
import math
import time
import struct  # Thêm thư viện struct

class MotorControllerROS2(Node):
    def __init__(self):
        super().__init__('motor_controller_ros2')
        
        # Parameters
        self.declare_parameter('wheel_radius', 0.034)
        self.declare_parameter('track_width', 0.309716)
        self.declare_parameter('can_tx_id', 0x21)
        self.declare_parameter('can_rx_id', 0x13)
        self.declare_parameter('max_rpm', 280.0)
        
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.track_width = self.get_parameter('track_width').get_parameter_value().double_value
        self.can_tx_id = self.get_parameter('can_tx_id').get_parameter_value().integer_value
        self.can_rx_id = self.get_parameter('can_rx_id').get_parameter_value().integer_value
        self.max_rpm = self.get_parameter('max_rpm').get_parameter_value().double_value
        
        # Initialize variables
        self.bus = None
        self.last_cmd_time = time.time()
        self.cmd_timeout = 0.5
        self.message_count = 0
        self.feedback_count = 0
        self.current_motor_rpm = [0.0, 0.0, 0.0, 0.0]
        
        self.get_logger().info("🚀 Starting Motor Controller ROS2...")
        self.get_logger().info("🔧 Motor Layout: M0=RL, M1=FL, M2=FR, M3=RR")
        
        # Initialize CAN bus
        try:
            self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
            self.get_logger().info("✅ CAN bus initialized successfully")
            self.get_logger().info(f"📡 CAN TX ID: 0x{self.can_tx_id:02X}, RX ID: 0x{self.can_rx_id:02X}")
            
            # Test connection
            self.send_motor_rpm(0, 0, 0, 0)
            self.get_logger().info("✅ CAN test passed")
            
        except Exception as e:
            self.get_logger().error(f"❌ CAN initialization failed: {e}")
            self.get_logger().error("Please ensure CAN interface is properly configured")
            raise Exception("CAN initialization failed")
        
        # Subscriber
        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # Timers
        self.heartbeat_timer = self.create_timer(0.05, self.heartbeat_callback)
        self.feedback_timer = self.create_timer(0.1, self.read_can_feedback)
        
        self.get_logger().info("✅ Motor Controller ROS2 started successfully")

    def send_motor_rpm(self, m0, m1, m2, m3):
        """Send RPM command to ESP32 - SỬA LẠI THEO ĐỊNH DẠNG THÀNH CÔNG"""
        if not self.bus:
            return False
            
        try:
            # Constrain RPM
            motors = [
                int(max(-self.max_rpm, min(self.max_rpm, m0))),  # M0 = Rear Left
                int(max(-self.max_rpm, min(self.max_rpm, m1))),  # M1 = Front Left
                int(max(-self.max_rpm, min(self.max_rpm, m2))),  # M2 = Front Right
                int(max(-self.max_rpm, min(self.max_rpm, m3)))   # M3 = Rear Right
            ]
            
            # SỬA: Đóng gói dữ liệu sử dụng struct.pack
            data = bytearray()
            for rpm in motors:
                # Sử dụng định dạng '>h' = big-endian signed short (2 byte)
                data.extend(struct.pack('>h', rpm))
            
            # Send message
            message = can.Message(
                arbitration_id=self.can_tx_id,
                data=data,
                is_extended_id=False
            )
            self.bus.send(message, timeout=0.1)
            self.message_count += 1
            
            # Log
            if self.message_count % 20 == 0 or any(abs(rpm) > 0 for rpm in motors):
                self.get_logger().info(f"📤 RPM [RL,FL,FR,RR]: [{motors[0]}, {motors[1]}, {motors[2]}, {motors[3]}]")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Send failed: {e}")
            return False

    def read_can_feedback(self):
        """Read feedback from ESP32"""
        if not self.bus:
            return
            
        try:
            message = self.bus.recv(timeout=0.01)
            if message and message.arbitration_id == self.can_rx_id:
                if len(message.data) == 8:
                    for i in range(4):
                        # SỬA: Sử dụng struct.unpack để giải mã
                        raw_rpm = struct.unpack('>h', message.data[i*2:i*2+2])[0]
                        self.current_motor_rpm[i] = raw_rpm / 100.0
                    
                    self.feedback_count += 1
                    if self.feedback_count % 50 == 0:
                        self.get_logger().info(f"📥 Feedback [RL,FL,FR,RR]: [{self.current_motor_rpm[0]:.1f}, {self.current_motor_rpm[1]:.1f}, {self.current_motor_rpm[2]:.1f}, {self.current_motor_rpm[3]:.1f}]")
        except Exception as e:
            self.get_logger().error(f"❌ Feedback error: {e}")

    def cmd_vel_callback(self, msg):
        """Process cmd_vel với mapping đúng"""
        try:
            self.last_cmd_time = time.time()
            
            v = msg.linear.x
            omega = msg.angular.z
            
            if abs(v) > 0.01 or abs(omega) > 0.01:
                self.get_logger().info(f"📥 cmd_vel: v={v:.3f}, ω={omega:.3f}")
            
            # Calculate wheel velocities for differential drive
            v_left = v - omega * self.track_width / 2
            v_right = v + omega * self.track_width / 2
            
            # Convert to RPM
            rpm_left = v_left / self.wheel_radius * 60 / (2 * math.pi)
            rpm_right = v_right / self.wheel_radius * 60 / (2 * math.pi)
            
            if abs(rpm_left) > 1 or abs(rpm_right) > 1:
                self.get_logger().info(f"🔧 Calculated RPM: Left={rpm_left:.1f}, Right={rpm_right:.1f}")
            
            # ===== MAPPING ĐÚNG THEO LAYOUT =====
            # M0 = Rear Left,  M1 = Front Left,  M2 = Front Right,  M3 = Rear Right
            # Left side: M0, M1 = rpm_left
            # Right side: M2, M3 = rpm_right
            
            self.send_motor_rpm(
                rpm_left,   # M0 = Rear Left
                rpm_left,   # M1 = Front Left  
                rpm_right,  # M2 = Front Right
                rpm_right   # M3 = Rear Right
            )
            
        except Exception as e:
            self.get_logger().error(f"❌ cmd_vel error: {e}")

    def heartbeat_callback(self):
        """Heartbeat and timeout check"""
        current_time = time.time()
        if current_time - self.last_cmd_time > self.cmd_timeout:
            self.send_motor_rpm(0, 0, 0, 0)

    def emergency_stop(self):
        """Emergency stop"""
        try:
            self.get_logger().info("🛑 EMERGENCY STOP!")
            for _ in range(5):
                self.send_motor_rpm(0, 0, 0, 0)
                time.sleep(0.01)
            self.get_logger().info("✅ Motors stopped")
        except Exception as e:
            self.get_logger().error(f"❌ Emergency stop error: {e}")

    def destroy_node(self):
        """Cleanup"""
        self.emergency_stop()
        if self.bus:
            try:
                self.bus.shutdown()
            except:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = MotorControllerROS2()
        rclpy.spin(node)
        
    except Exception as e:
        if node:
            node.get_logger().error(f"❌ Critical error: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()
        print("🏁 Node terminated")

if __name__ == '__main__':
    main()