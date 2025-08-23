#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import can
import math
import time
import threading
from collections import deque

class FastWheelVelocityCANReader(Node):
    def __init__(self):
        super().__init__('fast_wheel_velocity_can_reader')
        
        # Cấu hình CAN
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('calibration_factor', 1.15)
        
        can_interface = self.get_parameter('can_interface').get_parameter_value().string_value
        self.calibration_factor = self.get_parameter('calibration_factor').get_parameter_value().double_value
        
        # Khởi tạo publisher với queue size lớn hơn
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 1)  # Queue size nhỏ để real-time
        
        # Khởi tạo thông điệp JointState
        self.joint_msg = JointState()
        self.joint_msg.name = [
            'rear_left_wheel_joint',
            'front_left_wheel_joint', 
            'front_right_wheel_joint',
            'rear_right_wheel_joint'
        ]
        
        # Biến lưu trữ vị trí các khớp
        self.positions = [0.0, 0.0, 0.0, 0.0]
        self.current_velocities = [0.0, 0.0, 0.0, 0.0]
        self.last_time = time.time()
        
        # Thread-safe data sharing
        self.data_lock = threading.Lock()
        self.new_data_event = threading.Event()
        
        # Setup CAN interface
        self.get_logger().info(f"Setting up CAN interface {can_interface}...")
        
        try:
            self.can_bus = can.interface.Bus(channel='can0', bustype='socketcan')
            self.get_logger().info("CAN interface initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CAN: {e}")
            raise
        
        # Biến debug
        self.data_count = 0
        self.publish_count = 0
        self.last_rpm_values = [0.0, 0.0, 0.0, 0.0]
        
        # Bắt đầu CAN reader thread
        self.can_thread = threading.Thread(target=self.can_reader_thread, daemon=True)
        self.can_thread.start()
        
        # Timer với tần số rất cao để publish
        self.timer = self.create_timer(0.005, self.timer_callback)  # 200Hz
        
        # Timer debug
        #self.debug_timer = self.create_timer(1.0, self.debug_callback)

    def can_reader_thread(self):
        """Thread riêng để đọc CAN data liên tục"""
        self.get_logger().info("CAN reader thread started")
        
        while rclpy.ok():
            try:
                # Đọc CAN với timeout ngắn
                message = self.can_bus.recv(timeout=0.01)
                
                if message and message.arbitration_id == 0x13:
                    # Decode RPM nhanh
                    rpms = []
                    for i in range(4):
                        raw = (message.data[i*2] << 8) | message.data[i*2 + 1]
                        if raw > 32767: 
                            raw -= 65536
                        rpms.append(raw / 100.0)
                    
                    # Kiểm tra tính hợp lệ
                    if all(abs(rpm) < 10000 for rpm in rpms):
                        current_time = time.time()
                        
                        # Update data thread-safe
                        with self.data_lock:
                            # Tính dt
                            dt = current_time - self.last_time
                            
                            if 0.001 <= dt <= 0.2:  # 1ms đến 200ms
                                # Chuyển đổi RPM sang rad/s
                                new_velocities = [rpm * (math.pi / 30.0) for rpm in rpms]
                                
                                # Tích phân position
                                for i in range(4):
                                    delta_pos = new_velocities[i] * dt * self.calibration_factor
                                    self.positions[i] += delta_pos
                                
                                self.current_velocities = new_velocities
                                self.last_rpm_values = rpms.copy()
                                self.data_count += 1
                                
                            self.last_time = current_time
                            
                        # Signal có data mới
                        self.new_data_event.set()
                        
            except can.CanTimeoutError:
                continue
            except Exception as e:
                self.get_logger().debug(f"CAN thread error: {e}")
                continue

    def timer_callback(self):
        """Timer callback chỉ để publish data"""
        try:
            # Chỉ publish khi có data mới hoặc mỗi 50ms
            current_time = self.get_clock().now()
            
            with self.data_lock:
                positions = self.positions.copy()
                velocities = self.current_velocities.copy()
            
            # Tạo và publish message
            joint_msg = JointState()
            joint_msg.header.stamp = current_time.to_msg()
            joint_msg.name = self.joint_msg.name
            joint_msg.position = positions
            joint_msg.velocity = velocities
            
            self.publisher_.publish(joint_msg)
            self.publish_count += 1
            
            # Clear event sau khi publish
            self.new_data_event.clear()
            
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {str(e)}")

    def debug_callback(self):
        """Debug callback"""
        with self.data_lock:
            rpm_copy = self.last_rpm_values.copy()
            vel_copy = self.current_velocities.copy()
            pos_copy = self.positions.copy()
        
        self.get_logger().info(f"CAN: {self.data_count}/s, Publish: {self.publish_count}/s")
        self.get_logger().info(f"RPM: [{rpm_copy[0]:7.2f}, {rpm_copy[1]:7.2f}, {rpm_copy[2]:7.2f}, {rpm_copy[3]:7.2f}]")
        self.get_logger().info(f"Vel: [{vel_copy[0]:6.3f}, {vel_copy[1]:6.3f}, {vel_copy[2]:6.3f}, {vel_copy[3]:6.3f}] rad/s")
        
        self.data_count = 0
        self.publish_count = 0

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'can_bus'):
            try:
                self.can_bus.shutdown()
            except:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FastWheelVelocityCANReader()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped!")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Node failed: {str(e)}")
    finally:
        if node and hasattr(node, 'can_bus'):
            try:
                node.can_bus.shutdown()
            except:
                pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
