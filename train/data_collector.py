import rclpy
import cv2
import os
import csv
import numpy as np
import termios
import tty
import sys
import threading
import time
import shutil
from datetime import datetime

from puppet import Ros2RobotEnv # Imports your custom environment
from geometry_msgs.msg import Twist

# --- Helper function for non-blocking keyboard input on Linux ---
def get_key():
    """Reads a single keypress from the terminal without waiting for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# --- Main Data Collection Class ---

class DataCollector:
    def __init__(self, dataset_path='phase1_dataset', capture_interval=0.5):
        # --- File and Directory Setup ---
        self.dataset_path = dataset_path
        self.capture_interval = capture_interval

        if os.path.exists(self.dataset_path):
            print(f"Deleting existing dataset at: {self.dataset_path}")
            shutil.rmtree(self.dataset_path)
        
        self.images_path = os.path.join(self.dataset_path, 'images')
        os.makedirs(self.images_path, exist_ok=True)
        print(f"Created new dataset directory at: {self.dataset_path}")
        
        self.csv_path = os.path.join(self.dataset_path, 'data.csv')
        self.is_first_write = True
        self.data_lock = threading.Lock()
        
        # --- ROS2 and Environment Setup ---
        rclpy.init(args=None)
        self.env = Ros2RobotEnv()
        self.running = True

        # --- Robot Control Parameters ---
        self.linear_speed = 0.25  # m/s
        self.angular_speed = 0.5  # rad/s
        
        print("\n--- Data Collector Initialized ---")
        self.print_instructions()

    def print_instructions(self):
        """Prints the updated control instructions to the user."""
        print("-----------------------------------")
        print("       Robot Driving Controls      ")
        print("-----------------------------------")
        print("          w : Foward")
        print("     a : Turn Left   d : Turn Right")
        print("          s : Backward")
        print("")
        print("     <spacebar> : STOP") # <<< THIS LINE IS NOW FIXED
        print("     q : Quit and Save")
        print("\n >> Data is being collected automatically! <<")
        print("-----------------------------------")

    def capture_data(self):
        """Captures one frame of data (image + pose) and saves it."""
        obs = self.env._get_obs()
        
        rgb_image = obs.get('rgb')
        current_pose = obs.get('current_pose') 

        if rgb_image is None or current_pose is None:
            return

        with self.data_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.png"
            filepath = os.path.join(self.images_path, filename)
            
            cv2.imwrite(filepath, rgb_image)

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if self.is_first_write:
                    writer.writerow(['filename', 'pos_x', 'pos_y', 'yaw'])
                    self.is_first_write = False
                
                writer.writerow([filename, current_pose[0], current_pose[1], current_pose[2]])
            
            print(f"Captured: {filename}", end='\r')

    def _data_collection_loop(self):
        """This loop runs in a background thread to automatically save data."""
        while self.running and rclpy.ok():
            self.capture_data()
            time.sleep(self.capture_interval)

    def run_teleop(self):
        """Main loop for teleoperation and data collection."""
        collector_thread = threading.Thread(target=self._data_collection_loop)
        collector_thread.daemon = True
        collector_thread.start()

        vel_msg = Twist()
        try:
            while self.running and rclpy.ok():
                key = get_key()
                
                if key == 'w':
                    vel_msg.linear.x = self.linear_speed
                    vel_msg.angular.z = 0.0
                elif key == 's':
                    vel_msg.linear.x = -self.linear_speed
                    vel_msg.angular.z = 0.0
                elif key == 'a':
                    vel_msg.angular.z = self.angular_speed
                    vel_msg.linear.x = 0.0
                elif key == 'd':
                    vel_msg.angular.z = -self.angular_speed
                    vel_msg.linear.x = 0.0
                elif key == ' ': # Spacebar for immediate stop
                    vel_msg = Twist()
                elif key.lower() == 'q':
                    self.running = False
                    vel_msg = Twist() # Send a final stop command before quitting
                
                self.env.velocity_publisher.publish(vel_msg)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            print("\nStopping robot and shutting down...")
            self.running = False
            collector_thread.join(timeout=1.0)
            self.env.stop_robot()
            self.env.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    collector = DataCollector(capture_interval=0.5)
    collector.run_teleop()