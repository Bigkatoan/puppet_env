import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
import cv_bridge
import math
import time
import cv2
from collections import deque

class Ros2RobotEnv(gym.Env, Node):
    """
    Custom Gymnasium environment for a ROS2-enabled robot.

    The environment interacts with a robot in a simulated or real-world setting
    through ROS2 topics. It's designed for reinforcement learning tasks where
    the robot learns to navigate to a target position.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, render_size=None):
        # Initialize the Gym Environment and ROS2 Node
        gym.Env.__init__(self)
        Node.__init__(self, 'ros2_robot_env')

        # --- ROS2 Subscribers ---
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.rgb_subscriber = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_subscriber = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/fast_pose', self.pose_callback, 10)
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # --- ROS2 Publishers ---
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Data Storage ---
        self.latest_scan = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pose = None
        self.latest_joint_states = None
        self.target_pose = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.previous_distance_to_goal = None
        self.map_data = None
        self.map_info = None
        self.free_space_indices = None
        self.trajectory = deque(maxlen=15) # Buffer to store recent poses for recovery

        # CV Bridge for image conversion
        self.bridge = cv_bridge.CvBridge()

        # --- Rendering ---
        self.render_mode = render_mode
        self.render_size = render_size
        if self.render_mode == 'human':
            cv2.namedWindow("ROS2 RL Render", cv2.WINDOW_AUTOSIZE)

        # --- Environment Spaces ---
        self.observation_space = spaces.Dict({
            'scan': spaces.Box(low=0, high=100, shape=(360,), dtype=np.float32),
            'rgb': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=10, shape=(480, 640), dtype=np.float32),
            'wheel_velocities': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'current_pose': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'target_pose': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)

        # --- Episode and Reward Parameters ---
        self.max_steps = 1000
        self.current_step = 0
        self.goal_distance_threshold = 0.2
        self.collision_distance_threshold = 0.15
        self.goal_clearance_cells = 5 

        # --- Shaped Reward Parameters ---
        self.progress_reward_weight = 20.0
        self.obstacle_penalty_weight = 1.5
        self.safety_margin = 0.3
        self.time_penalty = -0.1

        self.get_logger().info("ROS2 Robot Environment Initialized.")
        time.sleep(2)

    # --- Subscriber Callbacks ---
    def map_callback(self, msg):
        if self.free_space_indices is not None:
            return

        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((self.map_info.height, self.map_info.width))
        
        binary_map = np.where(self.map_data == 0, 1, 0).astype(np.uint8)
        kernel = np.ones((self.goal_clearance_cells, self.goal_clearance_cells), np.uint8)
        eroded_map = cv2.erode(binary_map, kernel, iterations=1)
        
        free_space_cells = np.where(eroded_map == 1)
        self.free_space_indices = list(zip(free_space_cells[0], free_space_cells[1]))
        
        if self.free_space_indices:
            self.get_logger().info(f"Map processed. Found {len(self.free_space_indices)} safe goal locations.")
        else:
            self.get_logger().warn("Map received, but no safe free space was found after erosion!")

    def scan_callback(self, msg):
        processed_scan = np.full(360, np.inf, dtype=np.float32)
        for i, range_val in enumerate(msg.ranges):
            if not np.isfinite(range_val):
                continue
            angle_rad = msg.angle_min + i * msg.angle_increment
            angle_deg = np.rad2deg(angle_rad)
            degree_index = int(round(angle_deg)) % 360
            if range_val < processed_scan[degree_index]:
                processed_scan[degree_index] = range_val
        self.latest_scan = processed_scan

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.latest_pose = np.array([x, y, yaw], dtype=np.float32)

    def joint_state_callback(self, msg):
        if len(msg.velocity) >= 2:
            self.latest_joint_states = np.array([msg.velocity[0], msg.velocity[1]], dtype=np.float32)
        else:
            self.latest_joint_states = np.zeros(2, dtype=np.float32)

    # --- Core Environment Methods ---
    def _get_obs(self):
        while rclpy.ok() and (self.latest_scan is None or self.latest_rgb is None or self.latest_depth is None or self.latest_pose is None or self.latest_joint_states is None):
            self.get_logger().info("Waiting for sensor data...", once=True)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = {
            'scan': self.latest_scan if self.latest_scan is not None else np.full(360, np.inf, dtype=np.float32),
            'rgb': self.latest_rgb if self.latest_rgb is not None else np.zeros(self.observation_space['rgb'].shape, dtype=np.uint8),
            'depth': self.latest_depth if self.latest_depth is not None else np.zeros(self.observation_space['depth'].shape, dtype=np.float32),
            'wheel_velocities': self.latest_joint_states if self.latest_joint_states is not None else np.zeros(self.observation_space['wheel_velocities'].shape, dtype=np.float32),
            'current_pose': self.latest_pose if self.latest_pose is not None else np.zeros(self.observation_space['current_pose'].shape, dtype=np.float32),
            'target_pose': self.target_pose
        }
        return obs

    def _sample_new_goal(self):
        while self.free_space_indices is None and rclpy.ok():
            self.get_logger().info("Waiting for map to sample a new goal...", once=True)
            rclpy.spin_once(self, timeout_sec=0.5)
        
        random_cell_index = self.np_random.integers(len(self.free_space_indices))
        goal_cell = self.free_space_indices[random_cell_index]
        row, col = goal_cell
        
        goal_x = col * self.map_info.resolution + self.map_info.origin.position.x
        goal_y = row * self.map_info.resolution + self.map_info.origin.position.y
        
        return np.array([goal_x, goal_y], dtype=np.float32)

    def _recover_from_stuck_state(self):
        """
        Navigates the robot backwards along its recent trajectory to a known safe point.
        """
        self.get_logger().info("Collision or timeout: Attempting recovery by reversing trajectory...")
        
        if len(self.trajectory) < 5:
            self.get_logger().warn("Not enough trajectory points for recovery. Performing simple backup.")
            backup_msg = Twist()
            backup_msg.linear.x = -0.15
            end_time = time.time() + 1.5
            while time.time() < end_time:
                self.velocity_publisher.publish(backup_msg)
                time.sleep(0.1)
            return

        # Target the oldest point in our recent trajectory
        recovery_target = self.trajectory[0][:2]
        self.get_logger().info(f"Recovery target: {recovery_target}")

        # Simple P-controller to navigate to the recovery point
        recovery_timeout = time.time() + 10 # 10 second timeout for recovery
        while time.time() < recovery_timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_pose is None: continue

            current_pos = self.latest_pose[:2]
            current_yaw = self.latest_pose[2]
            
            distance_to_target = np.linalg.norm(current_pos - recovery_target)
            if distance_to_target < 0.15: # Close enough to recovery point
                self.get_logger().info("Successfully recovered to a safe point.")
                break

            angle_to_target = math.atan2(recovery_target[1] - current_pos[1], recovery_target[0] - current_pos[0])
            angle_error = angle_to_target - current_yaw
            
            # Normalize angle error to [-pi, pi]
            if angle_error > np.pi: angle_error -= 2 * np.pi
            if angle_error < -np.pi: angle_error += 2 * np.pi

            # Control logic
            twist_msg = Twist()
            if abs(angle_error) > 0.2: # If not facing the target, turn first
                twist_msg.angular.z = 0.8 * np.sign(angle_error)
            else: # If facing the target, move forward
                twist_msg.linear.x = 0.25
                twist_msg.angular.z = 0.4 * angle_error # Minor corrections while moving
            
            self.velocity_publisher.publish(twist_msg)
        else:
            self.get_logger().warn("Recovery maneuver timed out.")

        self.stop_robot()
        time.sleep(0.5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.stop_robot()
        self._recover_from_stuck_state()
        self.trajectory.clear() # Clear trajectory after recovery

        current_pose = self.latest_pose if self.latest_pose is not None else np.zeros(3)
        while True:
            self.target_pose = self._sample_new_goal()
            if np.linalg.norm(current_pose[:2] - self.target_pose) > 1.5: # Ensure goal is far enough
                break
        
        self.get_logger().info(f"New target generated: {self.target_pose}")
        
        self.latest_scan = self.latest_rgb = self.latest_depth = self.latest_joint_states = None
        observation = self._get_obs()
        
        initial_distance = np.linalg.norm(observation['current_pose'][:2] - self.target_pose)
        self.previous_distance_to_goal = initial_distance
        
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        self.last_action = action
        vel_msg = Twist()
        vel_msg.linear.x = float(action[0])
        vel_msg.angular.z = float(action[1])
        self.velocity_publisher.publish(vel_msg)
        rclpy.spin_once(self, timeout_sec=0.1)
        observation = self._get_obs()

        # Record current pose to the trajectory buffer
        if self.latest_pose is not None:
            self.trajectory.append(self.latest_pose)

        reward, terminated = self._calculate_reward(observation)
        truncated = self.current_step >= self.max_steps
        if terminated or truncated:
            self.stop_robot()
        info = {}
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, observation):
        terminated = False
        reward = 0.0
        current_pos = observation['current_pose'][:2]
        scan_data = observation['scan']
        distance_to_goal = np.linalg.norm(current_pos - self.target_pose)
        distance_reduction = self.previous_distance_to_goal - distance_to_goal
        reward += self.progress_reward_weight * distance_reduction
        self.previous_distance_to_goal = distance_to_goal
        
        d_obs = np.min(scan_data)
        if d_obs < self.safety_margin:
            reward += -self.obstacle_penalty_weight * (self.safety_margin - d_obs)**2
        
        reward += self.time_penalty
        
        if distance_to_goal < self.goal_distance_threshold:
            self.get_logger().info("Goal Reached!")
            reward += 100.0
            terminated = True
        
        if d_obs < self.collision_distance_threshold:
            self.get_logger().warn("Collision Detected!")
            reward -= 100.0
            terminated = True
            
        return reward, terminated

    def stop_robot(self):
        self.velocity_publisher.publish(Twist())
        self.get_logger().info("Robot stopped.", throttle_duration_sec=1)

    def render(self):
        if self.render_mode == 'human':
            obs = self._get_obs()
            rgb_img = obs['rgb'].copy()
            rgb_h, rgb_w, _ = rgb_img.shape
            info_panel_height = 150
            canvas = np.zeros((rgb_h + info_panel_height, rgb_w, 3), dtype=np.uint8)
            canvas[0:rgb_h, 0:rgb_w] = rgb_img
            info_panel = canvas[rgb_h:, :]
            info_panel.fill(50)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            texts = [
                f"Pose: [x:{obs['current_pose'][0]:.2f}, y:{obs['current_pose'][1]:.2f}, th:{np.rad2deg(obs['current_pose'][2]):.1f}]",
                f"Target: [x:{obs['target_pose'][0]:.2f}, y:{obs['target_pose'][1]:.2f}]",
                f"Wheel Vels: [L:{obs['wheel_velocities'][0]:.2f}, R:{obs['wheel_velocities'][1]:.2f}]",
                f"Action: [Lin:{self.last_action[0]:.2f}, Ang:{self.last_action[1]:.2f}]"
            ]
            colors = [(255, 255, 255), (150, 255, 150), (255, 255, 255), (150, 150, 255)]
            for i, (text, color) in enumerate(zip(texts, colors)):
                cv2.putText(info_panel, text, (10, 30 * (i + 1)), font, 0.7, color, 2)

            robot_center_x, robot_center_y = rgb_w // 2, rgb_h // 2 
            for i, distance in enumerate(obs['scan']):
                if not (0.3 < distance < 12.0):
                    continue
                angle = np.deg2rad(i) - np.pi / 2
                px = int(robot_center_x - distance * 30 * np.cos(angle))
                py = int(robot_center_y - distance * 30 * np.sin(angle))
                if 0 <= px < rgb_w and 0 <= py < rgb_h:
                    cv2.circle(canvas, (px, py), 3, (0, 255, 255), -1)
            
            final_canvas = cv2.resize(canvas, self.render_size, interpolation=cv2.INTER_AREA) if self.render_size else canvas
            cv2.imshow("ROS2 RL Render", final_canvas)
            cv2.waitKey(1)

    def close(self):
        self.stop_robot()
        if self.render_mode == 'human':
            cv2.destroyAllWindows()
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    env = Ros2RobotEnv(render_mode='human', render_size=(256, 256))
    try:
        for episode in range(5):
            obs, info = env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                env.render()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += reward
                step += 1
                env.get_logger().info(f"E:{episode+1} S:{step}, R:{reward:.2f}, TotalR:{score:.2f}", throttle_duration_sec=0.5)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
