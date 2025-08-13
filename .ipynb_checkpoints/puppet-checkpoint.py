import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
import cv_bridge
import math
import time
import cv2

class Ros2RobotEnv(gym.Env, Node):
    """
    Custom Gymnasium environment for a ROS2-enabled robot.

    The environment interacts with a robot in a simulated or real-world setting
    through ROS2 topics. It's designed for reinforcement learning tasks where
    the robot learns to navigate to a target position.

    Observation Space:
    A dictionary containing:
    - 'scan': LaserScan data, processed into a 360-point array.
    - 'rgb': RGB camera image.
    - 'depth': Depth camera image.
    - 'wheel_velocities': Velocities of the robot's wheels.
    - 'current_pose': The robot's current position and orientation.
    - 'target_pose': The target position for the robot to reach.

    Action Space:
    A continuous 2D space representing linear and angular velocity (cmd_vel).
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

        # --- ROS2 Publisher ---
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

        # CV Bridge for image conversion
        self.bridge = cv_bridge.CvBridge()

        # --- Rendering ---
        self.render_mode = render_mode
        self.render_size = render_size # Tuple (width, height) for resizing the render window
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
        
        # --- Shaped Reward Parameters (based on research paper) ---
        self.progress_reward_weight = 20.0  # C1: Scales reward for getting closer
        self.obstacle_penalty_weight = 1.5   # C2: Scales penalty for being near obstacles
        self.safety_margin = 0.3             # d_safe: Safety distance from obstacles
        self.time_penalty = -0.1             # Constant penalty per step

        self.get_logger().info("ROS2 Robot Environment Initialized.")
        time.sleep(2)

    # --- Subscriber Callbacks ---
    def scan_callback(self, msg):
        """
        Processes raw LaserScan data into a 360-point array.
        Each index in the array corresponds to an integer degree (0-359).
        """
        # Create a new array of 360 points, one for each degree.
        # Initialize with a large value (or np.inf) to represent no obstacle.
        processed_scan = np.full(360, np.inf, dtype=np.float32)

        # For each measurement in the raw scan data...
        for i, range_val in enumerate(msg.ranges):
            # If the range is invalid (inf or nan), skip it.
            if not np.isfinite(range_val):
                continue

            # Calculate the angle of the measurement in degrees.
            angle_rad = msg.angle_min + i * msg.angle_increment
            angle_deg = np.rad2deg(angle_rad)

            # Normalize the angle to be within [0, 359].
            # We round to the nearest integer degree.
            degree_index = int(round(angle_deg)) % 360

            # If we have multiple measurements for the same degree,
            # we take the minimum one, as it represents the closest obstacle at that angle.
            if range_val < processed_scan[degree_index]:
                processed_scan[degree_index] = range_val

        # Store the processed 360-point scan.
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stop_robot()
        self.latest_scan = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pose = None
        self.latest_joint_states = None
        self.target_pose = self.np_random.uniform(low=[-5.0, -5.0], high=[5.0, 5.0]).astype(np.float32)
        self.get_logger().info(f"New target generated: {self.target_pose}")
        
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
        r_progress = self.progress_reward_weight * distance_reduction
        reward += r_progress
        self.previous_distance_to_goal = distance_to_goal
        d_obs = np.min(scan_data) # np.min works because inf is ignored implicitly
        if d_obs < self.safety_margin:
            r_obstacle = -self.obstacle_penalty_weight * (self.safety_margin - d_obs)**2
            reward += r_obstacle
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
        stop_msg = Twist()
        self.velocity_publisher.publish(stop_msg)
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
            current_pose = obs['current_pose']
            target_pose = obs['target_pose']
            wheel_vels = obs['wheel_velocities']
            action = self.last_action
            pose_text = f"Pose: [x:{current_pose[0]:.2f}, y:{current_pose[1]:.2f}, th:{np.rad2deg(current_pose[2]):.1f}]"
            target_text = f"Target: [x:{target_pose[0]:.2f}, y:{target_pose[1]:.2f}]"
            vel_text = f"Wheel Vels: [L:{wheel_vels[0]:.2f}, R:{wheel_vels[1]:.2f}]"
            action_text = f"Action: [Lin:{action[0]:.2f}, Ang:{action[1]:.2f}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(info_panel, pose_text, (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, target_text, (10, 60), font, 0.7, (150, 255, 150), 2)
            cv2.putText(info_panel, vel_text, (10, 90), font, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, action_text, (10, 120), font, 0.7, (150, 150, 255), 2)
            scan_data = obs['scan']
            robot_center_x, robot_center_y = rgb_w // 2, rgb_h // 2 
            for i, distance in enumerate(scan_data):
                if not (0.3 < distance < 12.0):
                    continue
                angle = np.deg2rad(i)
                pixel_dist = distance * 30
                img_angle = angle - np.pi / 2
                px = int(robot_center_x - pixel_dist * np.cos(img_angle))
                py = int(robot_center_y - pixel_dist * np.sin(img_angle))
                if 0 <= px < rgb_w and 0 <= py < rgb_h:
                    cv2.circle(canvas, (px, py), 3, (0, 255, 255), -1)
            final_canvas = canvas
            if self.render_size is not None and len(self.render_size) == 2:
                final_canvas = cv2.resize(canvas, self.render_size, interpolation=cv2.INTER_AREA)
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
