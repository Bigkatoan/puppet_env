# ROS2 Robot Navigation - Gymnasium Environment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A `gymnasium`-compatible environment for training reinforcement learning agents on a real or simulated ROS2-enabled robot. This environment provides a standardized interface for navigation tasks, connecting state-of-the-art RL algorithms with rich sensor data and robust, real-world-ready recovery mechanisms.

<p align="center">
  <!-- You can add a GIF or image of your robot in action here -->
  <!-- <img src="path/to/your/robot_demo.gif" width="600"> -->
</p>

---
## Installation

The environment requires a working ROS2 installation (Humble, Iron, etc.) and Python 3.8+.

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/Bigkatoan/puppet_env.git
    cd puppet_env
    ```

2.  **Install Python dependencies:**
    The core dependencies are `gymnasium` for the RL framework and `opencv-python` for rendering.
    ```bash
    pip install gymnasium
    pip install "numpy<2" opencv-python==4.5.5.64
    ```
    > **Note:** `rclpy` and `cv_bridge` are included with your ROS2 installation and do not need to be installed via pip.

---

## Usage

To use the environment, import `Ros2RobotEnv` and create an instance. Ensure your ROS2 environment is sourced and the robot simulation or hardware is running and publishing the required topics (`/map`, `/scan`, etc.).

```python
import gymnasium
import rclpy
from puppet import Ros2RobotEnv # Make sure the file is in your PYTHONPATH

def main():
    rclpy.init()

    # Create the environment
    # To enable rendering, use render_mode='human'
    # To resize the render window, use render_size=(width, height)
    env = Ros2RobotEnv(render_mode='human', render_size=(256, 256))

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    try:
        for episode in range(5):
            obs, info = env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                # Render the environment
                env.render()

                # Your agent would choose an action here
                action = env.action_space.sample() # Take a random action

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += reward
                step += 1

            print(f"Episode {episode + 1} finished after {step} steps with score {score:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Environment Details

### Observation Space

The observation is a `gymnasium.spaces.Dict` with the following keys:

| Key                | Data Type     | Shape         | Description                                        | ROS2 Topic                          |
| ------------------ | ------------- | ------------- | -------------------------------------------------- | ----------------------------------- |
| `scan`             | `np.float32`  | `(360,)`      | 360-degree laser scan distances.                   | `/scan`                             |
| `rgb`              | `np.uint8`    | `(480,640,3)` | Color camera image.                                | `/camera/camera/color/image_raw`    |
| `depth`            | `np.float32`  | `(480,640)`   | Depth camera image.                                | `/camera/camera/depth/image_rect_raw` |
| `wheel_velocities` | `np.float32`  | `(2,)`        | Velocities of the two main drive wheels.           | `/joint_states`                     |
| `current_pose`     | `np.float32`  | `(3,)`        | Robot's `[x, y, yaw]` position and orientation.    | `/fast_pose`                        |
| `target_pose`      | `np.float32`  | `(2,)`        | The `[x, y]` coordinates of the navigation goal.   | (Internal)                          |

### Action Space

The action space is a `gymnasium.spaces.Box` of shape `(2,)`, representing `[linear_velocity_x, angular_velocity_z]`.

* **Linear Velocity**: `[-0.5, 0.5]` (m/s)
* **Angular Velocity**: `[-1.0, 1.0]` (rad/s)

### Reward Function

The environment uses a **shaped reward function** to provide dense feedback, accelerating learning. The total reward is a sum of the following components:

1.  **Progress Reward**: A positive reward proportional to the reduction in distance to the goal.
2.  **Obstacle Penalty**: A quadratic penalty that activates when the robot is within a predefined `safety_margin` of an obstacle.
3.  **Time Penalty**: A small constant penalty (`-0.1`) at each step to encourage path efficiency.
4.  **Terminal Rewards**: Large bonuses or penalties are given at the end of an episode:
    * **Goal Reached**: `+100`
    * **Collision**: `-100`

### Reset and Recovery Logic

This environment is designed for robust, real-world training. It does **not** rely on unrealistic "teleportation" to reset the robot's position.

1.  **Map-Aware Goal Sampling**: At the start of a new episode, a goal is randomly sampled from a list of known "safe" locations on the map. These locations are guaranteed to be a certain distance away from walls or obstacles.

2.  **Intelligent Trajectory Recovery**: If an episode ends due to a collision or timeout, the environment initiates an autonomous recovery maneuver.
    * A buffer stores the robot's last 15 positions, creating a "breadcrumb trail."
    * The robot uses a built-in controller to navigate backward along this known-safe path to its starting point.
    * This "undo" action allows the robot to physically extricate itself from stuck states, a critical feature for real-world operation.

---

## Rendering

To visualize the agent's performance, you can enable human-readable rendering by setting `render_mode='human'` during environment creation.

```python
env = Ros2RobotEnv(render_mode='human', render_size=(256, 256))
```

The render window displays:
* **Live RGB Feed**: The main view from the robot's camera.
* **LIDAR Overlay**: Laser scan points are projected onto the image (yellow dots), filtered to show ranges between 0.3m and 12m.
* **Info Panel**: A real-time display of the robot's current pose, target pose, wheel velocities, and the last action taken.

---
