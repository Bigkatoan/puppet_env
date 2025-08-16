import torch
import argparse
import os
import numpy as np
import rclpy
import time
from datetime import datetime

# Import your custom ROS2 environment
from puppet import Ros2RobotEnv

# Import perception and agent models
from models.perception_models import EdgeAutoencoder, PosePredictionNetwork
from agents.sac_agent import SAC_Agent
from agents.agent_ddpg import DDPG_Agent
from agents.agent_ppo import PPO_Agent

# Import memory buffers and logger
from utils.buffers import ReplayBuffer, PPO_Memory
from utils.logger import Logger

def create_fused_state(latent_vec, wheel_vel, pred_pose, target_pose):
    """
    Concatenates all processed inputs into the 136-dim state vector for the RL agent,
    as specified in Section 4.3.1 of the documentation.
    
    Args:
        latent_vec (np.ndarray): 128-dim vector from EdgeAutoencoder.
        wheel_vel (np.ndarray): 4-dim vector of wheel velocities.
        pred_pose (np.ndarray): Vector of predicted pose from PosePredictionNetwork.
        target_pose (np.ndarray): 2-dim vector of the goal coordinates.
    
    Returns:
        np.ndarray: The 136-dimensional fused state vector.
    """
    # Per docs, slice the predicted pose to 2 dimensions [x, y] for the state
    pred_pose_2d = pred_pose.flatten()[:2]
    
    return np.concatenate([
        latent_vec.flatten(),
        wheel_vel.flatten(),
        pred_pose_2d,
        target_pose.flatten()
    ])

def train_rl_phase(args, device):
    """
    Handles the main reinforcement learning training loop (Phase 2).
    """
    print(f"--- Starting Phase 2: RL Training with {args.agent.upper()} ---")
    print(f"Using device: {device}")

    # 1. Load Pre-trained Perception Models (from Phase 1)
    print("Loading pre-trained perception models...")
    edge_ae = EdgeAutoencoder().to(device)
    edge_ae.load_state_dict(torch.load(args.ae_weights, map_location=device))
    edge_ae.eval()  # Set to evaluation mode (no gradient updates)
    
    pose_net = PosePredictionNetwork().to(device)
    pose_net.load_state_dict(torch.load(args.pose_weights, map_location=device))
    pose_net.eval()
    print("Perception models loaded successfully.")
    
    # 2. Initialize Environment and Logger
    env = Ros2RobotEnv()
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(log_dir="logs", agent_name=f"{args.agent.upper()}_{start_time_str}", env_name="Ros2RobotEnv")
    model_dir = os.path.join("trained_models", f"{args.agent.upper()}_{start_time_str}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 3. Initialize Agent and Memory Buffer
    input_dim = 136  # As defined in Section 4.3.1
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = None
    replay_buffer = None
    ppo_memory = None
    is_on_policy = False

    if args.agent == 'sac':
        agent = SAC_Agent(input_dim, action_dim, device, lr=args.lr)
        replay_buffer = ReplayBuffer(input_dim, action_dim, max_size=1_000_000) # Size from Table 10
        print("SAC Agent initialized.")
    elif args.agent == 'ddpg':
        agent = DDPG_Agent(input_dim, action_dim, max_action, device, lr=args.lr)
        replay_buffer = ReplayBuffer(input_dim, action_dim, max_size=1_000_000)
        print("DDPG Agent initialized.")
    elif args.agent == 'ppo':
        agent = PPO_Agent(input_dim, action_dim, device, lr=args.lr, k_epochs=args.ppo_epochs)
        ppo_memory = PPO_Memory()
        is_on_policy = True
        print("PPO Agent initialized.")
    else:
        raise ValueError("Invalid agent specified.")

    # 4. Main Autonomous Training Loop (Algorithm 2)
    total_timesteps = 0
    print("\nStarting training loop...")
    for episode in range(1, args.max_episodes + 1):
        obs_dict, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        for step in range(args.max_steps_per_episode):
            total_timesteps += 1
            episode_steps += 1
            
            # --- Perception: Process raw observations to create fused state ---
            with torch.no_grad():
                # Process image through perception models
                img_tensor = torch.FloatTensor(obs_dict['rgb']).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                latent_vec = edge_ae.encoder(img_tensor).cpu().numpy()
                pred_pose = pose_net(torch.from_numpy(latent_vec).to(device)).cpu().numpy()
            
            state = create_fused_state(latent_vec, obs_dict['wheel_velocities'], pred_pose, obs_dict['target_pose'])
            
            # --- Agent-Environment Interaction ---
            if is_on_policy:
                action = agent.select_action(state, ppo_memory)
            else:
                # Add exploration noise for DDPG, SAC handles exploration via entropy
                if args.agent == 'ddpg':
                    action = agent.select_action(state, exploration_noise=args.expl_noise)
                else: # SAC
                    action = agent.select_action(state)
            
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # --- Process next state ---
            with torch.no_grad():
                next_img_tensor = torch.FloatTensor(next_obs_dict['rgb']).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                next_latent_vec = edge_ae.encoder(next_img_tensor).cpu().numpy()
                next_pred_pose = pose_net(torch.from_numpy(next_latent_vec).to(device)).cpu().numpy()
            
            next_state = create_fused_state(next_latent_vec, next_obs_dict['wheel_velocities'], next_pred_pose, next_obs_dict['target_pose'])
            
            # --- Store transition and Train ---
            if is_on_policy:
                ppo_memory.rewards.append(reward)
                ppo_memory.dones.append(done)
                # PPO updates after collecting a fixed number of steps
                if len(ppo_memory.states) == args.ppo_update_steps:
                    agent.update(ppo_memory)
                    ppo_memory.clear()
            else: # Off-policy (SAC/DDPG)
                replay_buffer.add(state, action, reward, next_state, done)
                # Start training after collecting enough samples (learning_starts from docs)
                if total_timesteps > args.learning_starts:
                    agent.train(replay_buffer, args.batch_size)
            
            obs_dict = next_obs_dict
            if done:
                break
        
        # --- End of Episode Logging ---
        print(f"Episode {episode} | Timesteps: {episode_steps} | Reward: {episode_reward:.2f}")
        # Add logging logic here if needed

        # --- Save Model Periodically ---
        if episode % args.save_freq == 0:
            torch.save(agent.policy.state_dict(), os.path.join(model_dir, f"{args.agent}_policy_epi{episode}.pth"))
            print(f"--- Model saved at episode {episode} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 2: RL Training for End-to-End Autonomous Navigation.")
    # General arguments
    parser.add_argument("--agent", type=str.lower, default="sac", choices=["sac", "ddpg", "ppo"], help="The RL algorithm to use for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for all optimizers (from Table 10).")
    parser.add_argument("--max_episodes", type=int, default=5000, help="Total number of episodes to train.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Maximum timesteps per episode (from Table 3).")
    parser.add_argument("--save_freq", type=int, default=100, help="Frequency (in episodes) to save the model.")
    # Perception model weights
    parser.add_argument("--ae_weights", type=str, default="edge_autoencoder.pth", help="Path to pre-trained Edge-AE weights.")
    parser.add_argument("--pose_weights", type=str, default="pose_network.pth", help="Path to pre-trained PoseNet weights.")
    # Off-policy arguments (SAC/DDPG)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for off-policy training (from Table 10).")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Timesteps before off-policy learning starts.")
    parser.add_argument("--expl_noise", type=float, default=0.1, help="Std of Gaussian exploration noise for DDPG.")
    # On-policy arguments (PPO)
    parser.add_argument("--ppo_update_steps", type=int, default=2048, help="Number of steps to collect before each PPO update.")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of optimization epochs per PPO update.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ROS2
    rclpy.init(args=None)
    try:
        train_rl_phase(args, device)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Ensure ROS2 is shut down properly
        if rclpy.ok():
            print("Shutting down ROS2 node.")
            rclpy.shutdown()