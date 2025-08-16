import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from models.ddpg_agent_models import Actor, Critic

class DDPG_Agent:
    """Implements the DDPG algorithm."""
    def __init__(self, input_dim, action_dim, max_action, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        # Use two critics for stability (a common practice inspired by TD3)
        self.actor = Actor(input_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(input_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(input_dim, action_dim).to(device)
        self.critic_1_target = Critic(input_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(input_dim, action_dim).to(device)
        self.critic_2_target = Critic(input_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)

    def select_action(self, state, exploration_noise=0.1):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add Gaussian noise for exploration
        noise = np.random.normal(0, self.max_action * exploration_noise, size=action.shape)
        return (action + noise).clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(next_state)
            
            # Compute the target Q value
            q1_target = self.critic_1_target(next_state, next_action)
            q2_target = self.critic_2_target(next_state, next_action)
            q_target_min = torch.min(q1_target, q2_target)
            q_target = reward + (1 - done) * self.gamma * q_target_min

        # Get current Q estimates
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)