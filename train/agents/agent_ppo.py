import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from models.ppo_agent_models import ActorCritic

class PPO_Agent:
    """Implements the PPO algorithm."""
    def __init__(self, input_dim, action_dim, device, lr=3e-4, gamma=0.99, k_epochs=80, eps_clip=0.2):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(input_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Action variance
        self.action_var = torch.full((action_dim,), 0.5*0.5).to(device)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_mean, _ = self.policy_old(state_tensor)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)
        
        return action.cpu().numpy().flatten()

    def update(self, memory):
        # Calculate rewards-to-go
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.stack(memory.states, dim=0).detach().to(self.device)
        old_actions = torch.stack(memory.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach().to(self.device)

        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            action_means, state_values = self.policy(old_states)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_means, cov_mat)
            
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())