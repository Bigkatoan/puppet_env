import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from models.sac_agent_models import Actor, Critic

class SAC_Agent:
    """
    Implements the Soft Actor-Critic algorithm as per the documentation.
    """
    def __init__(self, input_dim, action_dim, device, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma    # Discount factor from Table 10 [cite: 445]
        self.tau = tau
        self.alpha = alpha

        # Use two critics to mitigate Q-value overestimation [cite: 409]
        self.actor = Actor(input_dim, action_dim).to(device)
        self.critic_1 = Critic(input_dim, action_dim).to(device)
        self.critic_2 = Critic(input_dim, action_dim).to(device)
        
        self.target_critic_1 = Critic(input_dim, action_dim).to(device)
        self.target_critic_2 = Critic(input_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers (Learning rate from Table 10 [cite: 445])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        # Automated temperature tuning (Section 2.3.3) [cite: 175]
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state_tensor)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size):
        # Sample a minibatch from the replay buffer [cite: 192]
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # --- Update Critic Networks (Eq. 7) [cite: 164] ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target_next = self.target_critic_1(next_state, next_action)
            q2_target_next = self.target_critic_2(next_state, next_action)
            min_q_target = torch.min(q1_target_next, q2_target_next)
            # Soft Bellman backup (Eq. 8) [cite: 168]
            soft_q_target = min_q_target - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * soft_q_target

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # --- Update Actor and Alpha (Eq. 9 & 10) [cite: 173, 176] ---
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Policy loss
        actor_loss = (self.alpha * log_prob - min_q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (temperature) loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # --- Soft Update Target Networks [cite: 214] ---
        with torch.no_grad():
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)