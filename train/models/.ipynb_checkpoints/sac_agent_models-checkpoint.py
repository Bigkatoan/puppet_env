import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    SAC Actor Network (Policy).
    Architecture based on Table 9[cite: 417, 418].
    """
    def __init__(self, input_dim=136, action_dim=2): # Input is 136-dim fused vector [cite: 411]
        super().__init__()
        # Shared MLP architecture with 2 hidden layers of 256 units [cite: 414]
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Output layers for mean and std of the action distribution [cite: 419]
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        # Apply Tanh to bound actions to [-1, 1] [cite: 420]
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    """
    SAC Critic Network (Q-Function).
    Architecture based on Table 9[cite: 417, 418].
    """
    def __init__(self, input_dim=136, action_dim=2):
        super().__init__()
        # Input to the critic is the state + action
        state_action_dim = input_dim + action_dim
        
        # Shared MLP architecture with 2 hidden layers of 256 units [cite: 414]
        self.fc1 = nn.Linear(state_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # Single linear output unit for the Q-value [cite: 421]
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value