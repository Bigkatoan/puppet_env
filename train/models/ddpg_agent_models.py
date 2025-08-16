import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """DDPG Actor Network (Deterministic Policy)"""
    def __init__(self, input_dim=136, action_dim=2, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        # Architecture consistent with project docs: 2 hidden layers of 256 units
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use tanh to bound output to [-1, 1], then scale by max_action
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class Critic(nn.Module):
    """DDPG Critic Network (Q-Function)"""
    def __init__(self, input_dim=136, action_dim=2):
        super(Critic, self).__init__()
        state_action_dim = input_dim + action_dim
        self.fc1 = nn.Linear(state_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)