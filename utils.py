import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(7, fc1_dims)  # Adjusted to match observation size
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q



class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


def plot_performance(x, rewards, total_losses, portfolio_returns, figure_file):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Plotting Rewards
    axs[0].plot(x, rewards, label='Rewards')
    axs[0].set_title('Rewards Over Episodes')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Rewards')
    axs[0].legend()

    # Plotting Total Losses
    axs[1].plot(x, total_losses, label='Total Loss')
    axs[1].set_title('Total Loss Over Episodes')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Plotting Portfolio Returns
    axs[2].plot(x, portfolio_returns, label='Portfolio Return')
    axs[2].set_title('Portfolio Returns Over Episodes')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Portfolio Return')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(figure_file)
    plt.show()

# Additional utility functions can be added here if needed

