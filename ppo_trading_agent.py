import torch
import torch.nn as nn
import torch.optim as optim
from utils import ActorNetwork, CriticNetwork, PPOMemory
import os
import numpy as np
# from tqdm import tqdm
from tqdm.auto import tqdm

class PPO:
    def __init__(self, n_actions, input_dims, alpha, gamma, policy_clip, n_epochs, gae_lambda, batch_size, chkpt_dir):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        # self.input_dims = input_dims if isinstance(input_dims, (list, tuple)) else [input_dims]

        self.actor = ActorNetwork(n_actions)
        self.critic = CriticNetwork(input_dims)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        self.memory = PPOMemory(batch_size)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        torch.save(self.actor.state_dict(), os.path.join(self.chkpt_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.chkpt_dir, 'critic.pth'))

    def load_models(self):
        self.actor.load_state_dict(torch.load(os.path.join(self.chkpt_dir, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(self.chkpt_dir, 'critic.pth')))


    def choose_action(self, observation):
        # print("choosing actions")
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Reshape the observation to match the input shape of the network
        observation = observation.reshape(1, -1)



        # Convert the numpy array to a PyTorch tensor
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        # print(action)
        # print(action.item())
        return action.item(), log_prob.item(), value.item()


    def learn(self):
        print("Learning...")
        actor_losses, critic_losses = [], []
        for _ in tqdm(range(self.n_epochs), desc="Training Epochs"):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            # Convert to tensors and move to device
            values = torch.tensor(vals_arr, dtype=torch.float32).to(self.device)
            reward_arr = torch.tensor(reward_arr, dtype=torch.float32).to(self.device)
            dones_arr = torch.tensor(dones_arr, dtype=torch.float32).to(self.device)

            # Vectorized advantage calculation
            advantage = torch.zeros(len(reward_arr), dtype=torch.float32).to(self.device)
            for t in reversed(range(len(reward_arr) - 1)):
                delta = reward_arr[t] + self.gamma * values[t + 1] * (1 - dones_arr[t]) - values[t]
                advantage[t] = delta + self.gamma * self.gae_lambda * advantage[t + 1]

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.device)

                # Actor and critic updates
                probs = self.actor(states)
                dist = torch.distributions.Categorical(probs)
                new_probs = dist.log_prob(actions)

                critic_value = self.critic(states).squeeze()
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs = clipped_probs * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = nn.MSELoss()(critic_value, returns)

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Optimize critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.memory.clear_memory()
        return np.mean(actor_losses), np.mean(critic_losses)
