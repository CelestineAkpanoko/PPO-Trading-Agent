
import pandas as pd
import numpy as np
from ppo_trading_agent import PPO
from utils import plot_performance
import os
import gymnasium as gym
import gym_trading_env
from tqdm import tqdm
import time

# Load and preprocess the dataset
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
df.dropna(inplace=True)

# Environment setup
env = gym.make("TradingEnv",
               name="BTCUSDT",
               df=df,  # Your dataset with your custom features
               positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
               trading_fees=0.01/100,  # 0.01% per stock buy / sell
               borrow_interest_rate=0.0003/100)  # 0.0003% per timestep

env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

# PPO Agent setup
n_episodes = 200
n_actions = env.action_space.n
input_dims = env.observation_space.shape[0]
alpha = 0.0003
gamma = 0.99
policy_clip = 0.2
n_epochs = 4
gae_lambda = 0.95
batch_size = 64
chkpt_dir = './models/'
figure_file = './plots/performance.png'

if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
if not os.path.exists(os.path.dirname(figure_file)):
    os.makedirs(os.path.dirname(figure_file))

ppo_agent = PPO(n_actions, input_dims, alpha, gamma, policy_clip, n_epochs, gae_lambda, batch_size, chkpt_dir)

# Define thresholds for stopping criteria
portfolio_return_threshold = 0.95  # 95% portfolio return
portfolio_returns = []
# Initialize lists to store metrics
rewards = []
total_losses = []
portfolio_returns = []

# Start the timer
start_time = time.time()

for i in tqdm(range(n_episodes)):
    score = 0
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        action, log_prob, value = ppo_agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        ppo_agent.store_transition(observation, action, log_prob, value, reward, done)
        observation = observation_
        score += reward

    rewards.append(score)
    # Retrieve and store the "Portfolio Return" metric
    metrics = env.unwrapped.get_metrics()
    portfolio_return = metrics.get('Portfolio Return', None)
    if portfolio_return is not None:
        portfolio_returns.append(portfolio_return)
    else:
        print(f"Portfolio Return not found for episode {i}")

    # Convert to numpy array
    portfolio_returns_np = np.array([float(val.strip('%')) for val in portfolio_returns])
    actor_loss, critic_loss = ppo_agent.learn()
    total_loss = actor_loss + critic_loss
    total_losses.append(total_loss)

    
    
    # Print metrics every 10 episodes
    if (i + 1) % 10 == 0:
        avg_total_loss = np.mean(total_losses[max(0, i-9):i+1])  # Average total loss over the last 10 episodes
        print(f'Episode {i+1}, Score: {score}, Portfolio Return: {portfolio_returns[-1]}, Avg Total Loss: {avg_total_loss:.4f}')

    portfolio_returns_np = np.array([float(val.strip('%')) for val in portfolio_returns])
    # Check stopping criteria over the last 50 episodes
    if i >= 50:
        avg_portfolio_return = np.mean(portfolio_returns_np[-20:])
        if avg_portfolio_return > portfolio_return_threshold:
            print("Stopping criteria met: Market Return and Portfolio Return thresholds reached.")
            break

# End the timer
end_time = time.time()

# Calculate total training time
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")
# Assuming portfolio_returns is a list of strings with percent symbols
portfolio_returns_np = np.array([float(val.strip('%')) for val in portfolio_returns])
np.save('portfolio_returns.npy', portfolio_returns_np)
# Save rewards and losses as numpy files
np.save('rewards.npy', np.array(rewards))
np.save('total_losses.npy', np.array(total_losses))
ppo_agent.save_models()
x = [i+1 for i in range(len(rewards))]
plot_performance(x, rewards, total_losses, portfolio_returns_np, figure_file)