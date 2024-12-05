
import pandas as pd
from ppo_trading_agent import PPO
import numpy as np
import itertools
import os
import gymnasium as gym
import gym_trading_env
from tqdm import tqdm

# Load and preprocess the dataset
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
df.dropna(inplace=True)

# Environment setup
env = gym.make("TradingEnv", name="BTCUSDT", df=df, positions=[-1, 0, 1], trading_fees=0.01/100, borrow_interest_rate=0.0003/100)
env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))
n_actions = env.action_space.n
input_dims = env.observation_space.shape[0]

# Hyperparameters ranges
alphas = [0.0001, 0.00025, 0.0005]
gammas = [0.98, 0.99]
policy_clips = [0.1, 0.2, 0.3]
n_epochss = [3, 4, 5]
gae_lambdas = [0.92, 0.95, 0.98]
batch_sizes = [32, 64, 128]
chkpt_dir = './models/'
n_episodes = 10
# Grid search
best_score = -np.inf
best_params = {}

for alpha, gamma, policy_clip, n_epochs, gae_lambda, batch_size in itertools.product(alphas, gammas, policy_clips, n_epochss, gae_lambdas, batch_sizes):
    print(f"Training with alpha: {alpha}, gamma: {gamma}, policy_clip: {policy_clip}, n_epochs: {n_epochs}, gae_lambda: {gae_lambda}, batch_size: {batch_size}")
    ppo_agent = PPO(n_actions, input_dims, alpha, gamma, policy_clip, n_epochs, gae_lambda, batch_size, chkpt_dir)
    scores = []

    for i in tqdm(range(n_episodes)):
        # observation, info = env.reset()
        score = 0
        done, truncated = False, False
        observation, info = env.reset()
        while not done and not truncated:
            action, log_prob, value = ppo_agent.choose_action(observation)
            
            # Validate the action - make sure it's within a valid range
            if not env.action_space.contains(action):
                print(f"Invalid action: {action}")
                break

            observation_, reward, done, truncated, info = env.step(action)
            score += reward
        scores.append(score)
        ppo_agent.learn()

    avg_score = np.mean(scores[-10:])  # Consider the last 10 episodes
    if avg_score > best_score:
        best_score = avg_score
        best_params = {'alpha': alpha, 'gamma': gamma, 'policy_clip': policy_clip, 'n_epochs': n_epochs, 'gae_lambda': gae_lambda, 'batch_size': batch_size}
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("Best parameters:", best_params)
