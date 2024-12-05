import gymnasium as gym
import gym_trading_env
from tqdm import tqdm
import pandas as pd
import numpy as np
from ppo_trading_agent import PPO


def test_model(env, model, n_episodes=10, render_dir="render_logs"):
    """
    Test the model on the environment.
    """
    for episode in tqdm(range(n_episodes)):
        done, truncated = False, False
        observation, info = env.reset()
        while not done and not truncated:
            action, log_prob, value = model.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)

        # Render and save logs at the end of the episode
        env.unwrapped.save_for_render(dir=render_dir)


# Load and preprocess the dataset
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

# Parameters for PPO (ensure these match your training setup)
n_actions = env.action_space.n
input_dims = env.observation_space.shape[0]
chkpt_dir = './models/'  # Path to your model weights
alpha = 0
gamma = 0
policy_clip = 0
n_epochs = 0
gae_lambda = 0
batch_size = 0
# Initialize PPO model
model = PPO(n_actions, input_dims, alpha, gamma, policy_clip, n_epochs, gae_lambda, batch_size, chkpt_dir)
model.load_models()  # Load the trained weights



# Test the model
test_model(env, model, n_episodes=10, render_dir="render_logs")

# Close the environment if necessary
env.close()
