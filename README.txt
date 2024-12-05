This is to briefly document the files in this folder.

The important files for this work are:
ppo_trading_agent.py - The PPO Agent
hyperparameters_search.py - Grid hyperparameters search
train_gym-trading-env.py - trains the agent and saves the rewards, losses, portfolio returns as numpy arrays
test_gym-trading-env.py - tests on 10 episodes and saves renders logs to render the environment
ppo_test_renderer.py - to visualize the environment and the agent's actions (trading positions)

I also created notebooks for the train and test script.

For the second data.
data_2-test_gym-trading-env.py
data-2_ppo_test_renderer.py

NOTE: The conda_env contains the packages needed to successfully run the scripts. Finally, the script requires GPU to run. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.