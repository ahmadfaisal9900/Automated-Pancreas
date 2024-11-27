import gym
from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo

class InsulinEnvironment(gym.Env):
    def __init__(self, config):
        self.predictive_model = config["predictive_model"]
        self.glucose_target_range = (70, 180)  # mg/dL
        self.max_insulin = 10  # maximum insulin dose
        self.current_glucose = 120  # starting glucose level
        self.last_meal_size = 0
        self.time_since_last_dose = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=self.max_insulin, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,))

    def reset(self):
        self.current_glucose = 120
        self.last_meal_size = 0
        self.time_since_last_dose = 0
        return np.array([self.current_glucose, self.last_meal_size, self.time_since_last_dose])

    def step(self, action):
        insulin_dose = action[0]
        
        # Use predictive model to calculate next glucose level
        next_glucose = self.predictive_model.predict(
            self.current_glucose, self.last_meal_size, insulin_dose
        )
        
        # Calculate reward
        if self.glucose_target_range[0] <= next_glucose <= self.glucose_target_range[1]:
            reward = 1
        else:
            reward = -abs(next_glucose - np.mean(self.glucose_target_range))
        
        # Update state
        self.current_glucose = next_glucose
        self.time_since_last_dose = 0 if insulin_dose > 0 else self.time_since_last_dose + 1
        
        # Simulate a new meal randomly
        self.last_meal_size = np.random.randint(0, 100) if np.random.random() < 0.3 else 0
        
        done = False  # You might want to define episode termination conditions
        
        return np.array([self.current_glucose, self.last_meal_size, self.time_since_last_dose]), reward, done, {}

# Initialize Ray
ray.init()

# Configure the environment and RL algorithm
config = {
    "env": InsulinEnvironment,
    "env_config": {
        "predictive_model": 'prescriptive_model.pth'  # Your trained LSTM model
    },
    "framework": "torch",
    "num_workers": 4,
    "train_batch_size": 4000,
}

# Train the agent
analysis = tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 100},
    checkpoint_freq=10,
)

# Get the best trained agent
best_checkpoint = analysis.best_checkpoint
best_config = analysis.best_config