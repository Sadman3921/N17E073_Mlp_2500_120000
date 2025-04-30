from stable_baselines3 import PPO
from custom_environment import ImageExplorationEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from PIL import Image
import numpy as np

image_path = "N17E073.jpg"
img = Image.open(image_path).resize((256, 256))
map_array = np.array(img, dtype=np.uint8)

# Wrap your environment
env = DummyVecEnv([lambda: ImageExplorationEnv(map_array, max_steps=2500, render_mode="human")])

# Load the trained agent
model = PPO.load("ppo_agent.zip")


# Evaluate it
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Run it

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

    
obs = env.reset()
episode = 0
max_episodes = 10

while episode < max_episodes:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
    if dones:
        episode += 1
        obs = env.reset()
