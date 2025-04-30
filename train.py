import gymnasium as gym
from stable_baselines3 import PPO
from custom_environment import ImageExplorationEnv
from PIL import Image
import numpy as np
#from time import sleep


image_path = "N17E073.jpg"
img = Image.open(image_path).resize((256, 256))
map_array = np.array(img, dtype=np.uint8)
env = ImageExplorationEnv(map_array, max_steps=2500, render_mode="human")


#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_scalarfield_tensorboard/")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_scalarfield_tensorboard/", n_steps=512)

model.learn(total_timesteps=120000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    #obs, reward, done, _, _ = vec_env.step(action)
    obs, reward, done, info = vec_env.step(action)
    
    vec_env.render()
    # VecEnv resets automatically
    if done:
      #obs = env.reset()
      obs = vec_env.reset() 

env.close()
model.save('ppo_agent')

