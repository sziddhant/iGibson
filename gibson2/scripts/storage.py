import gym
import os
import setuptools
import gibson2
import numpy as np
from gibson2.envs.igibson_env import iGibsonEnv
# from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

env = iGibsonEnv(config_file=os.path.join(gibson2.example_config_path, 'fetch.yaml'),
                                  mode='headless')#,
                                #   action_timestep=1.0 / 120.0,
                                #   physics_timestep=1.0 / 120.0)


# It will check your custom environment and output additional warnings if needed
# check_env(env)
wp= env.task.get_shortest_path(env,entire_path=True)[0][1]
env.task.target_pos= np.array([wp[0],wp[1],0])
env.task.target_visual_object_visible_to_agent =True
# env.task.termination_conditions=np.array()
model = PPO("MlpPolicy", env, verbose=2,tensorboard_log="../PPO_fetch/")
model.learn(total_timesteps=250)
model.save("ppo_fetch1")
print("Saved")
# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_fetch1")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()