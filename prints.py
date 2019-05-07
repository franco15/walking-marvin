import gym
import numpy as np
from gym import wrappers
from gym import spaces

# env = gym.make('Marvin-v0')

env = gym.make('MountainCar-v0')

print(env.action_space.n)


print (env.reset())
# print (env.step([0, 0, 0, 0]))
print("action space: \n", env.action_space)

print("observation space: \n", env.observation_space)

# print(env.observation_space.low)
# print(env.observation_space.high)

# print(env.observation_space.high - env.observation_space.low)