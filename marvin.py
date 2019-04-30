import gym
import numpy as np
from gym import wrappers

# print(env.action_space)
# print(env.observation_space)

env = gym.make('Marvin-v0')

best_length = 0
episode_lengths = []

best_weigths = np.zeros(24)

for i in range(100):
    new_weigths = np.random.uniform(-2.0, 2.0, 24)

    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        cnt = 0
        while not done:
            # env.render()
            cnt += 1
            action = 0
            dot_product = np.dot(observation, new_weigths)
            print(dot_product)
            if dot_product < -1.0:
                action = 0
            elif dot_product >= -1.0 and dot_product < 0:
                action = 1
            elif dot_product >= 0 and dot_product < 1.0:
                action = 2
            elif dot_product >= 1.0 and dot_product < 2.0:
                action = 3
            
            print(action)
            observation, reward, done, info = env.step(action)

            print(reward)
            if done:
                break
        length.append(cnt)
    average_length = float(sum(length) / len(length))

    if average_length > best_length:
        best_length = average_length
        best_weigths = new_weigths
    episode_lengths.append(average_length)
    if i % 10 == 0:
        print("best length is ", best_length)

done = False
cnt = 0
env = wrappers.Monitor(env, 'MovieFiles2', force=True)
observation = env.reset()

while not done:
    cnt += 1
    action = 0

    dot_product = np.dot(observation, best_weigths)
    if dot_product < -1.0:
        action = 0
    elif dot_product >= -1.0 and dot_product < 0:
        action = 1
    elif dot_product >= 0 and dot_product < 1.0:
        action = 2
    elif dot_product >= 1.0 and dot_product < 2.0:
        action = 3
    observation, reward, done, info = env.step(action)

    if done:
        break

print("with best weigths, done after", cnt, "moves")
# env.close()
