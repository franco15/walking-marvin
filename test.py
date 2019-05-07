import numpy as np
import gym
import matplotlib.pyplot as plt
import random

env = gym.make('BipedalWalker-v2')
env.reset()

# for i in range(10):
#     obs = env.reset()
#     while True:
#         env.render()
#         ob, reward, done, _ = env.step([0, 1, 2, 3])
#         if done:
#             env.close()
#             break



q_table = np.zeros([24, 4])

# hyperparameters
alpha = 0.1 # learning rate
gamma = 0.6 # discount rate
epsilon = 0.05 #

# for plotting metrics
all_epochs = []
all_penalties = []

for i in range(20):
    state = env.reset()
    print(state)
    print(q_table)

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        env.render()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        
        next_state, reward, done, info = env.step(action)

        old_value = q_table(state, action)
        next_max = np.max(q_table[next_state])

        # q learning equation
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1

    if i % 100 == 0:
        env.close()
        print(f"Episode: {i}")

print('training done')



# def QLearning(env, learning, discount, epsilon, min_eps, episodes):
#     num_states = (env.observation_space.high - env.observation_space.low) *\
#         np.array([10, 100, 1000, 10000, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12, 10**13,
#         10**14, 10**14, 10**14, 10**14, 10**14, 10**14, 10**14, 10**14, 10**14, 10**14])

#     num_states = np.round(num_states, 0).astype(int) + 1

#     Q = np.random.uniform(low = -1, high = 1, size = (num_states[0], num_states[1], num_states[2], num_states[3],
#         num_states[4], num_states[5], num_states[6], num_states[7], num_states[8], num_states[9], num_states[10],
#         num_states[11], num_states[12], num_states[13], num_states[14], num_states[15], num_states[16],
#         num_states[17], num_states[18], num_states[19], num_states[20], num_states[21], num_states[22], num_states[23], env.action_space.n))