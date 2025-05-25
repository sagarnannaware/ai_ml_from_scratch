"""
Q-Learning Example (Reinforcement Learning)

Description:
Q-learning agent solves an unlabelled 4x4 FrozenLake (no slipping) environment.
There are no labels for actions or states: the agent learns only by reward feedback.

References:
- [Watkins, C. J. C. H. & Dayan, P. (1992). "Q-learning". Machine Learning, 8, 279–292.](https://link.springer.com/article/10.1007/BF00992698)
"""

import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

q_table = np.zeros((n_states, n_actions))

alpha = 0.8    # Learning rate
gamma = 0.95   # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(200):
    state = env.reset()[0]
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

print("Trained Q-table (rounded):\n", np.round(q_table, 2))

# Demonstrate a solution
state = env.reset()[0]
env.render()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = env.step(action)
    env.render()
print("Reward:", reward)
print("The agent never sees state or action labels—only numbers and rewards, learning purely from experience.")