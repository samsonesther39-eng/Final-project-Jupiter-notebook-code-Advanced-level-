import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

q_table = np.random.uniform(0, 1, size=(10, 10, 10, 10, 2))

def discretize_state(state):
    cart_pos = int(np.digitize(state[0], np.linspace(-2.4, 2.4, 10)) - 1)
    cart_vel = int(np.digitize(state[1], np.linspace(-1, 1, 10)) - 1)
    pole_angle = int(np.digitize(state[2], np.linspace(-0.209, 0.209, 10)) - 1)
    pole_vel = int(np.digitize(state[3], np.linspace(-1, 1, 10)) - 1)
    cart_pos = max(0, min(cart_pos, 9))
    cart_vel = max(0, min(cart_vel, 9))
    pole_angle = max(0, min(pole_angle, 9))
    pole_vel = max(0, min(pole_vel, 9))
    return cart_pos, cart_vel, pole_angle, pole_vel

alpha = 0.1
gamma = 0.9

for episode in range(1000):
    observation, info = env.reset()
    done = False
    rewards = 0
    while not done:
        state = discretize_state(observation)
        action = np.argmax(q_table[state[0], state[1], state[2], state[3]])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards += reward
        new_state = discretize_state(observation)
        q_table[state[0], state[1], state[2], state[3], action] = (1 - alpha) * q_table[state[0], state[1], state[2], state[3], action] + alpha * (reward + gamma * np.max(q_table[new_state[0], new_state[1], new_state[2], new_state[3]]))
    print(f'Episode: {episode+1}, Rewards: {rewards}')

env.close()