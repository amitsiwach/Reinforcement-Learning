import gym
import numpy as np
import time

env = gym.make("CartPole-v1")


def basic_policy(PoleAngle):
    if PoleAngle < 0: # falling right
        return 0 # move right
    return 1 # move left

total_rewards = list()
N_episodes = 200
N_steps = 200
for episode in range(N_episodes):
    rewards = 0
    CartPosition, CartVelocity, PoleAngle, PoleAngluarVelocity = env.reset()
    for step in range(N_steps):
        env.render()
        action = basic_policy(PoleAngle)
        Observation, reward, done, info = env.step(action)
        time.sleep(0.001)
        rewards += reward
        if done:
            break
    total_rewards.append(rewards)

stats = {
    "mean": np.mean(total_rewards),
    "standard deviation": np.std(total_rewards),
    "min": np.min(total_rewards),
    "max": np.max(total_rewards),
}

print(f"Final stats: \n{stats}")
