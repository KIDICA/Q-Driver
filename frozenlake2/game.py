import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.reset()

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

Q = np.zeros([NUM_STATES, NUM_ACTIONS])  # You could also make this dynamic if you don't know all games states upfront

gamma = 0.95  # discount factor
alpha = 0.01  # learning rate
epsilon = 0.5  # exploration

for episode in range(1, 500001):
    done = False
    obs = env.reset()
    while not done:
        if np.random.rand(1) < epsilon:
            # exploration with a new option with probability epsilon, the epsilon greedy approach
            action = env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(Q[obs])
        obs2, rew, done, info = env.step(action)  # take the action
        Q[obs, action] += alpha * (
                rew + gamma * np.max(Q[obs2]) - Q[obs, action])  # Update Q-marix using Bellman equation
        obs = obs2

    if episode % 5000 == 0:
        # report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved
        rew_average = 0.
        for i in range(100):
            obs = env.reset()
            done = False
            while not done:
                action = np.argmax(Q[obs])
                obs, rew, done, info = env.step(action)  # take step using selected action
                rew_average += rew
        rew_average = rew_average / 100
        print('Episode {} avarage reward: {}'.format(episode, rew_average))

        if rew_average > 0.8:
            # FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.
            # Test it on 0.8 so it is not a one-off lucky shot solving it
            print("Frozen lake solved")
            break
