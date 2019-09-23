import gym
import numpy as np
import random
from collections import deque
from tensorflow import keras

from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
)

# 1. Parameters of Q-leanring
gamma = .9
learning_rate = 0.002
episode = 5001
capacity = 64
batch_size = 32

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

env = gym.make("FrozenLakeNotSlippery-v0")

state_space = env.observation_space.n
action_space = env.action_space.n


def one_hot(index, size=state_space):
    v = np.zeros((1, size))
    v[0][index] = 1
    return v


def experience_replay():
    # Sample minibatch from the memory
    minibatch = random.sample(memory, batch_size)
    # Extract informations from each memory
    for state, action, reward, next_state, done in minibatch:
        # if done, make our target reward
        target = reward
        if not done:
            # predict the future discounted reward
            target = reward + gamma * np.max(model.predict(next_state))
        # make the agent to approximately map
        # the current state to future discounted reward
        # We'll call that target_f
        target_f = model.predict(state)
        target_f[0][action] = target
        # Train the Neural Net with the state and target_f
        model.fit(state, target_f, epochs=1, verbose=0)


# Neural network model for DQN
model = keras.models.Sequential([
    keras.layers.Dense(state_space, input_dim=state_space, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(action_space, activation='linear')
])

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

reward_array = []
memory = deque([], maxlen=capacity)

for i in range(1, episode):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        v_state = one_hot(state)
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = np.random.uniform()

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(model.predict(v_state))
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Training without experience replay
        next_state, reward, done, info = env.step(action)
        v_next_state = one_hot(next_state)
        target = (reward + gamma * np.max(model.predict(v_next_state)))

        target_f = model.predict(v_state)
        target_f[0][action] = target
        model.fit(v_state, target_f, epochs=1, verbose=0)
        total_reward += reward

        state = next_state

        # Training with experience replay buffer
        memory.append((v_state, action, reward, v_next_state, done))
    if i > batch_size:
        experience_replay()

    reward_array.append(total_reward)

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)

    reward_rate = sum(reward_array) / i

    if i % 10 == 0 and i != 0:
        print('Episode {} Total Reward: {} Reward Rate {}'.format(i, total_reward, str(reward_rate)))

done = False
steps = 0
state = env.reset()
env.render()
while not done:
    steps += 1
    v_state = one_hot(state)
    action = np.argmax(model.predict(v_state))
    s, r, done, _ = env.step(action)
    print(f"Steps={steps}")
    env.render()
    # time.sleep(1)
