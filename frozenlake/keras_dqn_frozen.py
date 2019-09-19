import gym
import numpy as np
import random
from collections import deque
import tensorflow.keras as keras

gym.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


def one_hot(state, state_space):
    state_m = np.zeros((1, state_space))
    state_m[0][state] = 1
    return state_m


def experience_replay():
    # Sample minibatch from the memory
    minibatch = random.sample(memory, batch_size)
    # Extract informations from each memory
    for state, action, reward, next_state, done in minibatch:
        # if done, make our target reward
        target = reward
        if not done:
            # predict the future discounted reward
            target = reward + gamma * \
                     np.max(model.predict(next_state))
        # make the agent to approximately map
        # the current state to future discounted reward
        # We'll call that target_f
        target_f = model.predict(state)
        target_f[0][action] = target
        # Train the Neural Net with the state and target_f
        model.fit(state, target_f, epochs=1, verbose=0)


# Hyper-params
gamma = .9
learning_rate = 0.02
episode = 5001
capacity = 64
batch_size = 32

epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

env = gym.make("FrozenLakeNotSlippery-v0")

state_space = env.observation_space.n
action_space = env.action_space.n

model = keras.Sequential([
    keras.layers.Dense(state_space, input_dim=state_space, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(action_space, activation='linear')
])
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

model.summary()

reward_array = []
memory = deque([], maxlen=capacity)
for i in range(episode):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state1_one_hot = one_hot(state, state_space)
        if np.random.uniform() > epsilon:
            action = np.argmax(model.predict(state1_one_hot))
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Training without experience replay
        state2, reward, done, info = env.step(action)
        state2_one_hot = one_hot(state2, state_space)
        target = (reward + gamma * np.max(model.predict(state2_one_hot)))

        target_f = model.predict(state1_one_hot)
        target_f[0][action] = target
        model.fit(state1_one_hot, target_f, epochs=1, verbose=0)
        total_reward += reward

        state = state2

        # Training with experience replay
        # appending to memory
        memory.append((state1_one_hot, action, reward, state2_one_hot, done))

    if i > batch_size:
        experience_replay()

    reward_array.append(total_reward)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)

    if i % 10 == 0 and i != 0:
        print('Episode {} Total Reward: {} Reward Rate {}'.format(i, total_reward, str(sum(reward_array) / i)))
