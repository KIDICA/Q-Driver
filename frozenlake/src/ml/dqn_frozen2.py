import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tqdm import tqdm, trange
import pandas as pd
import tensorflow as tf

STATE_COUNT = 8 ** 2


def loss(Qtarget, Q):
    return K.sum(K.square(Qtarget - Q))


from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=100,
)


def frozen_lake(env, e, learning_rate, gamma, episodes, steps):
    # Initialize history memory
    step_list = []
    reward_list = []
    loss_list = []
    e_list = []

    initializer = tf.random_uniform_initializer(0, 0.1, seed=1)
    model = Sequential([Dense(4, input_dim=env.observation_space.n,
                              kernel_initializer=initializer,
                              use_bias=False
                              )])
    model.compile(loss=loss, optimizer=Adam(lr=learning_rate))

    def one_hot(size, index):
        return np.identity(size)[index].reshape(1, size)

    for i in range(episodes):
        state = env.reset()
        reward_all = 0

        s = 0  # Step counter
        l = 0  # Loss

        for s in range(steps):
            # Choose action randomly or through agent model
            if np.random.rand(1) < e:
                Q = model.predict(one_hot(STATE_COUNT, state), batch_size=1)
                action = env.action_space.sample()
            else:
                Q = model.predict(one_hot(STATE_COUNT, state), batch_size=1)
                action = np.argmax(Q)

            new_state, reward, done, _ = env.step(action)
            # env.render()

            # Adjust reward if done without reaching end
            if done and reward == 0.0:
                reward = -1

            # Find max-Q for future state
            Q1 = model.predict(one_hot(STATE_COUNT, new_state), batch_size=1)
            maxQ1 = np.max(Q1)

            # Bellman Equation
            # Update target for training by adding reward for action and discounted max next state Q-value
            targetQ = Q
            targetQ[0, action] = reward + (gamma * maxQ1)

            # Train on target Q value
            history = model.fit(one_hot(STATE_COUNT, state), targetQ, verbose=False, batch_size=1)

            # Update history and set current state
            l += history.history['loss'][0]
            reward_all += reward
            state = new_state

            if done:
                if reward_all > 0:
                    print(i, episodes, reward_all)
                # Reduce eps if current episode is successful
                if reward > 0:
                    e = 1. / ((i / 50) + 10)
                break

        # Update history
        step_list.append(s)
        reward_list.append(reward_all)
        loss_list.append(l / s)
        e_list.append(e)
    print('\nSuccessful episodes: {}'.format(np.sum(np.array(reward_list) > 0.0) / episodes))

    window = int(episodes / 10)

    plt.figure(figsize=[9, STATE_COUNT])
    plt.subplot(411)
    plt.plot(pd.Series(step_list).rolling(window).mean())
    plt.title('Step Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Moves')
    plt.xlabel('Episode')

    plt.subplot(412)
    plt.plot(pd.Series(reward_list).rolling(window).mean())
    plt.title('Reward Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Reward')
    plt.xlabel('Episode')

    plt.subplot(413)
    plt.plot(pd.Series(loss_list).rolling(window).mean())
    plt.title('Loss Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Loss')
    plt.xlabel('Episode')

    plt.subplot(414)
    plt.plot(e_list)
    plt.title('Random Action Parameter')
    plt.ylabel('Chance Random Action')
    plt.xlabel('Episode')

    plt.tight_layout(pad=2)
    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLakeNotSlippery-v0')
    # Chance of random action
    e = 0.9
    learning_rate = 0.01
    # Discount Rate
    gamma = 0.99
    # Training Episodes
    episodes = 500
    # Max Steps per episode
    steps = 50

    frozen_lake(env, e, learning_rate, gamma, episodes, steps)
