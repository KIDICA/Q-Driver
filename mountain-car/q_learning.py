import gym
import pickle
from collections import defaultdict
import numpy as np

Q = defaultdict(lambda: [0, 0, 0])

env = gym.make('MountainCar-v0')


def transform_state(state):
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)


lr, factor = 0.7, 0.95
episodes = 10000
score_list = []
for i in range(episodes):
    s = transform_state(env.reset())
    score = 0
    while True:
        a = np.argmax(Q[s])
        if np.random.random() > i * 3 / episodes:
            a = np.random.choice([0, 1, 2])
        next_s, reward, done, _ = env.step(a)
        env.render()
        next_s = transform_state(next_s)
        Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
env.close()

with open('MountainCar-v0-q-learning.pickle', 'wb') as f:
    pickle.dump(dict(Q), f)
    print('model saved')
