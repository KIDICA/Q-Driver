import numpy as np
import random
from custom_frozenlake import FrozenLake
import seaborn as sns
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class QLearner:
    """Learns with epsilon-greedy algorithm."""

    def __init__(self, mdp: FrozenLake):
        # Hyper parameters
        self.SNAPSHOT_STATES = 2000
        self.EPISODES = 50000
        self.LEARNING_RATE = 0.8  # alpha
        self.DISCOUNT_FACTOR = 0.95  # gamma, decays with path length.
        self.EPSILON = 0.9
        self.mdp = mdp
        self.Q = None
        self.state_visits = []
        self.state_counter = None
        self.size = None
        self.steps = 0
        self.max_visit = 0
        self.reset()

    def update_hyperparams(self, alpha, gamma, epsilon):
        self.LEARNING_RATE = alpha
        self.DISCOUNT_FACTOR = gamma
        self.EPSILON = epsilon

    def epsilon_greedy(self, Q, s):
        """Either return a random action or the one with the most promising result in situation s from Q."""
        if random.uniform(0, 1) < self.EPSILON:
            return self.mdp.action_sample()
        # Array index with highest value.
        return np.argmax(Q[s, :])

    def reset(self):
        # Q = state space x action space -> reward (r element_of R is a any number).
        # Q: S x A -> R, initialize with zeros.
        self.Q = np.zeros([self.mdp.state_count(), self.mdp.action_count()], dtype=np.float32)
        # Equal probability for all actions. Non-zero values converge faster.
        for s in range(len(self.Q)):
            for a in range(len(self.Q[s, :])):
                self.Q[s, a] = 0.5
        self.state_visits = []
        self.state_counter = np.zeros(self.mdp.state_count(), dtype=np.uint64)
        self.size = int(math.sqrt(self.mdp.state_count()))
        self.steps = 0
        self.max_visit = 0

    def state_counter_snapshot(self):
        self.state_visits.append(np.reshape(self.state_counter.copy(), (self.size, self.size)))

    def train(self, callback=None):
        self.reset()

        for i in range(self.EPISODES):
            s = self.mdp.reset()
            done = False
            rewards = 0
            if i % self.SNAPSHOT_STATES == 0:
                print(f"Episode {i},e={self.EPSILON}")

            while not done:
                if self.steps % self.SNAPSHOT_STATES == 0:
                    self.state_counter_snapshot()

                self.state_counter[s] += 1
                # for statistics, i.e. heatmap
                self.max_visit = max(self.state_counter[s], self.max_visit)
                a = self.epsilon_greedy(self.Q, s)
                s_next, r, done, _ = self.mdp.step(a)
                next_q = r + self.DISCOUNT_FACTOR * np.max(self.Q[s_next, :])
                self.Q[s, a] = ((1 - self.LEARNING_RATE) * self.Q[s, a]) + (self.LEARNING_RATE * next_q)
                s = s_next
                rewards += r
                self.steps += 1
                if callback is not None:
                    do_continue = callback(self.mdp.move[a], s, self.Q[s, :], self.steps, rewards)
                    if not do_continue:
                        return
                    time.sleep(1 / 4)

            if i % self.SNAPSHOT_STATES == 0:
                # Snapshot current counter for animation later
                self.state_counter_snapshot()

        self.state_counter_snapshot()

    def save(self):
        filename = f"frozenlake_q_table"
        np.save(filename, self.Q)
        print(repr(self.Q))


class Drawer:
    def __init__(self, q: QLearner):
        # Norm to 1.0
        print('steps', q.steps)
        q.state_visits = q.state_visits / q.max_visit
        q.state_visits[-1][-1][-1] = -1
        self.q = q
        self.size = len(q.state_visits)
        sns.set()
        # Generate heat map.
        self.fig = plt.figure()
        self.cmap = 'magma'

    def init(self):
        #values = self.q.state_visits[-1]
        # print(values)
        #size = int(math.sqrt(self.q.mdp.state_count()))
        # labels = np.array([[round(values[j][i],3) for i in range(size)] for j in range(size)])
        # print('labels', labels)
        ax = sns.heatmap(self.q.state_visits[-1], cmap=self.cmap, xticklabels=False, yticklabels=False, square=True, annot=True, vmin=0.0, vmax=1.0, annot_kws={"size": 6}, linewidths=1, cbar=True, linecolor='white')
        ax.set_title(f"Value-Iteration: ε={round(self.q.EPSILON, 2)}, γ={round(self.q.DISCOUNT_FACTOR, 2)}, α={round(self.q.LEARNING_RATE, 2)}, steps={self.q.steps}")

    def animate(self, i):
        print("i:", i, self.size)
        data = self.q.state_visits[i]
        sns.heatmap(data, cmap=self.cmap, xticklabels=False, yticklabels=False, square=True, annot=False, vmin=0.0, vmax=1.0, linewidths=1, cbar=False, linecolor='white')

    def save(self):
        fps = 15
        intervals = int(1 / fps * 1000)
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, cache_frame_data=True, frames=self.size, repeat=False, interval=intervals)
        anim.save(f"q_learn_epis_{self.q.EPISODES}_snap_{self.q.SNAPSHOT_STATES}_eps_{self.q.EPSILON}_gamma_{self.q.DISCOUNT_FACTOR}_alpha_{self.q.LEARNING_RATE}.mp4", writer='ffmpeg', bitrate=5000, dpi=600)
