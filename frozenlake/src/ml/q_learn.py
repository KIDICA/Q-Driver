import random
from env.custom_frozenlake import FrozenLake
import math
import time
import numpy as np


class QLearner:
    """Value-iteration with epsilon-greedy strategy."""

    def __init__(self, mdp: FrozenLake, updates_per_second=2):
        # Hyper parameters
        self.SNAPSHOT_STATES = 2000
        self.EPISODES = 100000
        self.LEARNING_RATE = 0.8  # alpha
        self.DISCOUNT_FACTOR = 0.95  # gamma, decays with path length.
        self.EPSILON = 0.9
        self.updates_per_second = updates_per_second
        self.mdp = mdp
        self.Q = None
        self.state_visits = []
        self.state_counter = None
        self.size = None
        self.steps = 0
        self.max_visit = 0
        self.reset()
        self.CALLBACK_INTERVAL = 2000

    def update_hyperparams(self, alpha=None, gamma=None, epsilon=None):
        if alpha is not None:
            self.LEARNING_RATE = alpha

        if gamma is not None:
            self.DISCOUNT_FACTOR = gamma

        if epsilon is not None:
            self.EPSILON = epsilon

    def inc_epsilon(self, delta=0.05):
        self.EPSILON = min(self.EPSILON + delta, 0.99)

    def dec_epsilon(self, delta=0.05):
        self.EPSILON = max(self.EPSILON - delta, 0.01)

    def inc_updates(self):
        self.updates_per_second += 1

    def dec_updates(self):
        self.updates_per_second = max(1, self.updates_per_second - 1)

    def inc_alpha(self):
        self.LEARNING_RATE = min(0.99, self.LEARNING_RATE + 0.05)

    def dec_alpha(self):
        self.LEARNING_RATE = max(0.01, self.LEARNING_RATE - 0.05)

    def inc_gamma(self):
        self.DISCOUNT_FACTOR = min(0.99, self.DISCOUNT_FACTOR + 0.05)

    def dec_gamma(self):
        self.DISCOUNT_FACTOR = max(0.01, self.DISCOUNT_FACTOR - 0.05)

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
        # Non-zero values converge faster.
        for s in range(len(self.Q)):
            for a in range(len(self.Q[s, :])):
                self.Q[s, a] = 0.01
        self.state_visits = []
        self.state_counter = np.zeros(self.mdp.state_count(), dtype=np.uint64)
        self.size = int(math.sqrt(self.mdp.state_count()))
        self.steps = 0
        self.max_visit = 0

    def state_occupation(self, s):
        return self.state_counter[s] / self.max_visit

    def state_counter_snapshot(self):
        self.state_visits.append(np.reshape(self.state_counter.copy(), (self.size, self.size)))

    def train(self, callback=None):
        self.reset()
        running_reward = None
        for i in range(self.EPISODES):
            s = self.mdp.reset()
            done = False
            # invoke_callback = (i % self.CALLBACK_INTERVAL == 0) and (callback is not None)
            invoke_callback = callback is not None
            episode_steps = 0
            rewards = 0
            while not done:
                episode_steps += 1
                if self.steps % self.SNAPSHOT_STATES == 0:
                    self.state_counter_snapshot()

                self.state_counter[s] += 1
                # for statistics, i.e. heat-map
                self.max_visit = max(self.state_counter[s], self.max_visit)
                a = self.epsilon_greedy(self.Q, s)
                s_next, r, done, info = self.mdp.step(a)

                self.Q[s, a] += self.LEARNING_RATE * (r + self.DISCOUNT_FACTOR * np.max(self.Q[s, :]) - self.Q[s, a])
                # next_q = r + self.DISCOUNT_FACTOR * np.max(self.Q[s_next, :])
                # self.Q[s, a] = ((1 - self.LEARNING_RATE) * self.Q[s, a]) + (self.LEARNING_RATE * next_q)

                s = s_next
                rewards += r
                self.steps += 1
                running_reward = rewards if running_reward is None else running_reward * 0.99 + rewards * 0.01

                if invoke_callback:
                    continue_training = callback(self.mdp.move[a], self.steps, rewards, i, self.EPISODES, episode_steps)
                    if not continue_training:
                        return

                if self.updates_per_second > 0:
                    time.sleep(1 / self.updates_per_second)

            if i % self.SNAPSHOT_STATES == 0:
                # Snapshot current counter for animation later
                self.state_counter_snapshot()

        self.state_counter_snapshot()

    def save(self):
        filename = f"frozenlake_q_table"
        np.save(filename, self.Q)
        print(repr(self.Q))


if __name__ == "__main__":
    from env.custom_frozenlake import FrozenLake

    q = QLearner(FrozenLake(), updates_per_second=0)
    max_r = 0


    def update(move, steps, rewards, i, episodes, episode_steps, running_reward):
        global max_r
        max_r = max(rewards, max_r)
        if running_reward > 0:
            print(max_r, rewards, round(rewards, 2), running_reward, i)
        return True


    q.train(update)
