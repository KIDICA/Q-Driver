import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make("MsPacman-v0")

# Hyperparameter

MAX_EPISODES = 800
MAX_EPISODE_LENGTH = 18000
MINIBATCH_SIZE = 32  # org. 32
REPLAY_MEMORY_SIZE = 20000  # org. 10000000
AGENT_HISTORY_LENGTH = 1  # org. 4
LEARNING_RATE = 0.0000625  # org. 0.00025
DISCOUNT_FACTOR = 0.99  # org. 0.99

TARGET_NETWORK_UPDATE_FREQUENCY = 100  # org. 100000
REPLAY_START_SIZE = 200  # org. 50000
UPDATE_FREQUENCY = 4  # org. 4

NUM_ACTION = env.action_space.n
INPUT_SHAPE = (None, 84, 84, AGENT_HISTORY_LENGTH)

# Preprocessing

preprocess_input = tf.placeholder(shape=[210, 160, 3], dtype=tf.float32)  # dtype=tf.uint8)

p_frame = tf.image.rgb_to_grayscale(preprocess_input)
p_frame = tf.image.crop_to_bounding_box(p_frame, 0, 0, 176, 160)
p_frame = tf.image.resize_images(p_frame, [84, 84],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

preprocess_op = (p_frame - 128.0) / 128.0 - 1


def preprocessing(input_frame):
    return session.run(preprocess_op, feed_dict={preprocess_input: input_frame})


class Network:
    def __init__(self, scope_name):
        self.input_s = tf.placeholder(tf.float32, shape=INPUT_SHAPE)

        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(scope_name) as scope:
            # Inspired by Network from DQN (Mnih 2015)

            conv_layer_1 = Conv2D(self.input_s, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
                                  weights_initializer=initializer)
            conv_layer_2 = Conv2D(conv_layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
                                  weights_initializer=initializer)
            conv_layer_3 = Conv2D(conv_layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
                                  weights_initializer=initializer)

            flat = Flatten(conv_layer_3)

            fc_layer_4 = Dense(flat, num_outputs=128, weights_initializer=initializer)
            fc_layer_5 = Dense(fc_layer_4, num_outputs=NUM_ACTION, activation_fn=None, weights_initializer=initializer)

            self.output_q = fc_layer_5

            self._init_vars(scope_name)

    def _init_vars(self, scope_name):
        # vars enthalten die Gewichte
        self.vars = tf.trainable_variables(scope=scope_name)

    def add_learning(self, target):
        self.input_action = tf.placeholder(tf.int32, shape=(None,))
        self.target_q = tf.reduce_sum(target.output_q * tf.one_hot(self.input_action, NUM_ACTION), axis=-1,
                                      keepdims=True)

        self.current_output = tf.placeholder(tf.float32, shape=(None, 1))

        # Loss Function - Part 2
        self.loss = tf.reduce_mean(tf.square(self.current_output - self.target_q))

        # Optimierer -  Part 3
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.train_op = optimizer.minimize(self.loss)


# Epsilon Greedy

INITIAL_EXPLORATION = 1.0  # org. 1.0
FINAL_EXPLORATION = 0.1  # org. 0.1
FINAL_EXPLORATION_FRAME = 1000000  # org. 1000000


def eps_greedy(best_action, frame):
    epsilon = FINAL_EXPLORATION

    if frame < FINAL_EXPLORATION_FRAME:
        epsilon = (FINAL_EXPLORATION - INITIAL_EXPLORATION) / FINAL_EXPLORATION_FRAME * frame + INITIAL_EXPLORATION

    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTION)
    else:
        return best_action


# Replay Memory

from collections import deque
from random import randint


class Replay_memory:
    def __init__(self, capacity):
        self._states = deque(maxlen=capacity)
        self._actions = deque(maxlen=capacity)
        self._next_states = deque(maxlen=capacity)
        self._reward = deque(maxlen=capacity)
        self._dones = deque(maxlen=capacity)

    def append(self, s, a, next_s, r, done):
        self._states.append(s)
        self._actions.append(a)
        self._next_states.append(next_s)
        self._reward.append(r)
        self._dones.append(done)

    def get_batch(self, batch_size):
        s = []
        a = np.empty(batch_size, dtype=np.int32)
        s_next = []
        r = np.empty(batch_size, dtype=np.float32)
        done = np.empty(batch_size, dtype=np.bool)

        for i in range(batch_size):
            x = randint(0, len(self._states) - 1)
            s.append(self._states[x])
            a[i] = self._actions[x]
            s_next.append(self._next_states[x])
            r[i] = self._reward[x]
            done[i] = self._dones[x]

        return s, a, s_next, r, done


# Copy Operator current --> target

def copy_vars_to_target(session, current_vars, target_vars):
    for i, var in enumerate(current_vars):
        copy_op = target_vars[i].assign(var.value())
        session.run(copy_op)


# Statistics

class Statistics:
    def __init__(self, write_to_file=True):
        self._average_reward = deque(maxlen=100)
        self._write_to_file = write_to_file

        if self._write_to_file:
            with open('statistics.txt', 'a') as f:
                f.write('EPISODE FRAME REWARD AVERAGE_REWARD AVERAGE_LOSS\n')

    def append(self, episode, frame_number, episode_frame_number, episode_reward, episode_loss):
        self._average_reward.append(episode_reward)
        average_reward = sum(self._average_reward) / len(self._average_reward)
        # print(
        #    f'episode {episode} frame {frame_number} episode_frame {episode_frame_number} reward {episode_reward:.2f} average reward {average_reward:.2f} episode loss {episode_loss:.2f}')

        # if self._write_to_file:
        #    with open('statistics.txt', 'a') as f:
        #        f.write(f'{episode} {frame_number} {episode_reward:.2f} {average_reward:.2f} {episode_loss:.2f}\n')


stat = Statistics()

replay_memory = Replay_memory(REPLAY_MEMORY_SIZE)

current = Network('current')
target = Network('target')

current.add_learning(target)

# Train
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)

    frame_number = 0

    for i in range(MAX_EPISODES):

        s = env.reset()

        done = False
        episode_frame_number = 0
        episode_reward = 0
        episode_loss = []

        while not done and episode_frame_number <= MAX_EPISODE_LENGTH:

            s = preprocessing(s)

            # Q-Values for s
            actions = current.output_q.eval(feed_dict={current.input_s: [s]})
            best_action = np.argmax(actions, axis=-1)

            action = eps_greedy(best_action, frame_number)

            s_next, r, done, _ = env.step(action)

            # Memorize
            replay_memory.append(s, action, preprocessing(s_next), r, done)

            # Update after interval.
            if frame_number % UPDATE_FREQUENCY == 0 and frame_number > REPLAY_START_SIZE:
                sample_s, sample_a, sample_next_s, sample_r, sample_done = replay_memory.get_batch(MINIBATCH_SIZE)

                next_a = current.output_q.eval(feed_dict={current.input_s: sample_next_s})
                batch = sample_r + DISCOUNT_FACTOR * np.max(next_a, axis=-1) * (1 - sample_done)

                # Train
                train_loss, _ = session.run([current.loss, current.train_op], feed_dict={current.input_s: sample_s,
                                                                                         target.input_s: sample_s,
                                                                                         current.current_output: np.expand_dims(
                                                                                             batch, axis=-1),
                                                                                         current.input_action: sample_a})
                episode_loss.append(train_loss)

            # copy
            if frame_number % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and frame_number > REPLAY_START_SIZE:
                copy_vars_to_target(session, current.vars, target.vars)

            s = s_next
            episode_frame_number += 1
            frame_number += 1
            episode_reward += r

        stat.append(i, frame_number, episode_frame_number, episode_reward,
                    sum(episode_loss) / len(episode_loss))
