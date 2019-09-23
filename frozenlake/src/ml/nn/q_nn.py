import tensorflow.compat.v1 as tf
import numpy as np
import gym

tf.disable_eager_execution()
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
)

env = gym.make("FrozenLakeNotSlippery-v0")

# Define network
input_s = tf.placeholder(tf.float32, shape=(1, 16))
w = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))
output_Q = tf.matmul(input_s, w)
predicted_action = tf.argmax(output_Q, 1)

# Loss
next_Q = tf.placeholder(tf.float32, shape=(1, 4))
loss = tf.reduce_sum(tf.square(next_Q - output_Q))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# Train
init = tf.global_variables_initializer()

DISCOUNT_FACTER = 0.99
EPSILON = 0.1
EPISODES = 20000

with tf.Session() as session:
    session.run(init)

    for i in range(EPISODES):
        s = env.reset()
        done = False

        while not done:
            next_input = np.identity(16)[s:s + 1]  # one hot vector
            a, all_q = session.run([predicted_action, output_Q], feed_dict={input_s: next_input})

            if np.random.rand(1) < EPSILON:
                a[0] = env.action_space.sample()

            s_next, r, done, _ = env.step(a[0])

            next_input = np.identity(16)[s_next:s_next + 1]
            q_next = session.run(output_Q, feed_dict={input_s: next_input})

            max_q_next = np.max(q_next)
            target_q = all_q
            target_q[0, a[0]] = r + DISCOUNT_FACTER * max_q_next
            _, w1 = session.run([train, w], feed_dict={input_s: np.identity(16)[s:s + 1], next_Q: target_q})

            s = s_next

    # Result
    for i in range(16):
        next_input = np.identity(16)[i:i + 1]
        all_q = session.run([output_Q], feed_dict={input_s: next_input})
        print(i, all_q)
