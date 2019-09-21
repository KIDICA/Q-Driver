import tensorflow as tf
import numpy as np
from custom_frozenlake import FrozenLake

lake = FrozenLake()

tf.compat.v1.disable_eager_execution()

# Define network
input_s = tf.compat.v1.placeholder(tf.float32, shape=(1, 16))
w = tf.Variable(tf.random.uniform([16, 4], 0, 0.1))
output_Q = tf.matmul(input_s, w)
predicted_action = tf.argmax(input=output_Q, axis=1)

# Loss
next_Q = tf.compat.v1.placeholder(tf.float32, shape=(1, 4))
loss = tf.reduce_sum(input_tensor=tf.square(next_Q - output_Q))

# Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# Train
init = tf.compat.v1.global_variables_initializer()

DISCOUNT_FACTER = 0.99
EPSILON = 0.1
EPISODES = 20000

with tf.compat.v1.Session() as session:
    session.run(init)

    for i in range(EPISODES):
        s = lake.reset()
        done = False

        while not done:
            next_input = np.identity(16)[s:s + 1]  # one hot vector
            a, all_q = session.run([predicted_action, output_Q], feed_dict={input_s: next_input})

            if np.random.rand(1) < EPSILON:
                a[0] = lake.action_sample()

            s_next, r, done, _ = lake.step(a[0])

            next_input = np.identity(16)[s_next:s_next + 1]
            print("out", output_Q)
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
