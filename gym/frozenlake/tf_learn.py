import os
# disable GPU initialization messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from .custom_frozenlake import FrozenLake

lake = FrozenLake()

# tf.Variable(tf.random.uniform([lake.state_count(), lake.action_count()], 0.0, 0.01))

# Details see: https://hunkim.github.io/ml/RL/rl06.pdf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, lake.state_count())),
    tf.keras.layers.Dense(lake.state_count(), activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(lake.action_count(), activation='softmax')
])

total_states = lake.state_count() + lake.action_count()

# Q-Function approximator: q: S x A -> R
# The input is a vector of the current state and action
# i.e.: [1, 0, ..., 1, 0, 0, 0] means state=0,action=0
# last 4 elements are
q_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, total_states)),
    tf.keras.layers.Dense(total_states, activation='relu'),
    tf.keras.layers.Dense(1)
])
# The q function returns the rewards for a given action within a state.
# i.e. (0, LEFT) -> 0.3 means reward of moving LEFT from state 0.

# Will q* will return the optimal action given a state
# Q*(s,a|s), this needs to use q_model to approximate the
# optimal action for each state.
q_opt_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, lake.state_count())),
    tf.keras.layers.Dense(lake.state_count(), activation='relu'),
    tf.keras.layers.Dense(lake.action_count(), activation='softmax')
])
