import numpy as np

import gym
import tensorflow as tf
import tensorlayer as tl

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
import time

def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a

def get_model(inputs_shape):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.1), b_init=None, name='q_a_s_2')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def load_ckpt(model):  # load trained weights
    tl.files.load_and_assign_npz(name='dqn_model.npz', network=model)



env = gym.make('FrozenLakeNotSlippery-v0')

qnetwork = get_model([None, 16])
qnetwork.is_train = False
load_ckpt(qnetwork)  # load model
s = env.reset()

done = False
steps = 0
while not done:
    steps += 1
    q_row = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
    a = np.argmax(q_row, 1)
    s, r, done, _ = env.step(a[0])
    print(f"Steps={steps}")
    env.render()
    #time.sleep(1)
