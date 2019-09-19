import argparse
import numpy as np
import gym
import tensorflow as tf
import tensorlayer as tl
from gym.envs.registration import register
import time

"""
This Deep Q-Network convergence is terrible. Need to fine tune it.
"""

"""
Deep Q-Network Q(a, s)
-----------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).

Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)

See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.

Reference
----------
original paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327

Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py

Environment
-----------
# The FrozenLake v0 environment
https://gym.openai.com/envs/FrozenLake-v0
The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially
depends on the chosen direction. The agent is rewarded for finding a walkable
path to a goal tile.
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward
of 1 if you reach the goal, and zero otherwise.

Prerequisites
--------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0

To run
-------
python tf_dqn_frozen.py --train/test/load/render


"""

# FrozenLake: https://github.com/openai/gym/wiki/FrozenLake-v0
# This is a customized deterministic version of Frozen lake.
# Otherwise you would randomly move around and that's not optimal for presentation
# because it would take veeeeeeeeeeeeeeeeeeeeeery long time to learn those parameters.
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--render', dest='render', action='store_true', default=False)
parser.add_argument('--load', dest='load', action='store_true', default=False)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

#  hyper parameters
lambd = .9  # decay factor
e = 0.7  # e-Greedy Exploration, the larger the more random
num_episodes = 200000
render = args.render  # display the game environment
running_reward = None
load = args.load
MAP = None
STATE_COUNT = 4*4 #len(''.join(MAP))


def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


def get_model(inputs_shape):
    """
    Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
    encoding for state: nxp grid can be represented by one-hot vector with n*p integers.
    """
    ni = tl.layers.Input(inputs_shape, name='observation')
    no = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.1), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=no, name="Q-Network")


def save_ckpt(model):  # save trained weights
    tl.files.save_npz(model.trainable_weights, name='dqn_model.npz')


def load_ckpt(model):  # load trained weights
    tl.files.load_and_assign_npz(name='dqn_model.npz', network=model)


if __name__ == '__main__':

    q_network = get_model([None, STATE_COUNT])
    q_network.train()
    train_weights = q_network.trainable_weights

    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    env = gym.make('FrozenLakeNotSlippery-v0')

    if args.train:
        t0 = time.time()
        for i in range(num_episodes):
            # Reset environment and get first new observation
            # episode_time = time.time()
            s = env.reset()  # observation is state, integer 0 ~ 15
            rewards = 0
            for j in range(99):  # step index, maximum step is 99
                # if render: env.render()
                # Choose an action by greedily (with e chance of random action) from the Q-network
                q_values = q_network(np.asarray([to_one_hot(s, STATE_COUNT)], dtype=np.float32)).numpy()
                a = np.argmax(q_values, 1)

                # e-Greedy Exploration !!! sample random action
                if np.random.uniform() > e:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                # Obtain the Q' values by feeding the new state through our network
                Q1 = q_network(np.asarray([to_one_hot(s1, STATE_COUNT)], dtype=np.float32)).numpy()

                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                targetQ = q_values
                targetQ[0, a[0]] = r + lambd * maxQ1
                # Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                with tf.GradientTape() as tape:
                    _qvalues = q_network(np.asarray([to_one_hot(s, STATE_COUNT)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)

                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                rewards += r
                s = s1
                # Reduce chance of random action if an episode is done.
                if d:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break

            # The rewards with random action
            running_reward = rewards if running_reward is None else running_reward * 0.99 + rewards * 0.01
            # print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
            #     (i, num_episodes, rAll, running_reward, time.time() - episode_time))
            if i % 1000 == 0:
                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}' \
                        .format(i, num_episodes, rewards, running_reward, time.time() - t0))
            if running_reward > 0.8:
                break
        save_ckpt(q_network)  # save model

    if args.test:
        t0 = time.time()
        load_ckpt(q_network)  # load model
        for i in range(num_episodes):
            # Reset environment and get first new observation
            episode_time = time.time()
            s = env.reset()  # observation is state, integer 0 ~ 15
            rewards = 0
            for j in range(99):  # step index, maximum step is 99
                if render: env.render()
                # Choose an action by greedily (with e chance of random action) from the Q-network
                q_values = q_network(np.asarray([to_one_hot(s, STATE_COUNT)], dtype=np.float32)).numpy()
                a = np.argmax(q_values, 1)  # no epsilon, only greedy for testing

                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rewards += r
                s = s1
                # Reduce chance of random action if an episode is done.
                if d:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break

            # Note that, the rewards here with random action
            running_reward = rewards if running_reward is None else running_reward * 0.99 + rewards * 0.01

            print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(i, num_episodes, rewards, running_reward, time.time() - t0))

    if load:
        t0 = time.time()
        load_ckpt(q_network)  # load model
        steps = 0

        # Reset environment and get first new observation
        episode_time = time.time()
        s = env.reset()  # observation is state, integer 0 ~ 15
        rewards = 0
        done = False
        while True:
            steps += 1
            if render:
                env.render()

            # Choose an action by greedily (with e chance of random action) from the Q-network
            q_values = q_network(np.asarray([to_one_hot(s, STATE_COUNT)], dtype=np.float32)).numpy()
            a = np.argmax(q_values, 1)  # no epsilon, only greedy for testing

            # Get new state and reward from environment
            s1, r, done, _ = env.step(a[0])
            rewards += r
            s = s1

            if done:
                env.render()
                break
            time.sleep(1 / 2)

        print('Steps: {:d} | Running Time: {:.4f}'.format(steps, time.time() - t0))
