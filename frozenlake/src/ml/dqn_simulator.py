import gym
from gym.envs.registration import register
import numpy as np

from ml.dqn import DeepTDLambdaLearner

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


def tweak_reward(_reward, _done):
    if _reward == 0:
        _reward = -0.01
    if done:
        if _reward < 1:
            _reward = -1
    return reward


def package_state(_state, state_count):  # so that we can feed it into the tensorflow graph
    _state = convert_to_one_hot(_state, state_count)
    _state = _state.reshape(1, -1)
    return _state


def convert_to_one_hot(state_number, _n_states):
    _state = np.zeros((1, _n_states))
    _state[0][state_number] = 1
    return _state


episodes = 1000

env = gym.make('FrozenLakeNotSlippery-v0')
n_actions = env.action_space.n
n_states = env.observation_space.n

agent = DeepTDLambdaLearner(n_actions=n_actions, n_states=n_states)

# Iterate the game
for e in range(episodes):
    state = env.reset()
    state = package_state(state, n_states)

    total_reward = 0
    done = False
    while not done:
        action, greedy = agent.get_e_greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()

        next_state = package_state(next_state, n_states)

        # Tweaking the reward to help the agent learn faster
        tweaked_reward = tweak_reward(reward, done)

        agent.learn(state, action, next_state, tweaked_reward, greedy)

        state = next_state
        total_reward += tweaked_reward

        if done:
            if reward == 1:
                print("episode: {}/{}, score: {:.2f} and goal has been found!".format(e, episodes, total_reward))
            else:
                print("episode: {}/{}, score: {:.2f}".format(e, episodes, total_reward))
            break

    agent.reset_e_trace()
# env.close()
