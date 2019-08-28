import gym
from gym.envs.registration import register

# FrozenLake: https://github.com/openai/gym/wiki/FrozenLake-v0
# This is a customized deterministic version of Frozen lake.
# Otherwise you would randomly move around and that's not optimal for presentation
# because it would take veeeeeeeeeeeeeeeeeeeeeery long time to learn those parameters.
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


class FrozenLake:
    """
    Example coordinates for a 4x4 Grid.
    +----+----+----+----+
    |  0 |  1 |  2 |  3 |
    +----+----+----+----+
    |  4 |  5 |  6 |  7 |
    +----+----+----+----+
    |  8 |  9 | 10 | 11 |
    +----+----+----+----+
    | 12 | 13 | 14 | 15 |
    +----+----+----+----+
    """

    def __init__(self, game_map=None):
        # location within the grid..
        self.x = 0
        self.y = 0

        self.LEFT = "LEFT"
        self.DOWN = "DOWN"
        self.RIGHT = "RIGHT"
        self.UP = "UP"

        # Maps action numbers to actual names.
        self.move = [self.LEFT, self.DOWN, self.RIGHT, self.UP]

        if game_map is not None \
                :
            self.MAP = game_map
        else:
            self.MAP = [
                "SFHFFFF",
                "FHFFFFF",
                "FFFFFFF",
                "FFFHHHF",
                "FFFFFHF",
                "FFFFFHF",
                "FHFFFHG"
            ]
            self.flat_map = ''.join(self.MAP)
            self.MAP2 = [
                "SHFHHFFF",
                "FHFHFFFH",
                "FHFHFFFH",
                "FFFHFFFH",
                "FHFFFFFH",
                "FHFFFFFH",
                "FFFHFFFH",
                "HFFHFFFG"
            ]

        self.size = len(self.MAP)

        self.env = gym.make("FrozenLakeNotSlippery-v0", desc=self.MAP)
        self.env.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)

    def state_count(self):
        return self.env.observation_space.n

    def action_count(self):
        return self.env.action_space.n

    def action_sample(self):
        return self.env.action_space.sample()

    def navigate(self, action, x, y):
        move_type = self.move[action]
        if move_type == self.LEFT:
            self.x = max(x - 1, 0)
        elif move_type == self.RIGHT:
            self.x = min(x + 1, len(self.MAP[0]) - 1)
        elif move_type == self.UP:
            self.y = max(y - 1, 0)
        elif move_type == self.DOWN:
            self.y = min(len(self.MAP) - 1, y)
        return self.x, self.y
