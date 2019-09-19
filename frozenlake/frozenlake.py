from random import randrange
import sys

class FrozenLake:
    """
    Custom implementation of FrozenLake non-slippery for better control of the implementation 
    and the unnecessary use of the entire gym framework to test this game.

    S: Start
    F: Frozen
    H: Hole
    G: Goal
    P: Player

    Example coordinates for a 4x4 Grid:

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
    def __init__(self, map=[
                "SFHFFFF",
                "FHFFFFF",
                "FFFFFFF",
                "FFFHHHF",
                "FFFFFHF",
                "FFFFFHF",
                "FHFFFHG"
            ], random_map = True):
        self.set_map(map)

        # actions
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.action_name = ['LEFT', 'RIGHT', 'UP', 'DOWN']

        self.START = 'S'
        self.HOLE = 'H'
        self.FROZEN = 'F'
        self.GOAL = 'G'
        self.PLAYER = 'P'

        self.reward = {
            'S': 0.0,
            'F': 0.0,
            'H': -1.0,
            'G': 1.0
        }

        self.reset()

    def reset(self):
        self.state = 0
        self.x = 0
        self.y = 0

    def set_map(self, map):
        self.map = map
        self.flat_map = ''.join(map)
        # Start state is the index of 'S'
        self.state = self.flat_map.find('S')
        self.size = len(self.flat_map)

        self.max_height_index = len(map) - 1
        self.max_width_index = len(map[0]) - 1

        self.rows = len(self.map)
        self.cols = len(self.map[0])

    def act(self, action):
        y = self.y
        x = self.x
        state = self.state
        if action == self.UP:
            y = max(y - 1, 0)
            if y != self.y: state -= self.cols
        elif action == self.DOWN:
            y = min(y + 1, self.max_height_index)
            if y != self.y: state += self.cols
        elif action == self.LEFT:
            x = max(0, x - 1)
            if x != self.x: state -= 1
        elif action == self.RIGHT:
            x = min(self.max_width_index, x + 1)
            if x != self.x: state += 1
        return x, y, state

    def action_sample(self):
        return randrange(0, self.action_count() - 1)

    def state_count(self):
        return self.size

    def action_count(self):
        return len(self.action_name)

    def step(self, action):
        self.x, self.y, self.state = self.act(action)
        s = self.flat_map[self.state]
        done = s == self.GOAL
        m = self.map[self.y][self.x]
        if s == self.HOLE: self.reset()
        return m, self.reward[m], done

    def render(self):
        for y in range(self.rows):
            for x in range(self.cols):
                if self.y == y and self.x == x:
                    sys.stdout.write(self.PLAYER)
                else:
                    sys.stdout.write(self.map[y][x])
            sys.stdout.write('\n')
        sys.stdout.write('\n')

    def info(self):
        return self.x, self.y, self.state, self.map[self.y][self.x], self.map[self.y][self.x] == self.GOAL