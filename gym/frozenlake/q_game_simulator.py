from game import FrozenLakeGame
from custom_frozenlake import FrozenLake
from q_learn import QLearner

game = FrozenLakeGame(900, 900, QLearner(FrozenLake()))
game.start()
