import numpy as np
from custom_frozenlake import FrozenLake
from q_learn import QLearner, Drawer
import time

q = QLearner(FrozenLake())
q.train()
q.save()

d = Drawer(q)
d.save()

print(repr(q.Q))

lake = FrozenLake()

Q = np.load('frozenlake_q_table.npy')
print(repr(Q))

done = False
s = lake.reset()
start = s
lake.render()
path = []
steps = 0
while not done:
       steps += 1
       # take the action with the highest reward.
       a = np.argmax(Q[s, :])
       # x, y = navigate(a, x, y)
       # print(move[a], x, y)
       old_s = s
       # take action a and return new state s
       s, r, done, info = lake.step(a)
       lake.render()
       time.sleep(1)
       transition = (old_s, lake.move[a], s)
       path.append(transition)

print(steps, "start:", start, path)
