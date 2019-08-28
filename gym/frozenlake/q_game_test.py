import numpy as np
from game import FrozenLakeGame
from custom_frozenlake import FrozenLake
import time

# Space for 5x5x4
Q_sample = np.array(
       [[0.10737418, 0.13421773, 0.       , 0.10737418],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.13421773, 0.16777216, 0.        , 0.10737418],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.1164461 , 0.00821924, 0.        ],
       [0.03424683, 0.09830178, 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.16777216, 0.2097152 , 0.        , 0.13421773],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.26656423, 0.02014519],
       [0.14012601, 0.40959963, 0.14372615, 0.02510437],
       [0.25606269, 0.51190191, 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.2097152 , 0.262144  , 0.262144  , 0.16777216],
       [0.2097152 , 0.32768   , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.512     , 0.512     , 0.32767506],
       [0.40959998, 0.64      , 0.        , 0.40930653],
       [0.        , 0.        , 0.        , 0.        ],
       [0.262144  , 0.        , 0.32768   , 0.2097152 ],
       [0.262144  , 0.4096    , 0.4096    , 0.262144  ],
       [0.32768   , 0.512     , 0.512     , 0.        ],
       [0.4096    , 0.64      , 0.64      , 0.4096    ],
       [0.512     , 0.8       , 0.        , 0.512     ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.4096    , 0.512     , 0.32768   ],
       [0.4096    , 0.512     , 0.64      , 0.4096    ],
       [0.512     , 0.64      , 0.8       , 0.512     ],
       [0.64      , 0.8       , 1.        , 0.64      ],
       [0.        , 0.        , 0.        , 0.        ]])

game = FrozenLakeGame(800, 600)
game.start()
lake = FrozenLake()

Q = np.load('frozenlake_q_table.npy')

done = False
s = lake.reset()
start = s
path = []
steps = 0
while not done:
       game.left()
       steps += 1
       # take the action with the highest reward.
       a = np.argmax(Q[s, :])
       # x, y = navigate(a, x, y)
       # print(move[a], x, y)
       old_s = s
       # take action a and return new state s
       s, r, done, info = lake.step(a)
       #time.sleep(1)
       transition = (old_s, lake.move[a], s)
       path.append(transition)

print(steps, "start:", start, path)